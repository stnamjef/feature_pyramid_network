import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import utils.array_tool as at
from models.faster_rcnn_base import FasterRCNNBase
from models.utils.backbone_loader import load_resnet101
from models.utils.backbone_loader import load_vgg16_as_fully_convolutional
from models.rpn.region_proposal_network import _RPNFPN
from utils.config import opt


class FPN(FasterRCNNBase):
    def __init__(self, n_fg_class):
        if opt.backbone == 'vgg16':
            extractor = load_vgg16_as_fully_convolutional(pretrained=True)
        elif opt.backbone == 'resnet101':
            extractor = load_resnet101(pretrained=True)
        else:
            raise ValueError('Invalid backbone network')
        super(FPN, self).__init__(
            n_class=n_fg_class + 1,
            extractor=extractor,
            rpn=_RPNFPN(
                scales=[64, 128, 256, 512],
                ratios=[0.5, 1, 2],
                rpn_conv=nn.Conv2d(256, 512, 3, 1, 1),
                rpn_loc=nn.Conv2d(512, 3 * 4, 1, 1),
                rpn_score=nn.Conv2d(512, 3 * 2, 1, 1)
            ),
            top_layer=nn.Sequential(
                nn.Linear(7 * 7 * 256 * opt.n_features, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True)
            ),
            loc=nn.Linear(1024, (n_fg_class + 1) * 4),
            score=nn.Linear(1024, n_fg_class + 1),
            spatial_scale=[1/4., 1/8., 1/16., 1/32.],
            pooling_size=7,
            roi_sigma=opt.roi_sigma
        )
        normal_init(self.top_layer[0], 0, 0.01)
        normal_init(self.top_layer[2], 0, 0.01)
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.backbone = opt.backbone
        self.n_features = opt.n_features

        if self.backbone == 'vgg16':
            # FPN params
            self.extraction_point = [15, 22, 29, 34]
            self.default_level = 5
            # lateral connection
            self.lateral1 = nn.Conv2d(1024, 256, 1, 1, 0)
            self.lateral2 = nn.Conv2d(512, 256, 1, 1, 0)
            self.lateral3 = nn.Conv2d(512, 256, 1, 1, 0)
            self.lateral4 = nn.Conv2d(256, 256, 1, 1, 0)
        else:
            # FPN params
            self.extraction_point = [1, 2, 3, 4]
            self.default_level = 4
            # lateral connection
            self.lateral1 = nn.Conv2d(2048, 256, 1, 1, 0)
            self.lateral2 = nn.Conv2d(1024, 256, 1, 1, 0)
            self.lateral3 = nn.Conv2d(512, 256, 1, 1, 0)
            self.lateral4 = nn.Conv2d(256, 256, 1, 1, 0)
        normal_init(self.lateral1, 0, 0.01)
        normal_init(self.lateral2, 0, 0.01)
        normal_init(self.lateral3, 0, 0.01)
        normal_init(self.lateral4, 0, 0.01)

        # smooth layer
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        normal_init(self.smooth1, 0, 0.01)
        normal_init(self.smooth2, 0, 0.01)
        normal_init(self.smooth3, 0, 0.01)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if self.backbone == 'resnet101' and mode:
            # Set fixed blocks to be in eval mode
            self.extractor[0].eval()
            self.extractor[1].eval()
            self.extractor[2].train()
            self.extractor[3].train()
            self.extractor[4].train()

            self.smooth1.train()
            self.smooth2.train()
            self.smooth3.train()

            self.lateral1.train()
            self.lateral2.train()
            self.lateral3.train()
            self.lateral4.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.extractor[0].apply(set_bn_eval)
            self.extractor[1].apply(set_bn_eval)
            self.extractor[2].apply(set_bn_eval)
            self.extractor[3].apply(set_bn_eval)
            self.extractor[4].apply(set_bn_eval)

    def _extract_features(self, x):
        features = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in self.extraction_point:
                features.append(x)

        # lateral connection & top-down pathway
        p6 = self.lateral1(features[3])
        p5 = self._upsample_add(p6, self.lateral2(features[2]))
        p4 = self._upsample_add(p5, self.lateral3(features[1]))
        p3 = self._upsample_add(p4, self.lateral4(features[0]))

        # smoothing
        p5 = self.smooth1(p5)
        p4 = self.smooth2(p4)
        p3 = self.smooth3(p3)

        return [p3, p4, p5, p6]

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def _roi_pool(self, features, roi):
        roi = at.totensor(roi).float()
        # n_features -> the number of features to use for RoI-Pooling
        #               not that of all features
        n_rois = len(roi)
        n_features = self.n_features
        # compute the lowest & highest level
        low = self.default_level - 2
        high = self.default_level + 1
        # compute feature level
        feat_lv = compute_feature_level(
            roi, n_features, self.default_level, low, high
        )
        # possible starting level
        starting_lv = t.arange(low, high + 1)
        if n_features == 2:
            starting_lv = starting_lv[:-1]
        elif n_features == 3:
            starting_lv = starting_lv[:-2]
        # perform RoI-Pooling
        pooled_feats = []
        box_to_levels = []
        for i, l in enumerate(starting_lv):
            if (feat_lv == l).sum() == 0:
                continue

            level_idx = t.where(feat_lv == l)[0]
            box_to_levels.append(level_idx)

            index_and_roi = t.cat(
                [t.zeros(level_idx.size(0), 1).cuda(), roi[level_idx]],
                dim=1
            )
            # yx -> xy
            index_and_roi = index_and_roi[:, [0, 2, 1, 4, 3]].contiguous()

            pooled_feats_l = []
            for j in range(i, i + n_features):
                feat = tv.ops.roi_pool(
                    features[j],
                    index_and_roi,
                    self.pooling_size,
                    self.spatial_scale[j]
                )
                # feat -> n_roi_lx256x7x7
                pooled_feats_l.append(feat)

            pooled_feats.append(t.cat(pooled_feats_l, dim=1))

        pooled_feats = t.cat(pooled_feats, dim=0)
        box_to_level = t.cat(box_to_levels, dim=0)
        idx_sorted, order = t.sort(box_to_level)
        pooled_feats = pooled_feats[order]

        return pooled_feats

    def _bbox_regression_and_classification(self, roi_pool_feat):
        # flatten roi pooled feature
        roi_pool_feat = roi_pool_feat.view(roi_pool_feat.shape[0], -1)

        # RCNN top_layer
        fc9 = self.top_layer(roi_pool_feat)

        # bbox regression & classification
        roi_loc = self.loc(fc9)
        roi_score = self.score(fc9)

        return roi_loc, roi_score


def normal_init(layer, mean, std):
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()


def compute_feature_level(roi, n_features, default_level, low, high):
    h = roi.data[:, 2] - roi.data[:, 0] + 1
    w = roi.data[:, 3] - roi.data[:, 1] + 1
    level = t.log2(t.sqrt(h * w) / 224.) + default_level

    # get lower & upper limit of feature levels
    if n_features == 1:
        level = t.round(level)
        level[level < low] = low
        level[level > high] = high
    elif n_features == 2:
        l1, l2, l3 = low, low + 1, low + 2
        level[level < l2] = l1
        level[(level >= l2) & (level < l3)] = l2
        level[level >= l3] = l3
    elif n_features == 3:
        limit = (low + high) / 2.
        level[level < limit] = low
        level[level >= limit] = low + 1
    else:
        raise NotImplementedError('Not implemented yet.')

    return level
