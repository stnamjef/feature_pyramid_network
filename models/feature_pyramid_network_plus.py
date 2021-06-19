import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import utils.array_tool as at
from models.faster_rcnn_base import FasterRCNNBase
from models.utils.backbone_loader import load_vgg16_as_fully_convolutional
from models.rpn.region_proposal_network import _RPNFPN
from utils.config import opt


class FPNPlus(FasterRCNNBase):
    def __init__(self, n_fg_class):
        extractor = load_vgg16_as_fully_convolutional(pool_conv5=True)
        super(FPNPlus, self).__init__(
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
                nn.Linear(7 * 7 * 512, 1024),
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

        # lateral connection
        self.lateral1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.lateral2 = nn.Conv2d(512, 256, 1, 1, 0)
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

        self.n_features = 2

    def _extract_features(self, x):
        features = []
        for i, layer in enumerate(self.extractor):
            x = layer(x)
            if i in [15, 22, 29, 34]:
                features.append(x)

        # lateral connection & top-down pathway
        p6 = self.lateral1(features[3])
        p5 = self._upsample_add(p6, self.lateral2(features[2]))
        p4 = self._upsample_add(p5, self.lateral3(features[1]))
        p3 = self._upsample_add(p4, self.lateral4(features[0]))

        # smooth
        p5 = self.smooth1(p5)
        p4 = self.smooth2(p4)
        p3 = self.smooth3(p3)

        return [p3, p4, p5, p6]

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def _roi_pool(self, features, roi):
        roi = at.totensor(roi).float()
        roi_level = assign_level(roi, self.n_features)

        if self.n_features == 2:
            level_template = [3, 4, 5]
        else:
            level_template = [3, 4]

        roi_pool_feats, box_to_levels = [], []
        for l in level_template:
            if (roi_level == l).sum() == 0:
                continue

            level_idx = t.where(roi_level == l)[0]
            box_to_levels.append(level_idx)

            index_and_roi = t.cat(
                [t.zeros(level_idx.size(0), 1).cuda(), roi[level_idx]],
                dim=1
            )
            # yx -> xy
            index_and_roi = index_and_roi[:, [0, 2, 1, 4, 3]].contiguous()

            temp = []
            for i in range(l - 3, l - 3 + self.n_features):
                feat = tv.ops.roi_pool(
                    features[i],
                    index_and_roi,
                    self.pooling_size,
                    self.spatial_scale[i]
                )
                temp.append(feat)

            roi_pool_feats.append(t.cat(temp, dim=1))

        roi_pool_feats = t.cat(roi_pool_feats, dim=0)
        box_to_level = t.cat(box_to_levels, dim=0)
        idx_sorted, order = t.sort(box_to_level)
        roi_pool_feats = roi_pool_feats[order]

        return roi_pool_feats

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


def assign_level(roi, n_features):
    h = roi.data[:, 2] - roi.data[:, 0] + 1
    w = roi.data[:, 3] - roi.data[:, 1] + 1
    roi_level_float = t.log2(t.sqrt(h * w) / 224.) + 5

    roi_level = []
    if n_features == 2:
        for l in roi_level_float:
            if l < 4:
                roi_level.append(3)
            elif l < 5:
                roi_level.append(4)
            else:
                roi_level.append(5)
    elif n_features == 3:
        for l in roi_level_float:
            if l < 4.5:
                roi_level.append(3)
            else:
                roi_level.append(4)
    else:
        return ValueError
    return t.tensor(roi_level)