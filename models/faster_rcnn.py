import torch as t
import torch.nn as nn
import torchvision as tv
import utils.array_tool as at
from utils.config import opt
from models.faster_rcnn_base import FasterRCNNBase
from models.utils.backbone_loader import load_vgg16
from models.rpn.region_proposal_network import _RPNFasterRCNN


class FasterRCNN(FasterRCNNBase):
    def __init__(self, n_fg_class):
        extractor, top_layer = load_vgg16()
        super(FasterRCNN, self).__init__(
            n_class=n_fg_class + 1,
            extractor=extractor,
            rpn=_RPNFasterRCNN(
                scales=[64, 128, 256, 512],
                ratios=[0.5, 1, 2],
                rpn_conv=nn.Conv2d(512, 512, 3, 1, 1),
                rpn_loc=nn.Conv2d(512, 12 * 4, 1, 1),
                rpn_score=nn.Conv2d(512, 12 * 2, 1, 1)
            ),
            top_layer=top_layer,
            loc=nn.Linear(4096, (n_fg_class + 1) * 4),
            score=nn.Linear(4096, n_fg_class + 1),
            spatial_scale=1/16.,
            pooling_size=7,
            roi_sigma=opt.roi_sigma
        )
        normal_init(self.loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def _extract_features(self, x):
        return self.extractor(x)

    def _roi_pool(self, feature, roi):
        index_and_roi = t.cat(
            [t.zeros(roi.shape[0], 1).cuda(), at.totensor(roi).float()],
            dim=1
        )
        # yx -> xy
        index_and_roi = index_and_roi[:, [0, 2, 1, 4, 3]].contiguous()
        # RoI-Pooling
        roi_pool_feat = tv.ops.roi_pool(
            feature,
            index_and_roi,
            self.pooling_size,
            self.spatial_scale
        )

        return roi_pool_feat

    def _bbox_regression_and_classification(self, roi_pool_feat):
        # flatten roi pooled feature
        roi_pool_feat = roi_pool_feat.view(roi_pool_feat.shape[0], -1)

        # Classifier from the base network
        fc7 = self.top_layer(roi_pool_feat)

        # bbox regression & classification
        roi_loc = self.loc(fc7)
        roi_score = self.score(fc7)

        return roi_loc, roi_score


def normal_init(layer, mean, std):
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()