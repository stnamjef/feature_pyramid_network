import torch as t
import torch.nn as nn
import torch.nn.functional as F
import utils.array_tool as at
from utils.config import opt
from models.rpn.proposal_layer import _ProposalLayer
from models.rpn.proposal_target_layer import _ProposalTargetLayer
from models.rpn.anchor_target_layer import _AnchorTargetLayer
from models.utils.bbox_tools import generate_anchors, generate_anchors_fpn


class _RPNBase(nn.Module):
    def __init__(self, scales, ratios):
        super(_RPNBase, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.proposal_layer = _ProposalLayer(self)
        self.proposal_target_layer = _ProposalTargetLayer()
        self.anchor_target_layer = _AnchorTargetLayer()
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.rpn_sigma = opt.rpn_sigma

    def forward(self, features, img_size, scale, gt_bbox, gt_label):
        raise NotImplementedError


class _RPNFasterRCNN(_RPNBase):
    def __init__(self, scales, ratios, rpn_conv, rpn_loc, rpn_score):
        super(_RPNFasterRCNN, self).__init__(scales, ratios)
        self.n_anchor = len(scales) * len(ratios)
        self.rpn_conv = rpn_conv
        self.rpn_loc = rpn_loc
        self.rpn_score = rpn_score
        normal_init(self.rpn_conv, 0, 0.01)
        normal_init(self.rpn_loc, 0, 0.01)
        normal_init(self.rpn_score, 0, 0.01)

    def forward(self, features, img_size, scale, gt_bbox, gt_label):
        n = 1
        h = F.relu(self.rpn_conv(features))

        loc = self.rpn_loc(h)
        score = self.rpn_score(h)

        h, w = loc.shape[2:]

        loc = loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        score = score.permute(0, 2, 3, 1).contiguous()

        softmax_score = F.softmax(score.view(n, h, w, self.n_anchor, 2), dim=4)
        fg_score = softmax_score[:, :, :, :, 1].contiguous().view(n, -1)

        score = score.view(n, -1, 2)

        feat_shape = (h, w)
        feat_stride = img_size[0] / h
        anchor = generate_anchors(self.scales, self.ratios, feat_shape, feat_stride)

        loc = loc[0]
        score = score[0]
        fg_score = fg_score[0]

        roi = self.proposal_layer(
            loc.cpu().data.numpy(),
            fg_score.cpu().data.numpy(),
            anchor,
            img_size,
            scale
        )

        if self.training:
            # if training phase, then sample RoIs
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(gt_label),
                self.loc_normalize_mean,
                self.loc_normalize_std
            )

            # get gt_loc(offset from anchor to gt_bbox)
            gt_rpn_loc, gt_rpn_label = self.anchor_target_layer(
                at.tonumpy(gt_bbox),
                anchor,
                img_size
            )
            gt_rpn_loc = at.totensor(gt_rpn_loc)
            gt_rpn_label = at.totensor(gt_rpn_label).long()

            # bounding-box regression loss
            rpn_loc_loss = bbox_regression_loss(
                loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma
            )

            # foreground-background classification loss
            rpn_cls_loss = F.cross_entropy(score, gt_rpn_label.cuda(), ignore_index=-1)

            return sample_roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss

        return roi


class _RPNFPN(_RPNBase):
    def __init__(self, scales, ratios, rpn_conv, rpn_loc, rpn_score):
        super(_RPNFPN, self).__init__(scales, ratios)
        self.n_anchor = len(ratios)
        self.rpn_conv = rpn_conv
        self.rpn_loc = rpn_loc
        self.rpn_score = rpn_score
        normal_init(self.rpn_conv, 0, 0.01)
        normal_init(self.rpn_loc, 0, 0.01)
        normal_init(self.rpn_score, 0, 0.01)

    def forward(self, features, img_size, scale, gt_bbox, gt_label):
        n = 1  # batch size is always one
        feat_shapes = []
        locs, scores, fg_scores = [], [], []
        for x in features:
            h = F.relu(self.rpn_conv(x))

            loc = self.rpn_loc(h)
            score = self.rpn_score(h)

            h, w = loc.shape[2:]
            feat_shapes.append((h, w))

            loc = loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
            score = score.permute(0, 2, 3, 1).contiguous()

            softmax_score = F.softmax(score.view(n, h, w, self.n_anchor, 2), dim=4)
            fg_score = softmax_score[:, :, :, :, 1].contiguous().view(n, -1)

            locs.append(loc)
            scores.append(score.view(n, -1, 2))
            fg_scores.append(fg_score)

        loc = t.cat(locs, dim=1)[0]
        score = t.cat(scores, dim=1)[0]
        fg_score = t.cat(fg_scores, dim=1)[0]

        feat_strides = []
        for shape in feat_shapes:
            feat_strides.append(img_size[0] // shape[0])
        anchor = generate_anchors_fpn(self.scales, self.ratios, feat_shapes, feat_strides)

        roi = self.proposal_layer(
            loc.cpu().data.numpy(),
            fg_score.cpu().data.numpy(),
            anchor,
            img_size,
            scale
        )

        if self.training:
            # if training phase, then sample RoIs
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(gt_label),
                self.loc_normalize_mean,
                self.loc_normalize_std
            )

            # get gt_loc(offset from anchor to gt_bbox)
            gt_rpn_loc, gt_rpn_label = self.anchor_target_layer(
                at.tonumpy(gt_bbox),
                anchor,
                img_size
            )
            gt_rpn_loc = at.totensor(gt_rpn_loc)
            gt_rpn_label = at.totensor(gt_rpn_label).long()

            # bounding-box regression loss
            rpn_loc_loss = bbox_regression_loss(
                loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma
            )

            # foreground-background classification loss
            rpn_cls_loss = F.cross_entropy(score, gt_rpn_label.cuda(), ignore_index=-1)

            return sample_roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss

        return roi


def normal_init(layer, mean, std):
    layer.weight.data.normal_(mean, std)
    layer.bias.data.zero_()


def bbox_regression_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()