from __future__ import absolute_import

from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.ops import nms
from torchvision.models import vgg16
from torchnet.meter import ConfusionMeter, AverageValueMeter

from utils import array_tool as at
from utils.config import opt
from model.region_proposal_network import RPN
from model.utils.bbox_tools import loc2bbox
from model.utils.creator_tool_fpn import AnchorTargetCreator, ProposalTargetCreator


LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNN(nn.Module):
    def __init__(self, n_fg_class, scales, ratios):
        super(FasterRCNN, self).__init__()

        extractor, classifier = load_vgg16()

        # feature extractor
        self.extractor = extractor
        self.lateral = nn.Conv2d(256, 512, 3, 1, 1)
        normal_init(self.lateral, 0, 0.01)

        # Region proposal network
        self.rpn = RPN(
            in_chs=512,
            mid_chs=512,
            scales=scales,
            ratios=ratios,
            n_anchor=3,
            feat_strides=[4, 8, 16]
        )

        # Region of interest head
        self.head = RoIHead(
            n_class=n_fg_class + 1,
            spatial_scales=[1/4., 1/8., 1/16.],
            classifier=classifier
        )

        # Variables for training start
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # mean and std
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

        # target creator create gt_bbox, gt_label etc as training targets.
        self.proposal_target_layer = ProposalTargetCreator()
        self.anchor_target_layer = AnchorTargetCreator()

        self.optimizer = self.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, x, scale, gt_bboxes, gt_labels, original_size=None, visualize=False):
        if self.training:
            img_size = tuple(x.shape[2:])
            features = []
            for i, layer in enumerate(self.extractor):
                x = layer(x)
                if i in [15, 22, 29]:
                    features.append(x)

            # to make channel 256 -> 512
            features[0] = self.lateral(features[0])

            rpn_locs, rpn_scores, roi, anchor = self.rpn(features, img_size, scale)

            gt_bbox = gt_bboxes[0]
            gt_label = gt_labels[0]
            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]

            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(gt_label),
                self.loc_normalize_mean,
                self.loc_normalize_std
            )

            roi_cls_loc, roi_score = self.head(features, sample_roi)

            # -------------------------- RPN losses -------------------------- #
            gt_rpn_loc, gt_rpn_label = self.anchor_target_layer(
                at.tonumpy(gt_bbox),
                anchor,
                img_size
            )
            gt_rpn_label = at.totensor(gt_rpn_label).long()
            gt_rpn_loc = at.totensor(gt_rpn_loc)

            rpn_loc_loss = fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma
            )

            # NOTE: default value of ignore_index is -100 ...
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
            _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
            _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
            self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

            # ------------------- ROI losses (fast rcnn loss) --------------------#
            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                                  at.totensor(gt_roi_label).long()]
            gt_roi_label = at.totensor(gt_roi_label).long()
            gt_roi_loc = at.totensor(gt_roi_loc)

            roi_loc_loss = fast_rcnn_loc_loss(
                roi_loc.contiguous(),
                gt_roi_loc,
                gt_roi_label.data,
                self.roi_sigma)

            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

            self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
            losses = losses + [sum(losses)]

            return LossTuple(*losses)

        else:
            with torch.no_grad():
                x = at.totensor(x).float()
                img_size = tuple(x.shape[2:])
                features = []
                for i, layer in enumerate(self.extractor):
                    x = layer(x)
                    if i in [15, 22, 29]:
                        features.append(x)

                # to make channel 256 -> 512
                features[0] = self.lateral(features[0])

                rpn_locs, rpn_scores, rois, anchor = self.rpn(features, img_size, scale)

                roi_cls_locs, roi_scores = self.head(features, rois)

                # We are assuming that batch size is 1.
                roi_score = roi_scores.data
                roi_cls_loc = roi_cls_locs.data
                roi = at.totensor(rois) / scale

                # Convert predictions to bounding boxes in image coordinates.
                # Bounding boxes are scaled to the scale of the input images.
                mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
                    repeat(self.n_class)[None]
                std = torch.Tensor(self.loc_normalize_std).cuda(). \
                    repeat(self.n_class)[None]

                roi_cls_loc = (roi_cls_loc * std + mean)
                roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
                roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
                cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                    at.tonumpy(roi_cls_loc).reshape((-1, 4)))
                cls_bbox = at.totensor(cls_bbox)
                cls_bbox = cls_bbox.view(-1, self.n_class * 4)

                # clip bounding box
                cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=original_size[0])
                cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=original_size[1])

                prob = (F.softmax(at.totensor(roi_score), dim=1))

                bbox, label, score = self._suppress(cls_bbox, prob)

                return [bbox], [label], [score]

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @property
    def n_class(self):
        return self.head.n_class

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v, in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def get_optimizer(self):
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer


def load_vgg16(return_indices=False):
    model = vgg16(pretrained=True)
    # the 30th layer of features is relu of conv5_3
    features = list(model.features)[:-1]
    for i in range(len(features)):
        if isinstance(features[i], nn.MaxPool2d) and return_indices:
            features[i] = nn.MaxPool2d(2, stride=2, return_indices=True)
        if i < 10:
            for p in features[i].parameters():
                p.requires_grad = False
    features = nn.Sequential(*features)

    classifier = list(model.classifier)[:6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    return features, classifier


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss


class RoIHead(nn.Module):
    def __init__(self, n_class, spatial_scales, classifier):
        super(RoIHead, self).__init__()
        self.n_class = n_class
        self.spatial_scales = spatial_scales

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.cls_score = nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.cls_loc, 0, 0.01)

    def forward(self, features, rois):
        rois = at.totensor(rois).float()
        H = rois.data[:, 2] - rois.data[:, 0]
        W = rois.data[:, 3] - rois.data[:, 1]
        roi_level = torch.log(torch.sqrt(H * W) / 224.0)
        roi_level = torch.round(roi_level + 5)
        roi_level[roi_level < 3] = 3
        roi_level[roi_level > 5] = 5

        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(3, 6)):
            if (roi_level == l).sum() == 0:
                continue

            idx_l = []
            temp = roi_level == l
            for j, elem in enumerate(temp):
                if elem == True:
                    idx_l.append(j)

            idx_l = torch.tensor(idx_l)
            box_to_levels.append(idx_l)
            scale = self.spatial_scales[i]

            # yx -> xy
            indices_and_rois = torch.cat(
                [torch.zeros(idx_l.size(0), 1).cuda(), rois[idx_l]],
                dim=1
            )
            indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]].contiguous()

            feat = torchvision.ops.roi_pool(features[i], indices_and_rois, (7, 7), scale)
            roi_pool_feats.append(feat)

        roi_pool_feat = torch.cat(roi_pool_feats, dim=0)
        box_to_level = torch.cat(box_to_levels, dim=0)
        idx_sorted, order = torch.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]
        # roi_pool_feat = roi_pool_feat.view(roi_pool_feat.size(0), -1)

        fc7 = self.classifier(roi_pool_feat.view(roi_pool_feat.size(0), -1))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.cls_score(fc7)

        return roi_cls_locs, roi_scores