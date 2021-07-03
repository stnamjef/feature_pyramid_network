import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import utils.array_tool as at
from models.utils.bbox_tools import loc2bbox
from collections import namedtuple
from utils.config import opt

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'])


class FasterRCNNBase(nn.Module):
    def __init__(self, n_class, extractor, rpn, top_layer, loc, score,
                 spatial_scale, pooling_size, roi_sigma):
        super(FasterRCNNBase, self).__init__()
        self.n_class = n_class
        self.extractor = extractor
        self.rpn = rpn
        self.top_layer = top_layer
        self.loc = loc
        self.score = score

        self.spatial_scale = spatial_scale
        self.pooling_size = pooling_size
        self.roi_sigma = roi_sigma

        # variables for eval mode
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.nms_thresh = opt.nms_thresh
        self.score_thresh = opt.score_thresh

    def forward(self, x, scale, gt_bboxes, gt_labels, original_size=None):
        if self.training:
            img_size = tuple(x.shape[2:])

            # Feature extractor from the base network(e.g. VGG16, ResNet-101)
            feature = self._extract_features(x)

            # Region Proposal Network
            rpn_result = self.rpn(feature, img_size, scale, gt_bboxes[0], gt_labels[0])
            roi, gt_roi_loc, gt_roi_label, rpn_loc_loss, rpn_cls_loss = rpn_result

            # RoI Pooling Layer
            roi_pool_feat = self._roi_pool(feature, roi)

            # bbox regression & classification
            roi_loc, roi_score = self._bbox_regression_and_classification(roi_pool_feat)

            # Faster R-CNN loss
            n_sample = roi_loc.shape[0]
            roi_loc = roi_loc.view(n_sample, -1, 4)
            roi_loc = roi_loc[t.arange(0, n_sample).long().cuda(),
                              at.totensor(gt_roi_label).long()]

            gt_roi_loc = at.totensor(gt_roi_loc)
            gt_roi_label = at.totensor(gt_roi_label).long()

            roi_loc_loss = _bbox_regression_loss(
                roi_loc.contiguous(),
                gt_roi_loc,
                gt_roi_label.data,
                self.roi_sigma
            )

            roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label.cuda())

            # Stack losses
            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
            losses = losses + [sum(losses)]

            return LossTuple(*losses)
        else:
            with t.no_grad():
                x = at.totensor(x).float()
                img_size = tuple(x.shape[2:])

                # Feature extractor from the base network(e.g. VGG16, ResNet)
                feature = self._extract_features(x)

                # Region Proposal Network
                roi = self.rpn(feature, img_size, scale, None, None)
                # RoI Pooling Layer
                roi_pool_feat = self._roi_pool(feature, roi)

                # bbox regression & classification
                roi_loc, roi_score = self._bbox_regression_and_classification(roi_pool_feat)

                roi_loc = roi_loc.data
                roi_score = roi_score.data
                roi = at.totensor(roi) / scale

                # Convert predictions to bounding boxes in image coordinates.
                # Bounding boxes are scaled to the scale of the input images.
                mean = t.tensor(self.loc_normalize_mean).cuda(). \
                    repeat(self.n_class)[None]
                std = t.tensor(self.loc_normalize_std).cuda(). \
                    repeat(self.n_class)[None]

                roi_loc = (roi_loc * std + mean)
                roi_loc = roi_loc.view(-1, self.n_class, 4)

                roi = roi.view(-1, 1, 4).expand_as(roi_loc)
                bbox = loc2bbox(at.tonumpy(roi).reshape(-1, 4),
                                at.tonumpy(roi_loc).reshape(-1, 4))
                bbox = at.totensor(bbox)
                bbox = bbox.view(-1, self.n_class * 4)

                # clip bbox
                bbox[:, 0::2] = bbox[:, 0::2].clamp(min=0, max=original_size[0])
                bbox[:, 1::2] = bbox[:, 1::2].clamp(min=0, max=original_size[1])

                prob = F.softmax(at.totensor(roi_score), dim=1)

                bbox, label, score = self._suppress(bbox, prob)

                return bbox, label, score

    def _extract_features(self, x):
        raise NotImplementedError

    def _roi_pool(self, feature, roi):
        raise NotImplementedError

    def _bbox_regression_and_classification(self, roi_pool_feat):
        raise NotImplementedError

    def _suppress(self, raw_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            bbox_l = raw_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            bbox_l = bbox_l[mask]
            prob_l = prob_l[mask]
            keep = tv.ops.nms(bbox_l, prob_l, self.nms_thresh)
            # import ipdb;ipdb.set_trace()
            bbox.append(bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - data].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


def _bbox_regression_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()