from __future__ import absolute_import

from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms
from torchvision.ops import RoIPool
from torchvision.models import vgg16
from torchnet.meter import ConfusionMeter, AverageValueMeter

from utils import array_tool as at
from utils.config import opt
from model.utils.bbox_tools import generate_anchor_base, loc2bbox
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator, ProposalCreator


LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
           return f(*args, **kwargs)
    return new_f


class FasterRCNN(nn.Module):

    feat_stride = 16

    def __init__(self, n_fg_class, ratios, scales):
        super(FasterRCNN, self).__init__()

        extractor, classifier = load_vgg16()

        self.extractor = extractor
        self.rpn = RPN(
            in_chs=512,
            mid_chs=512,
            ratios=[0.5, 1., 2.],
            scales=[8, 16, 32],
            feat_strides=self.feat_stride
        )
        self.head = RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=1/16.,
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
        self.anchor_target_layer = AnchorTargetCreator()
        self.proposal_target_layer = ProposalTargetCreator()

        self.optimizer = self.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, x, scale, gt_bboxes, gt_labels, original_size=None):
        if self.training:
            img_size = tuple(x.shape[2:])
            features = self.extractor(x)

            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)

            # Since batch size is one, convert variables to singular form
            bbox = gt_bboxes[0]
            label = gt_labels[0]
            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            roi = rois

            # Sample RoIs and forward
            # it's fine to break the computation graph of rois,
            # consider them as constant input
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(
                roi,
                at.tonumpy(bbox),
                at.tonumpy(label),
                self.loc_normalize_mean,
                self.loc_normalize_std)

            # NOTE it's all zero because now it only support for batch=1 now
            sample_roi_index = torch.zeros(len(sample_roi))

            # print('ROI head start')
            roi_cls_loc, roi_score = self.head(
                features,
                sample_roi,
                sample_roi_index)
            # print('Anchor target layer start')
            # ------------------ RPN losses -------------------#
            gt_rpn_loc, gt_rpn_label = self.anchor_target_layer(
                at.tonumpy(bbox),
                anchor,
                img_size)
            gt_rpn_label = at.totensor(gt_rpn_label).long()
            gt_rpn_loc = at.totensor(gt_rpn_loc)

            # print('RPN losses start')
            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label.data,
                self.rpn_sigma)

            # NOTE: default value of ignore_index is -100 ...
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
            _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
            _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
            self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

            # print('Calc ROI losses start')
            # ------------------ ROI losses (fast rcnn loss) -------------------#
            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                                  at.totensor(gt_roi_label).long()]
            gt_roi_label = at.totensor(gt_roi_label).long()
            gt_roi_loc = at.totensor(gt_roi_loc)

            roi_loc_loss = _fast_rcnn_loc_loss(
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
                img_size = x.shape[2:]

                features = self.extractor(x)

                rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)
                roi_cls_locs, roi_scores = self.head(features, rois, roi_indices)

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


def load_vgg16():
    model = vgg16(pretrained=True)
    # the 30th layer of features is relu of conv5_3
    features = list(model.features)[:-1]
    for i in range(len(features)):
        # if isinstance(features[i], nn.MaxPool2d):
        #     features[i] = nn.MaxPool2d(2, stride=2, return_indices=True)
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


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss


class RPN(nn.Module):
    def __init__(self, in_chs, mid_chs, ratios, scales, feat_strides):
        super(RPN, self).__init__()
        self.anchor_base = generate_anchor_base(
            base_size=16,
            ratios=ratios,
            scales=scales
        )
        self.feat_strides = feat_strides
        self.proposal_layer = ProposalCreator(
            self,
            nms_thresh=0.7,
            n_train_pre_nms=12000,
            n_train_post_nms=2000,
            n_test_pre_nms=6000,
            n_test_post_nms=300,
            min_size=16
        )

        n_anchor = len(self.anchor_base)
        self.conv1 = nn.Conv2d(in_chs, mid_chs, 3, 1, 1)
        self.score = nn.Conv2d(mid_chs, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_chs, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, features, img_size, scale):
        '''
        :param features: The features(N, C, H, W) extracted from images.
        :param img_size: A tuple if (height, width), which contains image size after scaling.
        :param scale: The amount of scaling done to the input images after reading them from files.
        '''
        N, _, H, W = features.shape
        anchor = enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_strides, H, W)
        n_anchor = anchor.shape[0] // (H * W)
        mid_features = F.relu(self.conv1(features))

        rpn_locs = self.loc(mid_features)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)

        rpn_scores = self.score(mid_features)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(N, H, W, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(N, -1)
        rpn_scores = rpn_scores.view(N, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(N):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size, scale)
            batch_idx = i * np.ones(len(roi), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_idx)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    '''
    Enumerate all shifted anchors

    add A anchors (1, A, 4) to
    cell K shifts (K, 1, 4) to get
    shift anchors (K, A, 4)
    reshape to (K * A, 4) shifted anchors
    '''
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor


class RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, features, rois, roi_indices):
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)  # roi_indices만큼 roi가 있나?
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(features, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores

    # def forward(self, imgs, bboxes, labels, scale):
    #     img_size = tuple(imgs.shape[2:])
    #     features = self.extractor(imgs)
    #
    #     # print("RPN start")
    #     rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)
    #
    #     # Since batch size is one, convert variables to singular form
    #     bbox = bboxes[0]
    #     label = labels[0]
    #     rpn_loc = rpn_locs[0]
    #     rpn_score = rpn_scores[0]
    #     roi = rois
    #
    #     # Sample RoIs and forward
    #     # it's fine to break the computation graph of rois,
    #     # consider them as constant input
    #     # print('Proposal target layer start')
    #     sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(
    #         roi,
    #         at.tonumpy(bbox),
    #         at.tonumpy(label),
    #         self.loc_normalize_mean,
    #         self.loc_normalize_std)
    #     # NOTE it's all zero because now it only support for batch=1 now
    #     sample_roi_index = torch.zeros(len(sample_roi))
    #
    #     # print('ROI head start')
    #     roi_cls_loc, roi_score = self.head(
    #         features,
    #         sample_roi,
    #         sample_roi_index)
    #     # print('Anchor target layer start')
    #     # ------------------ RPN losses -------------------#
    #     gt_rpn_loc, gt_rpn_label = self.anchor_target_layer(
    #         at.tonumpy(bbox),
    #         anchor,
    #         img_size)
    #     gt_rpn_label = at.totensor(gt_rpn_label).long()
    #     gt_rpn_loc = at.totensor(gt_rpn_loc)
    #
    #     # print('RPN losses start')
    #     rpn_loc_loss = _fast_rcnn_loc_loss(
    #         rpn_loc,
    #         gt_rpn_loc,
    #         gt_rpn_label.data,
    #         self.rpn_sigma)
    #
    #     # NOTE: default value of ignore_index is -100 ...
    #     rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
    #     _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
    #     _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
    #     self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())
    #
    #     # print('Calc ROI losses start')
    #     # ------------------ ROI losses (fast rcnn loss) -------------------#
    #     n_sample = roi_cls_loc.shape[0]
    #     roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
    #     roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
    #                           at.totensor(gt_roi_label).long()]
    #     gt_roi_label = at.totensor(gt_roi_label).long()
    #     gt_roi_loc = at.totensor(gt_roi_loc)
    #
    #     roi_loc_loss = _fast_rcnn_loc_loss(
    #         roi_loc.contiguous(),
    #         gt_roi_loc,
    #         gt_roi_label.data,
    #         self.roi_sigma)
    #
    #     roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
    #
    #     self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())
    #
    #     losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
    #     losses = losses + [sum(losses)]
    #
    #     return LossTuple(*losses)
    #
    # def forward2(self, x, scale):
    #     img_size = x.shape[2:]
    #
    #     h = self.extractor(x)
    #     rpn_locs, rpn_scores, rois, roi_indices, anchor = \
    #         self.rpn(h, img_size, scale)
    #     roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
    #
    #     return roi_cls_locs, roi_scores, rois, roi_indices
    #
    # @nograd
    # def predict(self, imgs, sizes=None, visualize=False):
    #     """Detect objects from images.
    #     This method predicts objects for each image.
    #     Args:
    #         imgs (iterable of numpy.ndarray): Arrays holding images.
    #             All images are in CHW and RGB format
    #             and the range of their value is :math:`[0, 255]`.
    #     Returns:
    #        tuple of lists:
    #        This method returns a tuple of three lists,
    #        :obj:`(bboxes, labels, scores)`.
    #        * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
    #            where :math:`R` is the number of bounding boxes in a image. \
    #            Each bouding box is organized by \
    #            :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
    #            in the second axis.
    #        * **labels** : A list of integer arrays of shape :math:`(R,)`. \
    #            Each value indicates the class of the bounding box. \
    #            Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
    #            number of the foreground classes.
    #        * **scores** : A list of float arrays of shape :math:`(R,)`. \
    #            Each value indicates how confident the prediction is.
    #     """
    #     self.eval()
    #     bboxes = list()
    #     labels = list()
    #     scores = list()
    #     for img, size in zip(imgs, sizes):
    #         img = at.totensor(img[None]).float()
    #         scale = img.shape[3] / size[1]
    #         roi_cls_loc, roi_scores, rois, _ = self.forward2(img, scale=scale)
    #         # We are assuming that batch size is 1.
    #         roi_score = roi_scores.data
    #         roi_cls_loc = roi_cls_loc.data
    #         roi = at.totensor(rois) / scale
    #
    #         # Convert predictions to bounding boxes in image coordinates.
    #         # Bounding boxes are scaled to the scale of the input images.
    #         # mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
    #         #     repeat(self.n_class)[None]
    #         # std = torch.Tensor(self.loc_normalize_std).cuda(). \
    #         #     repeat(self.n_class)[None]
    #
    #         mean = torch.Tensor(self.loc_normalize_mean). \
    #             repeat(self.n_class)[None]
    #         std = torch.Tensor(self.loc_normalize_std). \
    #             repeat(self.n_class)[None]
    #
    #         roi_cls_loc = (roi_cls_loc * std + mean)
    #         roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
    #         roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
    #         cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
    #                             at.tonumpy(roi_cls_loc).reshape((-1, 4)))
    #         cls_bbox = at.totensor(cls_bbox)
    #         cls_bbox = cls_bbox.view(-1, self.n_class * 4)
    #         # clip bounding box
    #         cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
    #         cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
    #
    #         prob = (F.softmax(at.totensor(roi_score), dim=1))
    #
    #         bbox, label, score = self._suppress(cls_bbox, prob)
    #         bboxes.append(bbox)
    #         labels.append(label)
    #         scores.append(score)
    #
    #     self.train()
    #
    #     return bboxes, labels, scores