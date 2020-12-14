from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.creator_tool_fpn import ProposalCreator


class RPN(nn.Module):
    def __init__(self, in_chs, mid_chs, scales, ratios, n_anchor, feat_strides):
        super(RPN, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.n_anchor = n_anchor
        self.feat_strides = feat_strides
        self.proposal_layer = ProposalCreator(self)
        self.conv1 = nn.Conv2d(in_chs, mid_chs, 3, 1, 1)
        self.score = nn.Conv2d(mid_chs, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_chs, n_anchor * 4, 1, 1, 0)

    def forward(self, features, img_size, scale):
        N = features[0].shape[0]
        feat_shapes = [tuple(f.shape[2:]) for f in features]

        anchor = generate_anchors_all_pyramid(self.scales,
                                              self.ratios,
                                              self.feat_strides,
                                              feat_shapes)

        rpn_locs = []
        rpn_scores = []
        rpn_fg_scores = []
        for x in features:
            H, W = x.shape[2:]
            h = F.relu(self.conv1(x))

            rpn_loc = self.loc(h)
            rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)

            rpn_score = self.score(h)
            rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()

            rpn_softmax_score = F.softmax(rpn_score.view(N, H, W, self.n_anchor, 2), dim=4)
            rpn_fg_score = rpn_softmax_score[:, :, :, :, 1].contiguous()
            rpn_fg_score = rpn_fg_score.view(N, -1)

            rpn_locs.append(rpn_loc)
            rpn_scores.append(rpn_score.view(N, -1, 2))
            rpn_fg_scores.append(rpn_fg_score)

        rpn_locs = torch.cat(rpn_locs, dim=1)
        rpn_scores = torch.cat(rpn_scores, dim=1)
        rpn_fg_scores = torch.cat(rpn_fg_scores, dim=1)

        roi = self.proposal_layer(
            rpn_locs[0].cpu().data.numpy(),
            rpn_fg_scores[0].cpu().data.numpy(),
            anchor,
            img_size,
            scale
        )

        return rpn_locs, rpn_scores, roi, anchor


def generate_anchors_all_pyramid(scales, ratios, feat_strides, feat_shapes):
    anchors = []
    for i in range(len(scales)):
        anchor = generate_anchors_single_pyramid(scales[i],
                                                 ratios,
                                                 feat_shapes[i],
                                                 feat_strides[i])
        anchors.append(anchor)
    return np.concatenate(anchors, axis=0)


def generate_anchors_single_pyramid(scale, ratios, feat_shape, feat_stride):
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scale), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales * np.sqrt(ratios)
    widths = scales / np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, feat_shape[0]) * feat_stride
    shifts_x = np.arange(0, feat_shape[1]) * feat_stride

    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (w, h)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes