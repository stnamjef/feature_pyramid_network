from __future__ import division
import os
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import itertools
import numpy as np
import six
import utils.array_tool as at
from models.utils.bbox_tools import bbox_iou
from utils.config import opt


class COCOEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(COCOEncoder, self).default(obj)


def evaluate_coco(data, data_loader, model):
    n_ids = len(data.img_ids)
    result = []

    for i, (img, bbox, label, scale, size, _) in tqdm(zip(data.img_ids, data_loader), total=n_ids):
        scale = at.scalar(scale)
        original_size = [size[0][0].item(), size[1][0].item()]
        pred_bbox, pred_label, pred_score = model(img, scale, None, None, original_size)

        for b, l, s in zip(pred_bbox, pred_label, pred_score):
            ymin, xmin, ymax, xmax = b
            obj = OrderedDict({
                'image_id': i,
                'category_id': data.label_to_coco_label(l),
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                'score': float(s)
            })
            result.append(obj)

    result_path = f'./results/coco/predictions/{opt.model}.json'
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_path, 'w', encoding='utf-8') as fout:
        json.dump(result, fout, cls=COCOEncoder, ensure_ascii=False)

    eval_result = data.evaluate(result_path)

    return eval_result


def evaluate_voc(data_loader, model):
    pred_bboxes, pred_labels, pred_scores = [], [], []
    gt_bboxes, gt_labels, gt_difficults = [], [], []

    for img, gt_bbox, gt_label, scale, size, gt_difficult in tqdm(data_loader):
        scale = at.scalar(scale)
        original_size = [size[0][0].item(), size[1][0].item()]
        pred_bbox, pred_label, pred_score = model(img, scale, None, None, original_size)
        gt_bboxes += list(gt_bbox.numpy())
        gt_labels += list(gt_label.numpy())
        gt_difficults += list(gt_difficult.numpy())
        pred_bboxes += [pred_bbox]
        pred_labels += [pred_label]
        pred_scores += [pred_score]

    eval_results = {'AP': 0, 'AP_0.5': 0, 'AP_0.75': 0, 'AP_s': 0, 'AP_m': 0, 'AP_l': 0}
    iou_threshes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    area_names = ['s', 'm', 'l']
    area_ranges = [(0, 32 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, np.inf)]
    for name, range in zip(area_names, area_ranges):
        # evaluate predictions for multiple iou threshes
        for iou_thresh in iou_threshes:
            result = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults,
                iou_thresh, True, range
            )
            # accumulate results
            eval_results[f'AP_{name}'] += result['map']
        # average results
        eval_results[f'AP_{name}'] /= 10.

    # evaluate results regardless of area size
    for iou_thresh in iou_threshes:
        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            iou_thresh, True
        )
        # accumulate results
        eval_results['AP'] += result['map']
        # save map for iou 0.5 & 0.75
        if iou_thresh == 0.5:
            eval_results['AP_0.5'] = result['map']
        elif iou_thresh == 0.75:
            eval_results['AP_0.75'] = result['map']
        else:
            continue
    eval_results['AP'] /= 10

    # print results
    eval_log = ''
    for k, v in eval_results.items():
        eval_log += f'{k}: {v * 100:.2f},  '
    print(eval_log)

    return eval_results['AP']


def eval_detection_voc(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None, iou_thresh=0.5, use_07_metric=False, area_range=None):

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh, area_range)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_prec_rec(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
                                gt_difficults=None, iou_thresh=0.5, area_range=None):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)

    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in zip(
            pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults
    ):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            # generate ignore gt list by area_range
            def _is_ignore(bb):
                if area_range is None:
                    return False
                area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                return not (area_range[0] <= area <= area_range[1])

            gt_ignore = [_is_ignore(bb) for bb in gt_bbox_l]

            score[l].extend(pred_score_l)
            for difficult, ignore in zip(gt_difficult_l, gt_ignore):
                if not difficult and not ignore:
                    n_pos[l] += 1

            if len(pred_bbox_l) == 0:
                continue

            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            ious = bbox_iou(pred_bbox_l, gt_bbox_l)

            # sort gt_bbox by ignore list
            gt_sort = np.argsort(gt_ignore, kind='stable')
            gt_bbo_lx = [gt_bbox_l[i] for i in gt_sort]
            gt_ignore = [gt_ignore[i] for i in gt_sort]
            ious = ious[:, gt_sort]

            gtm = {}
            dtm = {}

            for d_idx, d in enumerate(pred_bbox_l):
                # information about best match so far (m=-1 -> unmatched)
                iou = min(iou_thresh, 1 - 1e-10)
                m = -1
                for g_idx, g in enumerate(gt_bbox_l):
                    # if this gt already matched, continue
                    if g_idx in gtm:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                        break
                    # continue to next gt unless better match made
                    if ious[d_idx, g_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = ious[d_idx, g_idx]
                    m = g_idx
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtm[d_idx] = m
                gtm[m] = d_idx

            # generate ignore list for dts
            dt_ignore = []
            for d_idx, d in enumerate(pred_bbox_l):
                if d_idx in dtm:
                    dt_ignore.append(gt_ignore[dtm[d_idx]])
                else:
                    dt_ignore.append(_is_ignore(d))

            for d_idx in range(len(pred_bbox_l)):
                if not dt_ignore[d_idx]:
                    if d_idx in dtm:
                        if gt_difficult_l[dtm[d_idx]]:
                            match[l].append(-1)
                        else:
                            match[l].append(1)
                    else:
                        match[l].append(0)
                else:
                    match[l].append(-1)

    for iter_ in (pred_bboxes, pred_labels, pred_scores,
                  gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
