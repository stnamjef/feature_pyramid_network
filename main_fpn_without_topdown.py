from __future__ import absolute_import

import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import array_tool as at
from utils.config import opt
from utils.eval_tool import eval_detection_voc
from data.dataset import Dataset, TestDataset
from model.feature_pyramid_network_without_topdown import FasterRCNN


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        original_size = [sizes[0][0].item(), sizes[1][0].item()]
        scale = imgs.shape[3] / original_size[1]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn(imgs, scale, None, None, original_size)
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    return result


def save(model):
    timestr = time.strftime('%m%d%H%M')
    save_path = 'checkpoints_fpn_without_topdown/fasterrcnn_%s' % timestr
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_path)

    return save_path


if __name__ == '__main__':
    trainset = Dataset(opt)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = TestDataset(opt)
    test_loader = DataLoader(testset, batch_size=1, num_workers=2, shuffle=False, pin_memory=True)
    print('data loaded')

    model = FasterRCNN(n_fg_class=20, scales=[16 * 8, 16 * 16, 16 * 32], ratios=[0.5, 1, 2]).cuda()

    best_map = 0
    lr = 0.001
    for epoch in range(20):
        model.train()
        model.reset_meters()
        for i, (img, bbox, label, scale) in tqdm(enumerate(train_loader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()

            model.optimizer.zero_grad()
            losses = model.forward(img, scale, bbox, label)
            losses.total_loss.backward()
            model.optimizer.step()
            model.update_meters(losses)

        model.eval()
        eval_result = eval(test_loader, model, test_num=opt.test_num)
        log_info = f'lr: {str(lr)}, map: {str(eval_result["map"])}, loss: {str(model.get_meter_data())}'
        print(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = save(model)
        if epoch == 14:
            state_dict = torch.load(best_path)
            model.load_state_dict(state_dict)
            lr = 0.0001


## test
# if __name__ == '__main__':
#     testset = TestDataset(opt)
#     test_loader = DataLoader(testset, batch_size=1, num_workers=2, shuffle=False, pin_memory=True)
#     print('data loaded')

#     model = FasterRCNN(n_fg_class=20, scales=[16 * 8, 16 * 16, 16 * 32], ratios=[0.5, 1, 2]).cuda()

#     state_dict = torch.load('./checkpoints_fpn_without_topdown/frcnn_12091048')
#     model.load_state_dict(state_dict)

#     model.eval()
#     eval_result = eval(test_loader, model, test_num=opt.test_num)
#     print(f'ap: {eval_result["ap"]}, map: {eval_result["map"]}')