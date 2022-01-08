import os
import shutil
import argparse
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from data.coco_dataset import COCODataset
from data.voc_dataset import VOCDataset
from models.faster_rcnn_base import LossTuple
from models.faster_rcnn import FasterRCNN
from models.feature_pyramid_network import FPN
from utils.config import opt
from utils.eval_tool import evaluate_voc, evaluate_coco
import utils.array_tool as at


def get_optimizer(model):
    lr = opt.lr
    params = []
    for k, v in dict(model.named_parameters()).items():
        if v.requires_grad:
            if 'bias' in k:
                params += [{'params': [v], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [v], 'lr': lr, 'weight_decay': opt.weight_decay}]

    return t.optim.SGD(params, momentum=0.9)


def reset_meters(meters):
    for key, meter in meters.items():
        meter.reset()


def update_meters(meters, losses):
    loss_d = {k: at.scalar(v) for k, v, in losses._asdict().items()}
    for key, meter in meters.items():
        meter.add(loss_d[key])


def get_meter_data(meters):
    return {k: v.value()[0] for k, v in meters.items()}


def save_model(model, model_name, epoch):
    save_path = f'./checkpoints/{model_name}/{epoch}.pth'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    t.save(model.state_dict(), save_path)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI options for training a model.')
    parser.add_argument('--model', type=str, default='fpn',
                        help='Model name: frcnn, fpn (default=fpn).')
    parser.add_argument('--backbone', type=str, default='vgg16',
                        help='Backbone network: vgg16, resnet101 (default=vgg16).')
    parser.add_argument('--n_features', type=int, default=1,
                        help='The number of features to use for RoI-pooling (default=1).')
    parser.add_argument('--dataset', type=str, default='voc07',
                        help='Training dataset: voc07, voc0712, coco (default=voc07).')
    parser.add_argument('--data_dir', type=str, default='../dataset',
                        help='Training dataset directory (default=../dataset).')
    parser.add_argument('--save_dir', type=str, default='./model_zoo',
                        help='Saving directory (default=./model_zoo).')
    parser.add_argument('--min_size', type=int, default=600,
                        help='Minimum input image size (default=600).')
    parser.add_argument('--max_size', type=int, default=1000,
                        help='Maximum input image size (default=1000).')
    parser.add_argument('--n_workers_train', type=int, default=8,
                        help='The number of workers for a train loader (default=8).')
    parser.add_argument('--n_workers_test', type=int, default=8,
                        help='The number of workers for a test loader (default=8).')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default=1e-3).')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='Learning rate decay (default=0.1; 1e-3 -> 1e-4).')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (default=5e-4).')
    parser.add_argument('--epoch', type=int, default=15,
                        help='Total epochs (default=15).')
    parser.add_argument('--epoch_decay', type=int, default=10,
                        help='The epoch to decay learning rate (default=10).')
    parser.add_argument('--nms_thresh', type=int, default=0.3,
                        help='IoU threshold for NMS (default=0.3).')
    parser.add_argument('--score_thresh', type=int, default=0.05,
                        help='BBoxes with scores less than this are excluded (default=0.05).')

    args = parser.parse_args()
    opt._parse(vars(args))

    t.multiprocessing.set_sharing_strategy('file_system')

    if opt.dataset == 'voc07':
        n_fg_class = 20
        train_data = [VOCDataset(opt.data_dir + '/VOCdevkit/VOC2007', 'trainval', 'train')]
        test_data = VOCDataset(opt.data_dir + '/VOCdevkit/VOC2007', 'test', 'test', True)
    elif opt.dataset == 'voc0712':
        n_fg_class = 20
        train_data = [VOCDataset(opt.data_dir + '/VOCdevkit/VOC2007', 'trainval', 'train'),
                      VOCDataset(opt.data_dir + '/VOCdevkit/VOC2012', 'trainval', 'train')]
        test_data = VOCDataset(opt.data_dir + '/VOCdevkit/VOC2007', 'test', 'test', True)
    elif opt.dataset == 'coco':
        n_fg_class = 80
        train_data = [COCODataset(opt.data_dir + '/COCO', 'train', 'train')]
        test_data = COCODataset(opt.data_dir + '/COCO', 'val', 'test')
    else:
        raise ValueError('Invalid dataset.')

    train_loaders = [DataLoader(dta, 1, True, num_workers=opt.n_workers_train) for dta in train_data]
    test_loader = DataLoader(test_data, 1, False, num_workers=opt.n_workers_test)

    print('Dataset loaded.')

    if opt.model == 'frcnn':
        model = FasterRCNN(n_fg_class).cuda()
        save_path = f'{opt.save_dir}/{opt.model}_{opt.backbone}.pth'
    elif opt.model == 'fpn':
        model = FPN(n_fg_class).cuda()
        save_path = f'{opt.save_dir}/{opt.model}_{opt.backbone}_{opt.n_features}.pth'
    else:
        raise ValueError('Invalid model. It muse be either frcnn or fpn.')

    print('Model construction completed.')

    optim = get_optimizer(model)
    print('Optimizer loaded.')

    meters = {k: AverageValueMeter() for k in LossTuple._fields}

    lr = opt.lr
    best_map = 0
    for e in range(1, opt.epoch + 1):
        model.train()
        reset_meters(meters)
        for train_loader in train_loaders:
            for img, bbox, label, scale in tqdm(train_loader):
                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
                optim.zero_grad()
                losses = model.forward(img, scale, bbox, label)
                losses.total_loss.backward()
                optim.step()
                update_meters(meters, losses)

        md = get_meter_data(meters)
        log = f'Epoch: {e:2}, lr: {str(lr)}, ' + \
              f'rpn_loc_loss: {md["rpn_loc_loss"]:.4f}, rpn_cls_loss: {md["rpn_cls_loss"]:.4f}, ' + \
              f'roi_loc_loss: {md["roi_loc_loss"]:.4f}, roi_cls_loss: {md["roi_cls_loss"]:.4f}, ' + \
              f'total_loss: {md["total_loss"]:.4f}'

        print(log)

        model.eval()
        if opt.dataset != 'coco':
            map = evaluate_voc(test_loader, model)
        else:
            map = evaluate_coco(test_data, test_loader, model)

        # update best mAP
        if map > best_map:
            best_map = map
            best_path = save_model(model, opt.model, e)

        if e == opt.epoch_decay:
            state_dict = t.load(best_path)
            model.load_state_dict(state_dict)
            # decay learning rate
            for param_group in optim.param_groups:
                param_group['lr'] *= opt.lr_decay
            lr = lr * opt.lr_decay

    # save best model to opt.save_dir
    shutil.copyfile(best_path, save_path)
