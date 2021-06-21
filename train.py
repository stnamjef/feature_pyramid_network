import os
import argparse
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from data.voc_dataset import VOCDataset
from models.faster_rcnn_base import LossTuple
from models.faster_rcnn import FasterRCNN
from models.feature_pyramid_network import FPN
from utils.config import opt
from utils.eval_tool import evaluate
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
    save_path = f'checkpoints/{model_name}/{epoch}.pth'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    t.save(model.state_dict(), save_path)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI options for training a model.')
    parser.add_argument('--model', type=str, default='fpn', help='Model name: frcnn, fpn.')
    parser.add_argument('--backbone', type=str, default='vgg16', help='Backbone network: vgg16, resnet101.')
    parser.add_argument('--n_features', type=int, default=1, help='The number of features to use for RoI-pooling.')
    parser.add_argument('--data_dir', type=str, default='../dataset/VOC2007', help='Path to VOC dataset.')
    parser.add_argument('--min_size', type=int, default=600, help='Minimum input image size.')
    parser.add_argument('--max_size', type=int, default=1000, help='Maximum input image size.')
    parser.add_argument('--n_workers_train', type=int, default=8, help='The number of workers for a train loader.')
    parser.add_argument('--n_workers_test', type=int, default=8, help='The number of workers for a test loader.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default=1e-3).')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--epoch', type=int, default=15, help='Total epochs.')
    parser.add_argument('--epoch_decay', type=int, default=10, help='The epoch to decay learning rate.')

    args = parser.parse_args()
    opt._parse(vars(args))

    t.multiprocessing.set_sharing_strategy('file_system')

    train_data = VOCDataset(
        root=opt.data_dir,
        split='trainval',
        is_training=True
    )

    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        num_workers=opt.n_workers_train
    )

    test_data = VOCDataset(
        root=opt.data_dir,
        split='test',
        is_training=False
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_workers_test
    )

    print('Dataset loaded.')

    if opt.model == 'frcnn':
        model = FasterRCNN(20).cuda()
    elif opt.model == 'fpn':
        model = FPN(20).cuda()
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
        for img, bbox, label, scale in tqdm(train_loader):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
            optim.zero_grad()
            losses = model.forward(img, scale, bbox, label)
            losses.total_loss.backward()
            optim.step()
            update_meters(meters, losses)

        model.eval()
        res = evaluate(test_loader, model)

        md = get_meter_data(meters)
        log = f'Epoch: {e:2}, lr: {str(lr)}, map: {str(res["map"])}' + \
              f'rpn_loc_loss: {md["rpn_loc_loss"]:.3f}, rpn_cls_loss: {md["rpn_cls_loss"]}' + \
              f'roi_loc_loss: {md["roi_loc_loss"]:.3f}, roi_cls_loss: {md["roi_cls_loss"]}' + \
              f'total_loss: {md["total_loss"]:.3f}'

        print(log)

        if res['map'] > best_map:
            best_map = res['map']
            best_path = save_model(model, opt.model, e)

        if e == opt.epoch_decay:
            state_dict = t.load(best_path)
            model.load_state_dict(state_dict)
            # deacy lr
            for param_group in optim.param_groups:
                param_group['lr'] *= opt.lr_decay
            lr = lr * opt.lr_decay
