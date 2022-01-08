import os
import argparse
import cv2
import numpy as np
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.voc_dataset import VOCDataset, VOC_BBOX_LABEL_NAMES
from data.coco_dataset import COCODataset
from models.faster_rcnn import FasterRCNN
from models.feature_pyramid_network import FPN
from utils.config import opt
from utils.viz_tool import add
import utils.array_tool as at


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI options for testing a model.')
    parser.add_argument('--model', type=str, default='fpn',
                        help='Model name: frcnn, fpn (default=fpn).')
    parser.add_argument('--backbone', type=str, default='vgg16',
                        help='Backbone network: vgg16, resnet101 (default=vgg16).')
    parser.add_argument('--n_features', type=int, default=1,
                        help='The number of features to use for RoI-pooling (default=1).')
    parser.add_argument('--dataset', type=str, default='voc07',
                        help='Testing dataset: voc07, coco (default=voc07).')
    parser.add_argument('--data_dir', type=str, default='../dataset',
                        help='Testing dataset directory (default=../dataset).')
    parser.add_argument('--save_dir', type=str, default='./model_zoo',
                        help='Saving directory (default=./model_zoo).')
    parser.add_argument('--min_size', type=int, default=600,
                        help='Minimum input image size (default=600).')
    parser.add_argument('--max_size', type=int, default=1000,
                        help='Maximum input image size (default=1000).')
    parser.add_argument('--n_workers_test', type=int, default=8,
                        help='The number of workers for a test loader (default=8).')
    parser.add_argument('--nms_thresh', type=int, default=0.3,
                        help='IoU threshold for NMS (default=0.3).')
    parser.add_argument('--score_thresh', type=int, default=0.6,
                        help='BBoxes with scores less than this are excluded (default=0.6).')
    parser.add_argument('--n_plots', type=int, default=-1,
                        help='The number of images to plot predictions (default=-1; all images).')

    args = parser.parse_args()
    opt._parse(vars(args))

    t.multiprocessing.set_sharing_strategy('file_system')

    if opt.dataset == 'voc07':
        n_fg_class = 20
        test_data = VOCDataset(opt.data_dir + '/VOCdevkit/VOC2007', 'test', 'plot')
    elif opt.dataset == 'coco':
        n_fg_class = 80
        test_data = COCODataset(opt.data_dir + '/COCO', 'val', 'plot')
    else:
        raise ValueError('Invalid dataset.')

    test_loader = DataLoader(test_data, 1, False, num_workers=opt.n_workers_test)

    print('Dataset loaded.')

    if opt.model == 'frcnn':
        model = FasterRCNN(n_fg_class).cuda()
        model_name = f'{opt.model}_{opt.backbone}'
        save_path = f'{opt.save_dir}/{model_name}.pth'
    elif opt.model == 'fpn':
        model = FPN(n_fg_class).cuda()
        model_name = f'{opt.model}_{opt.backbone}_{opt.n_features}'
        save_path = f'{opt.save_dir}/{model_name}.pth'
    else:
        raise ValueError('Invalid model. It muse be either frcnn or fpn.')

    print('Model construction completed.')

    model.load_state_dict(t.load(save_path))
    print('Pretrained weights loaded.')

    plot_dir = f'./results/{opt.dataset}/plots/{model_name}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model.eval()
    for i, (original_img, img, scale, size) in tqdm(enumerate(test_loader)):
        scale = at.scalar(scale)
        original_size = [size[0][0].item(), size[1][0].item()]

        pred_bboxes, pred_labels, pred_scores = model(img, scale, None, None, original_size)

        original_img = original_img.squeeze(0).cpu().numpy()
        original_img = original_img.transpose(1, 2, 0)
        original_img = np.uint8(original_img)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
            bbox = [int(p) for p in bbox]
            ymin, xmin, ymax, xmax = bbox
            label_name = VOC_BBOX_LABEL_NAMES[label]
            raw_img = add(original_img, xmin, ymin, xmax, ymax, label=label_name)

        cv2.imwrite(f'{plot_dir}/{test_data.img_ids[i]}.png', original_img)

        if i + 1 == opt.n_plots:
            break
