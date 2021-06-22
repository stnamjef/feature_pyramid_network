import os
import argparse
import cv2
import numpy as np
import torch as t
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.voc_dataset import VOCDataset, VOC_BBOX_LABEL_NAMES
from models.faster_rcnn import FasterRCNN
from models.feature_pyramid_network import FPN
from utils.config import opt
from utils.viz_tool import add
import utils.array_tool as at


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI options for testing a model.')
    parser.add_argument('--model', type=str, default='fpn', help='Model name: frcnn, fpn.')
    parser.add_argument('--backbone', type=str, default='vgg16', help='Backbone network: vgg16, resnet101.')
    parser.add_argument('--n_features', type=int, default=1, help='The number of features to use for RoI-pooling.')
    parser.add_argument('--pretrained_dir', type=str, default='./model_zoo', help='A path to pretrained models.')
    parser.add_argument('--data_dir', type=str, default='../dataset/VOC2007', help='A path to VOC dataset.')
    parser.add_argument('--min_size', type=int, default=600, help='Minimum input image size.')
    parser.add_argument('--max_size', type=int, default=1000, help='Maximum input image size.')
    parser.add_argument('--n_workers_test', type=int, default=8, help='The number of workers for a test loader.')
    parser.add_argument('--nms_thresh', type=int, default=0.3, help='IoU threshold for NMS.')
    parser.add_argument('--score_thresh', type=int, default=0.6, help='BBoxes with scores less than this are excluded.')

    args = parser.parse_args()
    opt._parse(vars(args))

    t.multiprocessing.set_sharing_strategy('file_system')

    test_data = VOCDataset(
        root=opt.data_dir,
        split='test',
        transform_mode='plot'
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
        full_model_name = f'{opt.model}_{opt.backbone}'
        pretrained_path = f'{opt.pretrained_dir}/{full_model_name}.pth'
    elif opt.model == 'fpn':
        model = FPN(20).cuda()
        full_model_name = f'{opt.model}_{opt.backbone}_{opt.n_features}'
        pretrained_path = f'{opt.pretrained_dir}/{full_model_name}.pth'
    else:
        raise ValueError('Invalid model. It muse be either frcnn or fpn.')

    print('Model construction completed.')

    model.load_state_dict(t.load(pretrained_path))
    print('Pretrained weights loaded.')

    save_dir = f'./plots/{full_model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

        cv2.imwrite(f'{save_dir}/{i + 1}.png', original_img)

        if i == 9:
            break
