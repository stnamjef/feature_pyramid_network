import argparse
import torch as t
from torch.utils.data import DataLoader
from data.voc_dataset import VOCDataset, VOC_BBOX_LABEL_NAMES
from models.faster_rcnn import FasterRCNN
from models.feature_pyramid_network import FPN
from utils.config import opt
from utils.eval_tool import evaluate


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

    args = parser.parse_args()
    opt._parse(vars(args))

    t.multiprocessing.set_sharing_strategy('file_system')

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

    print('Dataset loaded')

    if opt.model == 'frcnn':
        model = FasterRCNN(20).cuda()
        pretrained_path = f'{opt.pretrained_dir}/{opt.model}_{opt.backbone}.pth'
    elif opt.model == 'fpn':
        model = FPN(20).cuda()
        pretrained_path = f'{opt.pretrained_dir}/{opt.model}_{opt.backbone}_{opt.n_features}.pth'
    else:
        raise ValueError('Invalid model. It muse be either frcnn or fpn.')

    print('Model construction completed.')

    model.load_state_dict(t.load(pretrained_path))
    print('Pretrained weights loaded.')

    model.eval()
    eval_result = evaluate(test_loader, model)
    print('Average Precisions:')
    for k, v in zip(VOC_BBOX_LABEL_NAMES, eval_result['ap']):
        print(f'  {k}: {v * 100:.2f}')
    print(f'mAP: {eval_result["map"] * 100:.2f}')
