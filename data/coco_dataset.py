import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset
from utils.config import opt
from data.utils import read_image
from data.transforms import TransformsTrain, TransformsTest, TransformsPlot


class COCODataset(Dataset):
    def __init__(self, root, split, transform_mode):
        super(COCODataset, self).__init__()
        self.root = root
        self.split = split
        self.transform_mode = transform_mode

        self.coco = COCO(os.path.join(root, 'annotations', 'instances_' + split + '2017.json'))
        self.img_ids = self._load_image_ids()

        if self.transform_mode == 'train':
            self.transforms = TransformsTrain(opt.min_size, opt.max_size)
        elif self.transform_mode == 'test':
            self.transforms = TransformsTest(opt.min_size, opt.max_size)
        elif self.transform_mode == 'plot':
            self.transforms = TransformsPlot(opt.min_size, opt.max_size)
        else:
            raise ValueError('Invalid dataset mode.')

        self._load_classes()

    def _load_image_ids(self):
        self.img_ids = []
        self.img_without_annot_ids = []

        img_ids_all = self.coco.getImgIds()
        for i in img_ids_all:
            annot_ids = self.coco.getAnnIds(imgIds=i, iscrowd=False)
            if len(annot_ids) == 0:
                self.img_without_annot_ids.append(i)
            else:
                self.img_ids.append(i)

        print(f'{len(self.img_without_annot_ids)} image ids with no annotation are excluded.')
        print(f'A total of {len(self.img_ids)} image ids are loaded.')

        return self.img_ids

    def _load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, i):
        img, bbox, label = self._load_image_and_target(i)
        return self.transforms(img, bbox, label)

    def _load_image_and_target(self, i):
        # load image
        img_info = self.coco.loadImgs(self.img_ids[i])[0]
        path = os.path.join(self.root, 'images', self.split + '2017', img_info['file_name'])
        img = read_image(path)

        # load annotation
        annot_ids = self.coco.getAnnIds(imgIds=self.img_ids[i], iscrowd=False)

        # for debug
        # if len(annot_ids) == 0:
        #     print('CocoDataset::_load_image_and_target(): Empty annotations.')
        #     exit(1)

        bbox, label = [], []
        annots = self.coco.loadAnns(annot_ids)
        for annot in annots:
            x, y, w, h = annot['bbox']

            if w < 1 or h < 1:
                # print('Warning::Skipping annotation.')
                continue

            bbox.append([y, x, y + h, x + w])
            label.append(self.coco_label_to_label(annot['category_id']))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        return img, bbox, label

    def evaluate(self, result_path):
        cocoDt = self.coco.loadRes(result_path)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.params.imgIds = self.img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]