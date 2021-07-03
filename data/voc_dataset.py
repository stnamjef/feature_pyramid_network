import os
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset
from utils.config import opt
from data.utils import read_image
from data.transforms import TransformsTrain, TransformsTest, TransformsPlot


class VOCDataset(Dataset):
    def __init__(self, root, split, transform_mode, use_difficult=False):
        super(VOCDataset, self).__init__()
        img_ids_all = os.path.join(root, 'ImageSets/Main/{0}.txt'.format(split))
        self.img_ids = [i.strip() for i in open(img_ids_all)]
        self.root = root
        self.transform_mode = transform_mode
        self.use_difficult = use_difficult

        if self.transform_mode == 'train':
            self.transforms = TransformsTrain(opt.min_size, opt.max_size)
        elif self.transform_mode == 'test':
            self.transforms = TransformsTest(opt.min_size, opt.max_size)
        elif self.transform_mode == 'plot':
            self.transforms = TransformsPlot(opt.min_size, opt.max_size)
        else:
            raise ValueError('Invalid dataset mode.')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, i):
        img, bbox, label, difficult = self._load_image_and_target(i)
        return self.transforms(img, bbox, label, difficult)

    def _load_image_and_target(self, i):
        img_id = self.img_ids[i]
        annot = ET.parse(os.path.join(self.root, 'Annotations', img_id + '.xml'))

        bbox, label, difficult = [], [], []
        for obj in annot.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bbox_annot = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bbox_annot.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')
            ])

            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load an image
        img_file = os.path.join(self.root, 'JPEGImages', img_id + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')