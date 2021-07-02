import random
import numpy as np
import torch as t
import torchvision.transforms.functional as F
import skimage.transform as sktsf
from PIL import Image


def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        # This code raises the error below, and I do not know why.
        # TypeError: __array__() takes 1 positional argument but 2 were given
        # img = np.asarray(img, dtype=dtype)
        img = np.asarray(img).astype(dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def resize_image(img, min_size, max_size):
    # resize img
    c, h, w = img.shape
    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (c, h * scale, w * scale), mode='reflect', anti_aliasing=False)

    return img


def normalize_image(img):
    img = F.normalize(
        tensor=t.from_numpy(img),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True
    ).numpy()
    return img


def horizontal_flip_image(img):
    if random.choice([True, False]):
        return img[:, :, ::-1], True
    else:
        return img, False


def resize_bbox(bbox, original_size, transformed_size):
    bbox = bbox.copy()
    y_scale = float(transformed_size[0]) / original_size[0]
    x_scale = float(transformed_size[1]) / original_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def horizontal_flip_bbox(bbox, transformed_size, flip):
    H, W = transformed_size
    bbox = bbox.copy()
    if flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox
