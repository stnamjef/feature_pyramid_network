from .utils import *


class Transforms:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, bbox, label, difficult=None):
        raise NotImplementedError


class TransformsTrain(Transforms):
    def __init__(self, min_size, max_size):
        super(TransformsTrain, self).__init__(
            min_size, max_size
        )

    def __call__(self, img, bbox, label, difficult=None):
        original_size = img.shape[1:]

        # resize
        img = resize_image(img, self.min_size, self.max_size)
        transformed_size = img.shape[1:]

        # normalize
        img = normalize_image(img)

        # calc scale
        scale = transformed_size[0] / original_size[0]

        # resize bbox
        bbox = resize_bbox(bbox, original_size, transformed_size)

        # horizontal flip
        img, flip = horizontal_flip_image(img)
        bbox = horizontal_flip_bbox(bbox, transformed_size, flip)

        return img.copy(), bbox.copy(), label.copy(), scale


class TransformsTest(Transforms):
    def __init__(self, min_size, max_size):
        super(TransformsTest, self).__init__(
            min_size, max_size
        )

    def __call__(self, img, bbox, label, difficult=None):
        original_size = img.shape[1:]

        # resize
        img = resize_image(img, self.min_size, self.max_size)
        transformed_size = img.shape[1:]

        # normalize
        img = normalize_image(img)

        # calc scale
        scale = transformed_size[0] / original_size[0]

        return img, bbox, label, scale, original_size, difficult


class TransformsPlot(Transforms):
    def __init__(self, min_size, max_size):
        super(TransformsPlot, self).__init__(
            min_size, max_size
        )

    def __call__(self, img, bbox, label, difficult=None):
        # original img
        original_img = np.copy(img)

        # resize
        img = resize_image(img, self.min_size, self.max_size)

        # normalize
        img = normalize_image(img)

        # calc scale
        scale = img.shape[1] / original_img.shape[1]

        return original_img, img, scale, original_img.shape[1:]
