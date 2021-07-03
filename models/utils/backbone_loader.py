import torch as t
import torch.nn as nn
import torchvision as tv
from utils.config import opt


def load_vgg16(pretrained=True):
    model = tv.models.vgg16(pretrained=pretrained)
    # drop the last max-pooling layer
    features = list(model.features)[:-1]
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    features = nn.Sequential(*features)

    top_layer = list(model.classifier)[:6]
    del top_layer[5]
    del top_layer[2]
    top_layer = nn.Sequential(*top_layer)

    return features, top_layer


def load_vgg16_as_fully_convolutional(pretrained=True):
    model = tv.models.vgg16(pretrained=pretrained)

    features = list(model.features)
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    dilation = 3

    conv6 = nn.Conv2d(512, 1024, 3, 1, padding=dilation, dilation=dilation)
    conv7 = nn.Conv2d(1024, 1024, 1, 1)

    # reshape pretrained weight
    conv6_weight = model.classifier[0].weight.view(4096, 512, 7, 7)
    conv6_bias = model.classifier[0].bias

    conv7_weight = model.classifier[3].weight.view(4096, 4096, 1, 1)
    conv7_bias = model.classifier[3].bias

    # subsampling weight
    conv6.weight = nn.Parameter(decimate(conv6_weight, m=[4, None, 3, 3]))
    conv6.bias = nn.Parameter(decimate(conv6_bias, m=[4]))

    conv7.weight = nn.Parameter(decimate(conv7_weight, m=[4, 4, None, None]))
    conv7.bias = nn.Parameter(decimate(conv7_bias, m=[4]))

    features += [conv6, nn.ReLU(True), conv7, nn.ReLU(True)]

    return nn.Sequential(*features)


def decimate(tensor, m):
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(
                dim=d, index=t.arange(start=0, end=tensor.size(d), step=m[d]).long()
            )

    return tensor


def load_resnet101(pretrained=True):
    model = tv.models.resnet101(pretrained=pretrained)

    layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4

    # fix block
    for p in layer0[0].parameters(): p.requires_grad = False
    for p in layer0[1].parameters(): p.requires_grad = False
    for p in layer1.parameters(): p.requires_grad = False

    # fix batchnorm
    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad = False

    layer0.apply(set_bn_fix)
    layer1.apply(set_bn_fix)
    layer2.apply(set_bn_fix)
    layer3.apply(set_bn_fix)
    layer4.apply(set_bn_fix)

    features = [layer0, layer1, layer2, layer3, layer4]

    return nn.Sequential(*features)
