from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # model & backbone
    model = 'fpn'
    backbone = 'vgg16'
    n_features = 1
    pretrained_dir = './model_zoo'

    # dataset
    data_dir = './'
    min_size = 600   # image resize
    max_size = 1000  # image resize
    n_workers_train = 8
    n_workers_test = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # score thresh & IoU thresh for NMS
    nms_thresh = 0.3
    score_thresh = 0.05

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # training
    epoch = 15
    epoch_decay = 10

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            if k == 'data_dir' and v == None:
                raise ValueError('--data_dir must be given.')
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict(), sort_dicts=False)
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
