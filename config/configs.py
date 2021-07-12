from .default import DefaultConfig

class Config(DefaultConfig):


    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'baseline'
        self.DATA_DIR = 'E:/data/VehicleID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = 'F:/vehicle_reid/config/resnet50-19c8e357.pth'

        self.LOSS_TYPE = 'triplet+softmax+center'
        self.TEST_WEIGHT = './output/veri训练的结果/resnet50_130.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True


