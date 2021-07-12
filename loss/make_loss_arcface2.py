import torch.nn.functional as F
import torch
from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .arcface import ArcFace


def make_loss(cfg, num_classes):    # modified by gu
    feat_dim = 2048
    loss_f = torch.nn.CrossEntropyLoss()
    if 'softmax' in cfg.LOSS_TYPE:
        if cfg.LOSS_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, numclasses:", num_classes)

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss


    def loss_func(arc_score,cls_score, feat, target):
        if cfg.LOSS_TYPE == 'arcface+softmax+center':
            return cfg.CE_LOSS_WEIGHT * xent(cls_score, target) + \
                       cfg.ARCFACE_LOSS_WEIGHT * loss_f(arc_score, target)+ \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('unexpected loss type')

    return loss_func, center_criterion