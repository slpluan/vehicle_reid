import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):    # modified by gu
    feat_dim = 2048

    if 'triplet' in cfg.LOSS_TYPE:
        triplet = TripletLoss(cfg.MARGIN, cfg.HARD_FACTOR)  # triplet loss

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'softmax' in cfg.LOSS_TYPE:
        if cfg.LOSS_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, numclasses:", num_classes)

    def loss_func(outputs, target):
        if cfg.LOSS_TYPE == 'triplet+softmax+center':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if cfg.LOSS_LABELSMOOTH == 'on':
                losses = []
                for output in outputs[0:3]:
                    loss = [xent(output, target)]
                loss = sum(loss) / len(loss)
                effective_loss = cfg.CE_LOSS_WEIGHT * loss
                losses.append(effective_loss)
                loss = triplet(outputs[3], target)[0]
                effective_loss = cfg.TRIPLET_LOSS_WEIGHT * loss
                losses.append(effective_loss)
                effective_loss = cfg.CENTER_LOSS_WEIGHT * center_criterion(outputs[3], target)
                losses.append(effective_loss)
                loss_sum = sum(losses)
                return loss_sum

            else:
                losses = []
                for output in outputs[1:3]:
                    loss = [F.cross_entropy(output, target)]
                loss = sum(loss) / len(loss)
                loss = loss + F.cross_entropy(outputs[0], target)  # 局部+全局softmax
                effective_loss = cfg.CE_LOSS_WEIGHT * loss
                losses.append(effective_loss)
                loss = triplet(outputs[3], target)[0]
                effective_loss = cfg.TRIPLET_LOSS_WEIGHT * loss
                losses.append(effective_loss)
                effective_loss = cfg.CENTER_LOSS_WEIGHT * center_criterion(outputs[3], target)
                losses.append(effective_loss)
                loss_sum = sum(losses)
                return loss_sum
        else:
            print('unexpected loss type')

    return loss_func, center_criterion