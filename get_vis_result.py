import os
import sys
from sys import path
path.append(r'F:\vehicle_reid')
from config import Config
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
sys.path.append('.')
from utils.logger import setup_logger
from model import make_model
from datasets import make_dataloader

from PIL import Image, ImageOps

import re
import numpy as np
import cv2
from utils.reranking import re_ranking

def add_border(img_file):
    image = Image.open(img_file)
    image = ImageOps.expand(image, border=5, fill='red')
    return image

def visualizer(test_img,top_k=10, img_size=[320,320]):
    k = 0
    i = 0
    input = False
    num_q, num_g = distmat.shape
    if num_g < top_k:
        top_k = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    while k < top_k:
        label, g_camid = map(int, pattern.search(img_path[indices[0][i]]).groups())
        query_label, q_camid = map(int, pattern.search(query_path).groups())
        if label == query_label and g_camid == q_camid:
            i = i + 1
        else:
            img = np.asarray(Image.open(img_path[indices[0][i]]).resize((img_size[1], img_size[0])))
            if label == query_label:
                figure = np.hstack((figure, img))
            else:
                image = add_border(img_path[indices[0][i]])
                img = np.asarray(image.resize((img_size[1], img_size[0])))
                figure = np.hstack((figure, img))
                input = True
            k = k + 1
            i = i + 1
    if input:
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        cv2.imwrite(Cfg.LOG_DIR + "/resultserror_CBAM/{}.png".format(test_img), figure)

    #     title=name
    # figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    # if not os.path.exists(Cfg.LOG_DIR+ "/results/"):
    #     print('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))
    # cv2.imwrite(Cfg.LOG_DIR+ "/results/{}-cam{}.png".format(test_img,camid),figure)

if __name__ == "__main__":
    Cfg = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_dataloader(Cfg)
    model = make_model(Cfg, num_classes)
    model.load_param(Cfg.TEST_WEIGHT)

    device = 'cuda'
    model = model.to(device)
    transform = T.Compose([
        T.Resize(Cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    log_dir = Cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(Cfg.PROJECT_NAME), log_dir)
    model.eval()
    for test_img in os.listdir(Cfg.QUERY_DIR):
        logger.info('Finding ID {} ...'.format(test_img))

        gallery_feats = torch.load('./log/gfeats.pth')
        img_path = np.load('./log/imgpath.npy')#gallery的地址
        # camid = np.load('./log/camids.npy')
        # q_camids = np.asarray(self.camids[:num_query])
        # g_camids = np.asarray(self.camids[num_query:])
        #
        # pids = np.load('./log/pids.npy')
        # q_pids = np.asarray(self.pids[:num_query])
        # g_pids = np.asarray(self.pids[num_query:])

        # print(gallery_feats.shape, len(img_path))
        query_img = Image.open(Cfg.QUERY_DIR + test_img)
        query_path = Cfg.QUERY_DIR + test_img
        input = torch.unsqueeze(transform(query_img), 0)
        input = input.to(device)
        with torch.no_grad():
            query_feat = model(input)
        distmat = re_ranking(query_feat, gallery_feats, k1=30, k2=10, lambda_value=0.2)

        # dist_mat = cosine_similarity(query_feat, gallery_feats)
        indices = np.argsort(distmat, axis=1)
        visualizer(test_img, top_k=10, img_size=Cfg.INPUT_SIZE)