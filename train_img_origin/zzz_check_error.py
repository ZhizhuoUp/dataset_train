import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from train_img.VD_model import *
from VD_model_ori_x1y1x2y2 import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import cv2
import torchvision.transforms.functional as TF
# from pre_view import *
from sklearn.preprocessing import MinMaxScaler


torch.manual_seed(42)

class VD_Data(Dataset):
    def __init__(self, img_data, label_data, transform=None):
        self.img_data = img_data
        self.label_data = label_data
        self.transform = transform

    def __getitem__(self, idx):
        img_sample = self.img_data[idx]
        label_sample = self.label_data[idx]

        sample = {'image': img_sample, 'xyzyaw': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_data)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_sample, label_sample = sample['image'], sample['xyzyaw']
        img_sample = np.asarray([img_sample])

        # print(type(img_sample))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print(img_sample.shape)
        img_sample = np.squeeze(img_sample)
        img_sample = img_sample[:,:,:3]
        # print(img_sample.shape)

        image = img_sample.transpose((2, 0, 1))
        # print(image.shape)
        img_sample = torch.from_numpy(image)

        return {'image': img_sample,
                'xyzyaw': torch.from_numpy(label_sample)}

def check_dataset():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    log_path = 'model/'
    label = np.loadtxt('../Dataset/label/label_221_combine.csv')[:, :5]
    # print('this is label', label)

    xyzyaw = np.copy(label)

    scaler = MinMaxScaler()
    scaler.fit(label)
    print(scaler.data_max_)
    print(scaler.data_min_)

    model = ResNet50(img_channel=3, output_size=5).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load('../model/best_model_222_combine.pt', map_location='cuda:0'))
    # add map_location='cuda:0' to run this model trained in multi-gpu environment on single-gpu environment
    model.eval()

    img_pth = "../Dataset/yolo_221_combine/"
    data_num = 30000
    data_4_train = int(data_num * 0.8)

    test_data = []
    for i in range(data_4_train,data_num):
        # print(i)
        img = plt.imread(img_pth + "img%d.png" % i)
        # print(np.shape(img))
        test_data.append(img)

    test_dataset = VD_Data(
        img_data=test_data, label_data=xyzyaw[data_4_train:data_num], transform=ToTensor())

    BATCH_SIZE = 32

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    with torch.no_grad():
        total_loss = []
        for batch in test_loader:
            img, x1y1x2y2l = batch["image"], batch["xyzyaw"]

            # ############################## test the shape of img ##############################
            # img_show = img.cpu().detach().numpy()
            # print(img_show[0].shape)
            # temp = img_show[3]
            # temp_shape = temp.shape
            # temp = temp.reshape(temp_shape[1], temp_shape[2], temp_shape[0])
            # print(temp.shape)
            # cv2.namedWindow("affasdf",0)
            # cv2.imshow('affasdf', temp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ############################## test the shape of img ##############################

            img = img.to(device)
            pred_x1y1x2y2l = model.forward(img)

            target_x1y1x2y2l = scaler.transform(x1y1x2y2l)
            target_x1y1x2y2l = torch.from_numpy(target_x1y1x2y2l)
            target_x1y1x2y2l = target_x1y1x2y2l.to(device)
            # print('this is pred\n', pred_x1y1x2y2l)
            # print('this is target\n', target_x1y1x2y2l)
            loss = model.loss(pred_x1y1x2y2l, target_x1y1x2y2l)

            if loss.item() < 0.1:
                print(loss)
                pred_x1y1x2y2l = pred_x1y1x2y2l.cpu().detach().numpy()
                # print('this is', pred_x1y1x2y2l)
                pred_x1y1x2y2l = scaler.inverse_transform(pred_x1y1x2y2l)
                print('this is pred after scaler\n', pred_x1y1x2y2l)
                print('this is target after scaler\n', x1y1x2y2l)

            # pred_xyzyaw_ori[:, 0] = pred_xyzyaw_ori[:, 0] * np.pi / 180

            total_loss.append(loss.item())

        total_loss = np.asarray(total_loss)

if __name__ == "__main__":
    check_dataset()