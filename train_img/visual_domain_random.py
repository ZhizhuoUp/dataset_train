import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from train_img.VD_model import *
from VD_model import *
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    # eval_model()
# '''
    label = np.loadtxt('../Dataset/label/label_112_ang.csv')
    xyzyaw = np.copy(label)

    xyzyaw3 = np.copy(xyzyaw)[:, 3:5]

    img_pth = "../Dataset/yolo_test/"
    log_path = '../model/'

    data_num = 120000
    data_4_train = int(data_num*0.8)

    training_data = []
    for i in range(data_4_train):
        # print(i)
        img = plt.imread(img_pth + "img%d.png" % i)
        training_data.append(img)

    test_data = []
    for i in range(data_4_train,data_num):
        # print(i)
        img = plt.imread(img_pth + "img%d.png" % i)
        # print(np.shape(img))
        test_data.append(img)

    train_dataset = VD_Data(
        img_data = training_data, label_data = xyzyaw[0:data_4_train], transform=ToTensor())

    test_dataset = VD_Data(
        img_data = test_data, label_data =  xyzyaw[data_4_train:data_num], transform=ToTensor())
    # print(len(xyzyaw[data_4_train:data_num]))
    # print(data_4_train)

    num_epochs = 400
    BATCH_SIZE = 32
    learning_rate = 1e-4

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    model = ResNet50(img_channel=3, output_size=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma = 0.1)
    min_loss = + np.inf
    all_train_L, all_valid_L = [], []
    pre_array = []
    tar_array = []

    scaler = MinMaxScaler()
    scaler.fit(xyzyaw3)

    abort_learning = 0
    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        model.train()
        for batch in train_loader:
            img, xyzyaw = batch["image"], batch["xyzyaw"]
            img = img.to(device)
            xyzyaw_lw = xyzyaw[:, 3:5]
            xyzyaw_lw = scaler.transform(xyzyaw_lw)
            xyzyaw_sc = xyzyaw[:, 5:]
            xyzyaw = np.column_stack((xyzyaw_lw,xyzyaw_sc))

            xyzyaw = torch.from_numpy(xyzyaw)
            xyzyaw = xyzyaw.to(device)
            optimizer.zero_grad()
            pred_xyzyaw = model.forward(img)
            loss = model.loss(pred_xyzyaw, xyzyaw)

            loss.backward()
            optimizer.step()

            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                img, xyzyaw = batch["image"], batch["xyzyaw"]
                img = img.to(device)
                xyzyaw_lw = xyzyaw[:, 3:5]
                xyzyaw_lw = scaler.transform(xyzyaw_lw)
                xyzyaw_sc = xyzyaw[:, 5:]

                xyzyaw = np.column_stack((xyzyaw_lw, xyzyaw_sc))
                xyzyaw = torch.from_numpy(xyzyaw)
                xyzyaw = xyzyaw.to(device)

                pred_xyzyaw = model.forward(img)

                loss = model.loss(pred_xyzyaw, xyzyaw)

                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)
        scheduler.step()
        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L

            PATH = log_path + 'best_model_114.pt'

            torch.save({
                        'model_state_dict': model.state_dict(),
                        }, PATH)
            torch.save(model.state_dict(), PATH)

            abort_learning = 0
        else:
            abort_learning += 1

        np.savetxt(log_path + "training_L_yolo_114.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L_yolo_114.csv", np.asarray(all_valid_L))

        if abort_learning > 20:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0), "lr:", scheduler.get_last_lr())

    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(log_path + "lc_114.png")
    plt.show()
# '''




