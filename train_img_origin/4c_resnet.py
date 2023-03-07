import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train_img.VD_model import *
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
    def __init__(self, img_data1, img_data2, label_data, transform=None):
        self.img_data1 = img_data1
        self.img_data2 = img_data2
        self.label_data = label_data
        self.transform = transform

    def __getitem__(self, idx):
        img_sample1 = self.img_data1[idx]
        # print(img_sample1.shape)
        img_sample2 = self.img_data2[idx]
        # print(img_sample2.shape)
        img_sample = np.dstack((img_sample1, img_sample2))
        # print(img_sample.shape)
        label_sample = self.label_data[idx]

        sample = {'image': img_sample, 'xyzyaw': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_data1)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img_sample, label_sample = sample['image'], sample['xyzyaw']
        img_sample = np.asarray([img_sample])

        # print(type(img_sample))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        print(img_sample.shape)
        img_sample = np.squeeze(img_sample)
        print(img_sample.shape)

        image = img_sample.transpose((2, 0, 1))
        print(image.shape)
        img_sample = torch.from_numpy(image)
        #
        # plt.imshow(img_sample.permute(1, 2, 0))
        # plt.show()

        # if random.random() > 0.5:
        #     brightness_f = random.randint(1,5)
        #     img_sample = TF.adjust_brightness(img_sample, brightness_f)
        #
        # if random.random() > 0.5:
        #     # print("c")
        #     contrast_f = random.randint(1,50)
        #     img_sample = TF.adjust_contrast(img_sample, contrast_f)
        #
        # if random.random() > 0.5:
        #     # print("g")
        #     gamma_f = random.randint(0,30)
        #     img_sample = TF.adjust_gamma(img_sample, gamma_f)
        #
        # if random.random() > 0.5:
        #     # print("blur")
        #     sig_r_xy = random.uniform(0.1, 5)
        #     win_r = 2 * random.randint(1, 20) + 1
        #     img_sample = TF.gaussian_blur(img_sample, win_r, sig_r_xy)



        # plt.imshow(img_sample.permute(1, 2, 0))
        # plt.show()
        # input("come on")
        # return {'image': torch.from_numpy(img_sample),
        #         'xyzyaw': torch.from_numpy(label_sample)}

        return {'image': img_sample,
                'xyzyaw': torch.from_numpy(label_sample)}

#
def eval_img(img_array):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    log_path = 'model/'
    label = np.loadtxt('Dataset/label/label_test.csv')[:,2:]

    xyzyaw = np.copy(label)

    model = ResNet18(img_channel=3, output_size=3).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load(log_path + 'best_model_yolo.pt'))
    model.eval()

    test_data = []
    image = plt.imread(img_array)
    # print('shape',img_array.shape())
    test_data.append(image)

    test_dataset = VD_Data(
        img_data=test_data, label_data=xyzyaw, transform=ToTensor())

    BATCH_SIZE = 1

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    with torch.no_grad():

        for batch in test_loader:
            img, xyzyaw = batch["image"], batch["xyzyaw"]
            img = img.to(device)
            scaler = MinMaxScaler()
            scaler.fit(label)
            pred_xyzyaw = model.forward(img)
            pred_xyzyaw_ori = pred_xyzyaw.cpu().detach().numpy()
            pred_xyzyaw_ori = scaler.inverse_transform(pred_xyzyaw_ori)[0]

    # print(pred_xyzyaw_ori)
    return pred_xyzyaw_ori[0], pred_xyzyaw_ori[1], pred_xyzyaw_ori[2]


def eval_model():

    log_path = '../model/'
    model = ResNet18(img_channel=3, output_size=3).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load(log_path + 'best_model_yolo.pt'))
    model.eval()
    valid_L = []
    pre_array = []
    tar_array = []

    label = np.loadtxt('../Dataset/label/label_yolo3.csv')
    # label = np.delete(label,[0,1,2,12,13,14,24,25,26], axis = 1)
    # print(len(label[0]))
    xyzyaw = np.copy(label)


    # for i in range(len(label)):
    #     if label[i, 3] < (0):
    #         new_label[i, 3] = (label[i, 3] + (np.pi * 2)) % (np.pi/2)
    #     elif label[i, 3] > (0):
    #         new_label[i, 3] = label[i, 3] % (np.pi/2)

    xyzyaw3 = np.copy(xyzyaw)[:,2:]


    # print(xyzyaw)
    img_pth = "../res_set2/"
    log_path = '../model/'


    data_num = 3

    test_data = []
    for i in range(data_num):
        image = plt.imread(img_pth + "IMG%d.png" % i)[:,:,0:3]
        plt.imshow(image)
        plt.show()
        time.sleep(1)
        # img = plt.imread("../img.png")
        # image = img[112:368, 192:448]
        # print(np.shape(img))
        test_data.append(image)

    test_dataset = VD_Data(
        img_data = test_data, label_data =  xyzyaw[0:data_num], transform=ToTensor())

    BATCH_SIZE = 32

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    with torch.no_grad():

        for batch in test_loader:
            img, xyzyaw = batch["image"], batch["xyzyaw"]
            img = img.to(device)
            xyzyaw = xyzyaw[:, 2:]
            scaler = MinMaxScaler()
            scaler.fit(xyzyaw3)
            xyzyaw = scaler.transform(xyzyaw)
            xyzyaw_ori = xyzyaw
            xyzyaw_ori = xyzyaw_ori
            xyzyaw_ori = scaler.inverse_transform(xyzyaw_ori)
            xyzyaw = torch.from_numpy(xyzyaw)
            xyzyaw = xyzyaw.to(device)

            pred_xyzyaw = model.forward(img)
            pred_xyzyaw_ori = pred_xyzyaw
            pred_xyzyaw_ori = pred_xyzyaw_ori.cpu().detach().numpy()
            pred_xyzyaw_ori = scaler.inverse_transform(pred_xyzyaw_ori)


            xyzyaw2 = xyzyaw_ori
            pred_xyzyaw2 = pred_xyzyaw_ori

            for i in range(len(pred_xyzyaw2)):
                pre_array.append(pred_xyzyaw2[i])
                tar_array.append(xyzyaw2[i])

            loss = model.loss(pred_xyzyaw, xyzyaw)

            valid_L.append(loss.item())

    avg_valid_L = np.mean(valid_L)
    np.savetxt(log_path + "pre_data.csv", np.asarray(pre_array, dtype= np.float64))
    np.savetxt(log_path + "tar_data.csv", np.asarray(tar_array, dtype= np.float64))
    # scheduler.step()
    print('Testing_Loss :\t' + str(avg_valid_L))


# def cam_pred():
#
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     print("Device:", device)
#
#     log_path = '../model/'
#     model = ResNet50(img_channel=3, output_size=27).to(device)
#     model.load_state_dict(torch.load(log_path + 'best_model_corner_noise.pt'))
#     model.eval()
#
#     data_num = 1
#
#     test_data = []
#     for i in range(data_num):
#         # img = plt.imread(img_pth + "IMG%d.png" % i)
#         img = plt.imread("../img.png")
#         image = img[112:368, 192:448]
#         # print(np.shape(img))
#         test_data.append(image)
#
#     test_dataset = VD_Data(
#         img_data=test_data, transform=ToTensor())
#
#     BATCH_SIZE = 1
#
#     test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
#                              shuffle=False, num_workers=0)
#
#     with torch.no_grad():
#
#         for batch in test_loader:
#             img = batch["image"]
#             img = img.to(device)
#             pred_xyzyaw = model.forward(img)
#             print(pred_xyzyaw)
#
#             for i in range(3):
#                 o = pred_xyzyaw[0 + 9*i:9 + 9*i]
#
#                 x, y, yaw = corn2pose(o[0:2],o[2:4], o[4:6], o[6:8])
#
#                 print(x, y, yaw)


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device:", device)

    # eval_model()
# '''
    label = np.loadtxt('../Dataset/label/label_1119.csv')
    xyzyaw = np.copy(label)

    # for i in range(len(label)):
    #     if label[i, 3] < (0):
    #         new_label[i, 3] = (label[i, 3] + (np.pi * 2)) % (np.pi/2)
    #     elif label[i, 3] > (0):
    #         new_label[i, 3] = label[i, 3] % (np.pi/2)

    xyzyaw3 = np.copy(xyzyaw)[:,2:]
    # print(new_label[:, 3])
    # print(label[:, 3])

    # print(xyzyaw)
    img_pth = "../real_test2/"
    img2_pth = "../real_test3/"
    log_path = '../model/'


    data_num = 10
    data_4_train = int(data_num*0.8)

    training_data1 = []
    training_data2 = []
    for i in range(data_4_train):
        img = plt.imread(img_pth + "IMG%d.png" % i)
        training_data1.append(img)
        img2 = plt.imread(img2_pth + "IMG%d.png" % i)
        training_data2.append(img2)

    test_data1 = []
    test_data2 = []
    for i in range(data_4_train, data_num):
        img = plt.imread(img_pth + "IMG%d.png" % i)
        # print(np.shape(img))
        test_data1.append(img)
        img2 = plt.imread(img2_pth + "IMG%d.png" % i)
        test_data2.append(img2)

    train_dataset = VD_Data(
        img_data1 = training_data1, img_data2 = training_data2, label_data = xyzyaw[:data_4_train], transform=ToTensor())

    test_dataset = VD_Data(
        img_data1 = test_data1, img_data2 = test_data2, label_data =  xyzyaw[data_4_train:data_num], transform=ToTensor())
    # print(len(xyzyaw[data_4_train:data_num]))
    # print(data_4_train)



    num_epochs = 400
    BATCH_SIZE = 32
    learning_rate = 1e-4



    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)

    # model.eval()

    model = ResNet50(img_channel=4, output_size=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # model.load_state_dict(torch.load(log_path + 'best_model101_50k.pt'))
    # optimizer.load_state_dict(torch.load['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.1)
    min_loss = + np.inf
    all_train_L, all_valid_L = [], []
    pre_array = []
    tar_array = []

    abort_learning = 0
    for epoch in range(num_epochs):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        model.train()
        for batch in train_loader:
            img, xyzyaw = batch["image"], batch["xyzyaw"]
            img = img.to(device)
            xyzyaw = xyzyaw[:,2:]
            scaler = MinMaxScaler()
            scaler.fit(xyzyaw3)
            xyzyaw = scaler.transform(xyzyaw)
            xyzyaw = torch.from_numpy(xyzyaw)
            xyzyaw = xyzyaw.to(device)
            # print(len(xyzyaw))
            optimizer.zero_grad()
            pred_xyzyaw = model.forward(img)
            # print(len(pred_xyzyaw))
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
                xyzyaw = xyzyaw[:, 2:]
                scaler = MinMaxScaler()
                scaler.fit(xyzyaw3)
                xyzyaw = scaler.transform(xyzyaw)
                xyzyaw = torch.from_numpy(xyzyaw)
                xyzyaw = xyzyaw.to(device)

                pred_xyzyaw = model.forward(img)

                # error_yaw = torch.mean(pred_xyzyaw[:,0] - xyzyaw[:,0])
                # error_l = torch.mean(pred_xyzyaw[:,1] - xyzyaw[:,1])
                # error_w = torch.mean(pred_xyzyaw[:,2] - xyzyaw[:,2])


                loss = model.loss(pred_xyzyaw, xyzyaw)

                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)
        scheduler.step()
        if avg_valid_L < min_loss:
            print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L

            PATH = log_path + 'best_model_yolo_4c.pt'

            # torch.save({
            #             'model_state_dict': model.state_dict(),
            #             }, PATH)
            torch.save(model.state_dict(), PATH)

            abort_learning = 0
        else:
            abort_learning += 1

        np.savetxt(log_path + "training_L_yolo_4c.csv", np.asarray(all_train_L))
        np.savetxt(log_path + "testing_L_yolo_4c.csv", np.asarray(all_valid_L))

        if abort_learning > 20:
            break
        t1 = time.time()
        print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", scheduler.get_last_lr())

    # all_train_L = np.loadtxt("../model/training_L_single.csv")
    # all_valid_L = np.loadtxt("../model/testing_L_single.csv")
    plt.plot(np.arange(len(all_train_L)), all_train_L, label='training')
    plt.plot(np.arange(len(all_valid_L)), all_valid_L, label='validation')
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(log_path + "lc_yolo_4c.png")
    plt.show()
# '''




