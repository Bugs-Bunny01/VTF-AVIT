import os
import re
import random
import cv2
import torch
from torch.utils import data
from torch import nn

# For creating a custom dataset: it needs to contain three funcs: __init__, __len__, __getitem__
# Default: no scale ratio
class Tactile_Vision_dataset(data.Dataset):
    def __init__(self, Fruit_type = ['apple'], label_encoding= {'apple':'0'},
                 Tactile_scale_ratio = 1, Visual_scale_ratio = 0.25, video_length = 8, data_path='D:\dl\slip_dataset\Fruit_dataset'):
        self.data_path = data_path
        self.label_files = []
        self.train_data = []
        self.Fruit_type = Fruit_type
        self.label_type = label_encoding
        self.Tactile_scale_ratio = Tactile_scale_ratio
        self.Visual_scale_ratio = Visual_scale_ratio
        self.video_length = video_length
        for fruit in self.Fruit_type:
            data_path_fruit = data_path + '/' + fruit
            root = data_path + '/' + fruit
            label_file = "labels.txt"
            fp = open(os.path.join(root, label_file), 'r')
            lines = fp.readlines()
            self.train_data.extend([fruit + '/' + line.replace('\n','') + " " + label_encoding[fruit] for line in lines])
            self.train_data.sort()
        # print(self.train_data)
        print(len(self.train_data))
    def __len__(self):
        return len(self.train_data)

    global train_data_0_one
    train_data_0_one = []
    global train_data_0_two
    train_data_0_two = []
    global train_data_0_three
    train_data_0_three = []
    global train_data_0_four
    train_data_0_four = []
    def __getitem__(self, index):  #need to be defined to let data_loader work\
        train_data = self.train_data[index].split()
        # print(train_data[0])
        # print(train_data[1])
        # print(train_data[2])
        # print(train_data[3])
        status = int(train_data[1][0])
        label = torch.tensor([status]).long()
        label = torch.squeeze(label)
        output_tactile_imgs_sliding = []
        output_rgb_imgs_sliding = []
        data_dir = self.data_path + '/' + train_data[0]
        # print(data_dir)
        rgb_img_paths_sliding = []
        for root, dirs, files in os.walk(data_dir + '/Sliding/RealSense' , topdown=True):
            for file in files:
                if file.endswith('.jpg'):  # select camera images
                    rgb_img_paths_sliding.append(os.path.join(root, file))
        rgb_img_paths_sliding.sort()
        rgb_img_paths_selected_sliding = []
        if (train_data[0] in train_data_0_one) and (train_data[0] in train_data_0_two) and (
                train_data[0] in train_data_0_three) and (train_data[0] in train_data_0_four):
            start_image = [0,1,2,3,4,5,6,7]
        if (train_data[0] in train_data_0_one) and (train_data[0] in train_data_0_two) and (
                train_data[0] in train_data_0_three) and (train_data[0] not in train_data_0_four):
            start_image = [0,1,2,3,4,5,6,7]
            train_data_0_four.append(train_data[0])
        if (train_data[0] in train_data_0_one) and (train_data[0] in train_data_0_two) and (
                train_data[0] not in train_data_0_three) and (train_data[0] not in train_data_0_four):
            start_image = [0,1,2,3,4,5,6,7]
            train_data_0_three.append(train_data[0])
        if (train_data[0] in train_data_0_one) and (train_data[0] not in train_data_0_two) and (
                train_data[0] not in train_data_0_three) and (train_data[0] not in train_data_0_four):
            start_image = [0,1,2,3,4,5,6,7]
            train_data_0_two.append(train_data[0])
        if (train_data[0] not in train_data_0_one) and (train_data[0] not in train_data_0_two) and (
                train_data[0] not in train_data_0_three) and (train_data[0] not in train_data_0_four):
            start_image = [6,7,8,9,10,11,12,13]
            train_data_0_one.append(train_data[0])
        # print(start_image)
        # customized image index sequence
        # select given image for the purpose for comparison
        # try to select discrete images
        index = 0
        while(len(rgb_img_paths_selected_sliding) <  self.video_length):  # 8 frames per time (LSTM)
            if index in start_image:
                rgb_img_paths_selected_sliding.append(rgb_img_paths_sliding[index])
            index += 1
            # 滑动
        index = 0
        for rgb_img_path in rgb_img_paths_selected_sliding:
            tactile_img_path = rgb_img_path.replace('RealSense', 'Gelsight')
            rgb_img = cv2.imread(rgb_img_path)
            tactile_img = cv2.imread(tactile_img_path)
            visual_size = rgb_img.shape  # 480, 640, 3 (width, height, channel)
            tactile_size = tactile_img.shape
            # new width / new height = 480 / 640 * scale_percent

            # commented lines are for the attn visual using pretrained K400 models
            # the only goal is to test the attn method on the released dataset
            rgb_img_resized = cv2.resize(rgb_img, (int(visual_size[1] * self.Visual_scale_ratio), int(visual_size[0] * self.Visual_scale_ratio)), interpolation = cv2.INTER_AREA)
            tactile_img_resized = cv2.resize(tactile_img, (int(tactile_size[1] * self.Tactile_scale_ratio), int(tactile_size[0] * self.Tactile_scale_ratio)), interpolation=cv2.INTER_AREA)
            visual_size = rgb_img_resized.shape
            tactile_size = tactile_img_resized.shape
            # size = tactile_img_resized.shape
            rgb_img_tensor = torch.from_numpy(rgb_img_resized.transpose(2,0,1)).float()
            # rgb_img_tensor = torch.from_numpy(rgb_img_resized.reshape(3, 224, 224)).float()

            #turn into a tensor (3, 240, 320)  -> resized one
            tactile_img_tensor = torch.from_numpy(tactile_img_resized.transpose(2,0,1)).float()


            if index == 0:
                output_rgb_imgs_sliding = rgb_img_tensor[None,:]
                output_tactile_imgs_sliding = tactile_img_tensor[None,:]
            else:
                output_rgb_imgs_sliding = torch.cat([output_rgb_imgs_sliding, rgb_img_tensor[None,:]], dim=0)
                output_tactile_imgs_sliding = torch.cat([output_tactile_imgs_sliding, tactile_img_tensor[None,:]], dim=0)
            index += 1
        # print(output_rgb_imgs_sliding.transpose(0, 1).shape)  # [8, 3, 120, 160]
        return output_rgb_imgs_sliding.transpose(0, 1),output_tactile_imgs_sliding.transpose(0, 1), label # rgb images; visual images; label

if __name__ == "__main__":
    # set a global dataset path
    train_dataset = Tactile_Vision_dataset(Fruit_type = ['apple'], label_encoding={'apple':'0'}, data_path = 'D:/dl/test')
    # train_dataset = Tactile_Vision_dataset(Fruit_type = ['apple'], label_encoding={'apple':'0'}, data_path = 'D:/desk/Fruit_dataset')
    # print(len(train_dataset))
    data = train_dataset[30][0]
    label = train_dataset[30][4]
    thresh = train_dataset[30][5]

    # print (label)
    # print (thresh)
    '''
    print(" ")
    print(train_dataset[20][0].shape)
    print(" ")
    print(train_dataset[20][1].shape)
    print(" ")
    print(train_dataset[20][2].shape)
    print(" ")
    print(train_dataset[20][3].shape)
    print(" ")
    print(train_dataset[20][4].shape)
    print(" ")
    print(train_dataset[20][5].shape)
    print(" ")
    '''
    # for i in range(2):
    #     output_rgb_imgs, output_tactile_imgs, label = train_dataset[i]
    #     print(output_rgb_imgs[0].shape)
    #     print(output_tactile_imgs[0].shape)
    #
    # for i in range(1000):
    #     train_dataset[i]
    #     print(i)
