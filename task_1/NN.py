# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:09:29 2021

@author: harry
"""

import torch
import torchvision
import torchvision.transforms as transforms
import cv2 #4.5.1
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class torch_loader(Dataset):
    
    def __int__(self, img_data, img_path, transform=None):
        self.img_path = img_path
        self.img_data = img_data
        self.transform = transform
    
    def __len__(self):
        return len(self.img_data)
    #https://github.com/utkuozbulak/pytorch-custom-dataset-examples#custom-dataset-fundamentals
    """
    Look at link above for pandas usage with loader issues
    """
    def __getitem__(self, index):#Change how items are collected does not need to be like this we have the df so just path and image name
        #img_name = os.path.join(self.img_path, self.img_data.loc[index, 'Images'])
        img_name = self.img_data[index]
        image = Image.open(img_name)
        image = image.resize((300,300))
        label = torch.tensor(img_name)
        if self.transform is not None:
            image = self.transform(image)
        return image, label



def labeler(image, labels, directory):
    for file in os.listdir(directory):
        if file=='blackgrass':
            for c in os.listdir(os.path.join(directory, file)):
                if c!= 'annotations':
                    image.append(c)
                    labels.append('blackgrass')
        if file=='charlock':
            for c in os.listdir(os.path.join(directory, file)):
                if c!= 'annotations':
                    image.append(c)
                    labels.append('charlock')
        if file=='cleavers':
            for c in os.listdir(os.path.join(directory, file)):
                if c!= 'annotations':
                    image.append(c)
                    labels.append('cleavers')
        if file=='fat hen':
            for c in os.listdir(os.path.join(directory, file)):
                if c!= 'annotations':
                    image.append(c)
                    labels.append('fat hen')
        if file=='maize':
            for c in os.listdir(os.path.join(directory, file)):
                if c!= 'annotations':
                    image.append(c)
                    labels.append('maize')
        if file=='wheat':
            for c in os.listdir(os.path.join(directory, file)):
                if c!= 'annotations':
                    image.append(c)
                    labels.append('wheat')
    return image, labels

def merge(image, labels):
    data = {'Images': image, 'labels':labels}
    data = pd.DataFrame(data)
    lb = LabelEncoder()
    data['encoded_labels'] = lb.fit_transform(data['labels'])
    return data

def img_display(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    np.img = np.transpose(npimg, (1,2,0))

class NN(nn.Module):
    
    def __init__(self):
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3,stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64,kernel_size=3,stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64,kernel_size=3,stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.4)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*5*5,512 )
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        
    def forward(self, x):
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batchnorm2(self.pool(x)))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(-1, 64*5*5) # Flatten layer
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim = 1)
        return x

def main():
    test_dir = './test'
    vald_dir = './val'
    train_dir = './train'


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    
    image_train, labels_train, image_test, labels_test, image_val, labels_val = [], [], [], [], [], []
    image_train, labels_train = labeler(image_train, labels_train, train_dir)
    image_test, labels_test = labeler(image_test, labels_test, test_dir)
    image_val, labels_val = labeler(image_val, labels_val, vald_dir)
    
    train_data = merge(image_train, labels_train)
    test_data = merge(image_test, labels_test)
    val_data = merge(image_val, labels_val)
    print(train_data)
    
    batch_size =128
    train_sampler = SubsetRandomSampler(len(train_data))
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torch_loader(train_data, train_dir, transform)#rework class
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    plant_types = {0:'blackgrass', 1:'charlock', 2:'cleavers', 3:'fat hen', 4:'maize', 5:'wheat'}
    fig, axis = plt.subplots(3,5, figsize=(15,10))
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            image, label = images[i], labels[i]
            ax.imshow(img_display(image))
            ax.set(title=f"{plant_types[label.item()]}")
    
    
    model = NN().to(device)

main()