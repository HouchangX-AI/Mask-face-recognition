import torch
import numpy as np 
import torch.nn.functional as F
import os
import time
import dlib 
import cv2 
# from Data_loader.Data_loader_facenet import train_dataloader
# from Data_loader.Data_loader_train import TrainDataset
from config import config
# # import torchvision.transforms as transforms 
# from tqdm import tqdm

# for animal in dataset:
#     # print(animal)
#     print('啊啊'*10)
#     time.sleep(1)

# a = np.array((0,5), dtype=np.float32)
# b = np.array((0,7), dtype=np.float32)
# c = [a,b]
# d = [a,b]

# print(torch.cat([c, d]))

# a = [1.32,0.78,0.51]
# a = np.array(a, dtype=np.float64)
# a = torch.from_numpy(a)
# b = [0.77,0.45,0.62]
# b = np.array(b, dtype=np.float64)
# b = torch.from_numpy(b)

# print(a)
# print(torch.cat([a, b]))
# print(torch.mean(a))



# a = (True)
# print(a)
# print(type(a))

# validation_images = np.load('Datasets/test_pairs.npy')
# a, b, c = validation_images[400]
# c = 1
# print(c)
# print(type(c))
# c = bool(c)
# print(type(c))
# print(c)



import torch
# from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import csv
import numpy as np

# from Models.Model_for_facenet import model
from Models.Attention_resnet_lossinforward import ResNet, BasicBlock
model = ResNet(BasicBlock, [3, 4, 6, 3])

for name, m in model.named_modules():
    print(name)

# num_sa1 = model.fc.in_features
# for name, param in model.named_parameters():
#     print(name)
    # if 'sa1' not in name and 'sa2' not in name:
    #     param.requires_grad = False
    # else:
    #     print(name)
    #     print(param.size())
    #     print(param)
    #     print("#"*30)

# checkpoint = torch.load('Model_training_checkpoints/model_resnet34_cheahom_triplet_epoch_17_roc0.9339.pt')
# pretrained_dict = checkpoint['model_state_dict']
# model_dict=model.state_dict()

# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

