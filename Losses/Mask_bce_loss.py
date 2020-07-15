# import minpy.numpy as np 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2


# class Mask_BCE_Loss(Function):
#     def __init__(self):
#         super(TripletLoss, self).__init__()
#         # self.mask = mask
#         # self.predict = predict
#     def forward(self, mask, predict):

#         loss = list()
#         for one_pre_forward, one_mask in zip(predict, mask):
#             for one_pre_mask in one_pre_forward:
#                 shape = one_pre_mask.shape
#                 one_mask = cv2.resize(one_mask, shape)
#                 one_mask = torch.from_numpy(one_mask)
#                 one_loss = F.binary_cross_entropy(predict, one_mask)
#                 loss.append(one_loss)

#         Loss = torch.mean(loss)
#         return Loss

class Attention_loss(nn.Module):
    def forward(self, hot_map, mask):
        Loss = list()

        for one_hot_map, one_mask in zip(hot_map, mask):

            shape = one_hot_map.shape[2]

            one_mask = one_mask.cpu()
            one_mask = np.array(one_mask)/255
            # print(one_mask.shape)
            # one_mask = np.transpose(one_mask, (1, 2, 0))
            # one_mask = cv2.cvtColor(one_mask, cv2.COLOR_RGB2GRAY)
            one_mask = cv2.resize(one_mask, (shape, shape))
            one_mask = one_mask[np.newaxis, :]
            # print(one_mask.shape)

            one_mask = torch.from_numpy(one_mask)
            one_mask = one_mask.cuda().float()

            one_loss = F.binary_cross_entropy(one_hot_map, one_mask)
            Loss.append(one_loss)

        Loss = np.array(Loss, dtype=np.float64)
        Loss = torch.from_numpy(Loss)
        return Loss



# class Attention_loss(nn.Module):
#     def forward(self, hot_map_list, mask):
#         Loss = []

#         mask = mask.cpu()
#         mask = np.array(mask)
#         # hot_map_list = np.array(hot_map_list)
#         hot_map_list = hot_map_list

#         for hot_map in hot_map_list:
#             for one_mask, one_hotmap in zip(mask, hot_map):
#                 one_hotmap = one_hotmap.cpu()
#                 # one_hotmap = np.array(one_hotmap)
#                 shape = one_hotmap.shape[2]
#                 one_mask = cv2.cvtColor(one_mask,cv2.COLOR_RGB2GRAY)
#                 one_mask = cv2.resize(one_mask, (shape, shape))

#                 one_mask = torch.from_numpy(one_mask)
#                 one_mask = one_mask.cuda()
#                 one_hotmap = torch.from_numpy(one_hotmap)
#                 one_hotmap = one_hotmap.cuda()

#                 loss = F.binary_cross_entropy(one_hotmap, one_mask)
#                 Loss.append(loss)
#         Loss = np.array(Loss, dtype=np.float64)
#         Loss = torch.from_numpy(Loss)
#         return torch.mean(Loss)










