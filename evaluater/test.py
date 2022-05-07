import json
from pathlib import Path

import numpy as np
import pykitti
import torch
import torchvision
from PIL import Image
from scipy import sparse
from skimage.transform import resize
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from utils import map_fn

# dataset_dir = "../data/dataset"
# sequences = ["00","04"]
# datasets = [pykitti.odometry(dataset_dir, sequence) for sequence in sequences]

# print(type(datasets[0]))

# 为了生成depth error map
def imsave(mask,depth_prediction,depth_gt):
    '''
    mask - [B,1,256,512]
    depth_prediction - [B,1,256,512]
    depth_gt - [B,1,256,512]
    '''
    mask1 = mask[0,0,:,:]
    depth_pred1 = depth_prediction[0,0,:,:]
    depth_gt1 = depth_gt[0,0,:,:]
    abs_rel1 = torch.abs(depth_pred1 - depth_gt1) / depth_gt1
    plt.imsave("abs_rel1.png",abs_rel1)
    abs_rel1[mask1] = 0
    plt.imsave("abs_rel1_mask.png",abs_rel1)


    mask2 = mask[1,0,:,:]
    depth_pred2 = depth_prediction[1,0,:,:]
    depth_gt2 = depth_gt[1,0,:,:]
    abs_rel2 = torch.abs(depth_pred2 - depth_gt2) / depth_gt2

    plt.imsave("abs_rel2.png",abs_rel2)
    abs_rel2[mask2] = 0
    plt.imsave("abs_rel2_mask.png",abs_rel2)


    return None

# eval过程，查看kf和results，也可以查看gt lidar map
import torch
kf = data["keyframe"]
re = data["result"]

kf1 = torch.permute(kf[0,:,:,:],[1,2,0])
kf2 = torch.permute(kf[1,:,:,:],[1,2,0])

re1 = re[0,0,:,:]
re2 = re[1,0,:,:]

import matplotlib.pyplot as plt
plt.imshow(kf1+0.5)
plt.savefig("kf1.png")

plt.imshow(kf2+0.5)
plt.savefig("kf2.png")

# example过程，查看CV的plane的
import matplotlib.pyplot as plt
import torch

kf = torch.permute(keyframe.squeeze(),[1,2,0])
plt.imsave("kf.png",kf.numpy()+0.5)

prev = warped_images[:,0,:,:,:]
next = warped_images[:,1,:,:,:]

for i in range(1,6):
    img = torch.permute(prev[i*5,:,:,:],[1,2,0])
    plt.imsave(f"img{i*5}prev.png",img.numpy()+0.5)
img = torch.permute(prev[31,:,:,:],[1,2,0])
plt.imsave(f"img32prev.png",img.numpy()+0.5)

for i in range(1,6):
    img = torch.permute(next[i*5,:,:,:],[1,2,0])
    plt.imsave(f"img{i*5}next.png",img.numpy()+0.5)
img = torch.permute(next[31,:,:,:],[1,2,0])
plt.imsave(f"img32next.png",img.numpy()+0.5)