import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np

import sys
sys.path.append("..")
# sys.path.append("/Users/handsomeb/MChan/Pg_ANU/S3_8603/MonoRec")

from data_loader.kitti_odometry_dataset import KittiOdometryDataset
from data_loader.nuscenes_dataset import NuscenesDataset
from data_loader.nuscenes_dataset2 import NuscenesDataset2
from data_loader.nuscenes_dataset3 import NuscenesDataset3
from model.monorec.monorec_model import MonoRecModel
from utils import unsqueezer, map_fn, to
from matplotlib import cm
from utils import operator_on_dict, median_scaling
import model.metric as module_metric


def eval_metrics(data_dict):
    mets = [
      "abs_rel_sparse_metric",
      "sq_rel_sparse_metric",
      "rmse_sparse_metric",
      "rmse_log_sparse_metric",
      "a1_sparse_metric",
      "a2_sparse_metric",
      "a3_sparse_metric"
    ]
    metrics = [getattr(module_metric, met) for met in mets]
    acc_metrics = np.zeros(len(metrics))  # (7,) - 7个0
    for i, metric in enumerate(metrics):  # 对于每个metrics都for一次
        # if self.median_scaling:
        #     data_dict = median_scaling(data_dict)
        # 运行function, abs_rel_sparse_metrics
        acc_metrics[i] += metric(data_dict, None, 80)
        #self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
    if np.any(np.isnan(acc_metrics)):
        acc_metrics = np.zeros(len(metrics))
        valid = np.zeros(len(metrics))
    else:
        valid = np.ones(len(metrics))
    return acc_metrics, valid

# 0. 准备工作
target_image_size = (256, 512) # 测试图片大小; 原始大小为(370,1226)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = NuscenesDataset()
# dataset = NuscenesDataset2()
# dataset = NuscenesDataset3()
# dataset = KittiOdometryDataset("data/kitti", sequences=["07"], target_image_size=target_image_size, frame_count=2,
#                                depth_folder="image_depth_annotated", lidar_depth=True, use_dso_poses=True,
#                                use_index_mask=None)
# Next three lines are a hack required because Kitti files are incomplete
# dataset._dataset_sizes = [1000]
# dataset._datasets[0].cam2_files = [f"data/kitti/sequences/07/image_2/{i:06d}.png" for i in range(dataset._dataset_sizes[0])] # 'data/kitti/sequences/07/image_2/0~999.png'
# dataset._datasets[0].cam3_files = [f"data/kitti/sequences/07/image_3/{i:06d}.png" for i in range(dataset._dataset_sizes[0])]

## 设置pretrained model位置
# checkpoint_location = Path("../saved/checkpoints/monorec_depth_ref.pth")
checkpoint_location = Path("../saved/checkpoints/monorec_depth_ref.pth")

## 定义inv depth范围
inv_depth_min_max = [0.33, 0.0025]  # 正常的depth range 3～400

## 加载pretrained model
print("Initializing model...")
monorec_model = MonoRecModel(checkpoint_location=checkpoint_location, inv_depth_min_max=inv_depth_min_max)
monorec_model.to(device)
monorec_model.eval()

print("Fetching data...")
index = 0
# Corresponds to image index 169

batch, depth = dataset.__getitem__(index)  # batch - dict; depth - tensor(1,W=256,H=512)
'''
batch - {
    keyframe - tensor (3,256,512) - 3张图
    keyframe_pose - tensor (4,4) - [R|t] - [R3 t3; 0 0 0 1] kf的内参
    keyframe_intrinsics - tensor - K - kf camera intrinsics (4,4) kf的外参
    frames - list - 是src frame的图片(3,256,512)
    poses - list - 是src frame的相机pose [R|t] (4,4)
    intrinsics - list - 是src frame的内参矩阵 K (4,4)
    sequence - tensor - 属于哪个sequence e.g. seq07 - 7
    image_id - tensor - 这个seq里的哪张图片 - 169
}
depth - tensor(1,256,512)
'''
batch = map_fn(batch, unsqueezer)
depth = map_fn(depth, unsqueezer)

batch = to(batch, device)
########## eval ############
batch["target"] = depth

print("Starting inference...")
s = time.time()
with torch.no_grad():
    data = monorec_model(batch)

prediction = data["result"][0, 0].cpu()
mask = data["cv_mask"][0, 0].cpu()
depth = depth[0, 0].cpu()

############################### eval 
metric,b = eval_metrics(batch)
############################### eval 
e = time.time()
print(f"Inference took {e - s}s")
# print(depth.max())
# print(prediction.max())
print("metrics: ",metric)

plt.imsave("depth.png", prediction.detach().squeeze(), cmap = cm.gray)
plt.imsave("mask.png", mask.detach().squeeze(), cmap = cm.gray)
plt.imsave("kf.png", batch["keyframe"][0].permute(1, 2, 0).cpu().numpy() + 0.5, cmap = cm.gray)
plt.imsave("gt.png", depth.detach().numpy(), cmap = cm.gray)

# plt.title(f"MonoRec (took {e - s}s)")
# plt.imshow(prediction.detach().squeeze(), vmin=1 / 80, vmax=1 / 5)
# plt.show()
# plt.imshow(mask.detach().squeeze())
# plt.show()
# plt.imshow(batch["keyframe"][0].permute(1, 2, 0).cpu() + .5)
# plt.show()
