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

from utils import map_fn

dataset_dir = "../data/dataset"
sequences = ["00","04"]
datasets = [pykitti.odometry(dataset_dir, sequence) for sequence in sequences]

print(type(datasets[0]))