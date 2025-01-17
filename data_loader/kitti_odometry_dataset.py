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


class KittiOdometryDataset(Dataset):

    def __init__(self, dataset_dir, frame_count=2, sequences=None, depth_folder="image_depth",
                 target_image_size=(256, 512), max_length=None, dilation=1, offset_d=0, use_color=True, use_dso_poses=False,
                 use_color_augmentation=False, lidar_depth=False, dso_depth=True, annotated_lidar=True, return_stereo=False, return_mvobj_mask=False, use_index_mask=()):
        """
        Dataset implementation for KITTI Odometry.
        - dataset_dir: Top level folder for KITTI Odometry (should contain folders sequences, poses, poses_dvso (if available)
        - frame_count: Number of frames used per sample (excluding the keyframe). By default, the keyframe is in the middle of those frames. (Default=2)
        - sequences: Which sequences to use. Should be tuple of strings, e.g. ("00", "01", ...)
        - depth_folder: The folder within the sequence folder that contains the depth information (e.g. sequences/00/{depth_folder})
        - target_image_size: Desired image size (correct processing of depths is only guaranteed for default value). (Default=(256, 512))
        - max_length: Maximum length per sequence. Useful for splitting up sequences and testing. (Default=None)
        - dilation: Spacing between the frames (Default 1)
        - offset_d: Index offset for frames (offset_d=0 means keyframe is centered). (Default=0)
        - use_color: Use color (camera 2) or greyscale (camera 0) images (default=True)
        - use_dso_poses: Use poses provided by d(v)so instead of KITTI poses. Requires poses_dvso folder. (Default=True)
        - use_color_augmentation: Use color jitter augmentation. The same transformation is applied to all frames in a sample. (Default=False)
        - lidar_depth: Use depth information from (annotated) velodyne data. (Default=False)
        - dso_depth: Use depth information from d(v)so. (Default=True)
        - annotated_lidar: If lidar_depth=True, then this determines whether to use annotated or non-annotated depth maps. (Default=True)
        - return_stereo: Return additional stereo frame. Only used during training. (Default=False)
        - return_mvobj_mask: Return additional moving object mask. Only used during training. If return_mvobj_mask=2, then the mask is returned as target instead of the depthmap. (Default=False)
        - use_index_mask: Use the listed index masks (if a sample is listed in one of the masks, it is not used). (Default=())
                            我也不知道这个是干啥的，读不懂，但是好像问题不大。好像是使用mask中没有的samples。为了sample中有moving objects?
        """
        # 1. 准备工作
        self.dataset_dir = Path(dataset_dir)   # 下边有poses, sequences, poses_dvso等文件夹
        self.frame_count = frame_count         # 周围的src view image个数
        self.sequences = sequences             # 文件名list: ["07"]
        self.depth_folder = depth_folder       # 标注的lidar depth map数据
        self.lidar_depth = lidar_depth         # 使用lidar_depth数据
        self.annotated_lidar = annotated_lidar  # 使用annotated_lidar_depth数据。即从lidar_depth处理过的png文件
        self.dso_depth = dso_depth             # 使用DVSO生成的depth map
        self.target_image_size = target_image_size   # 最后要给出的image size
        self.use_index_mask = use_index_mask         # 是否滤掉没有moving object的samples
        self.offset_d = offset_d                     # index的offset
        ## 如果没有设置sequence，则使用00～10
        if self.sequences is None:    
            self.sequences = [f"{i:02d}" for i in range(11)] # list - [00~10]
        self._datasets = [pykitti.odometry(dataset_dir, sequence) for sequence in self.sequences]
        '''
        list of <class 'pykitti.odometry.odometry'>
        每个元素都是 dataset_dir/sequence/00～10 里造出来的class, 可以取各种信息。一个seq一个class
        '''
        self._offset = (frame_count // 2) * dilation    # index的前后range,例如 使用 kf-offset ~ kf+offset index的数据
        extra_frames = frame_count * dilation     # 不是逐帧计算的，所以有额外的frame张数
        if self.annotated_lidar and self.lidar_depth:  # 最多使用10张src view的图
            extra_frames = max(extra_frames, 10)
            self._offset = max(self._offset, 5)
        ## 获取dataset_sizes. 即dataset中有多少张图片，len(cam2_files)
        self._dataset_sizes = [
            len((dataset.cam0_files if not use_color else dataset.cam2_files)) - (extra_frames if self.use_index_mask is None else 0) for dataset in
            self._datasets]
        ## 如果使用index_mask，重新确定dataset_size
        if self.use_index_mask is not None: 
            index_masks = []
            for sequence_length, sequence in zip(self._dataset_sizes, self.sequences):
                index_mask = {i:True for i in range(sequence_length)}
                for index_mask_name in self.use_index_mask:
                    with open(self.dataset_dir / "sequences" / sequence / (index_mask_name + ".json")) as f:
                        m = json.load(f)
                        for k in list(index_mask.keys()):
                            if not str(k) in m or not m[str(k)]:
                                del index_mask[k]
                index_masks.append(index_mask)
            self._indices = [
                list(sorted([int(k) for k in sorted(index_mask.keys()) if index_mask[k] and int(k) >= self._offset and int(k) < dataset_size + self._offset - extra_frames]))
                for index_mask, dataset_size in zip(index_masks, self._dataset_sizes)
            ]
            self._dataset_sizes = [len(indices) for indices in self._indices]
        ## dataset中最多1000张图
        if max_length is not None:  
            self._dataset_sizes = [min(s, max_length) for s in self._dataset_sizes] # 每个sequence文件中有多少个data
        
        ## 获取总length
        self.length = sum(self._dataset_sizes)

        # 2. 获取相机矩阵K
        intrinsics_box = [self.compute_target_intrinsics(dataset, target_image_size, use_color) 
                        for dataset in self._datasets]  # 每个dataset都有一个新K。原因是每录一段都是重新矫正的，所以相同设备也不一样
        self._crop_boxes = [b for _, b in intrinsics_box] # 每段seq都一样。取原图的中心位置。代表的是4个点的坐标 (w左,h上,w右,h下)
        # 3. 获取depth参数
        if self.dso_depth: # 每段sequence的H, W, 原始K的fx
            self.dso_depth_parameters = [self.get_dso_depth_parameters(dataset) for dataset in self._datasets]
        elif not self.lidar_depth:
            self._depth_crop_boxes = [  # depth也是裁剪成对应的区域
                self.compute_depth_crop(self.dataset_dir / "sequences" / s / depth_folder) for s in
                self.sequences]
        # 4. 构建4*4的K
        self._intrinsics = [format_intrinsics(i, self.target_image_size) for i, _ in intrinsics_box] # i - (f_x, f_y, c_x, c_y)
        # 缓冲，咱也不知道是啥
        self.dilation = dilation 
        self.use_color = use_color
        self.use_dso_poses = use_dso_poses  # 是否使用dso_poses
        self.use_color_augmentation = use_color_augmentation   # 是否使用color aug
        if self.use_dso_poses:
            for dataset in self._datasets:
                dataset.pose_path = self.dataset_dir / "poses_dvso"
                dataset._load_poses()
        if self.use_color_augmentation:
            # 使用color aug
            self.color_transform = ColorJitterMulti(brightness=.2, contrast=.2, saturation=.2, hue=.1)
        self.return_stereo = return_stereo   # 是否返回
        if self.return_stereo:
            self._stereo_transform = []
            for d in self._datasets:
                st = torch.eye(4, dtype=torch.float32)  # 4*4 I
                st[0, 3] = d.calib.b_rgb if self.use_color else d.calib.b_gray   # blue赋值给第一行
                self._stereo_transform.append(st)

        self.return_mvobj_mask = return_mvobj_mask   # 是否返回动态物体的mask

    def get_dataset_index(self, index: int):
        # 返回sequence index，和在这个seq中的index
        for dataset_index, dataset_size in enumerate(self._dataset_sizes):
            if index >= dataset_size:  # index out of range
                index = index - dataset_size
            else:   # index在范围内
                return dataset_index, index # 返回sequence index，和在这个seq中的index
        return None, None

    def preprocess_image(self, img: Image.Image, crop_box=None):
        '''
        返回target size的image
        '''
        if crop_box:
            # 1.裁剪
            img = img.crop(crop_box)
        if self.target_image_size:
            # 2.scale
            img = img.resize((self.target_image_size[1], self.target_image_size[0]), resample=Image.BILINEAR)
        if self.use_color_augmentation:
            # 3.增强
            img = self.color_transform(img)
        # 4. 化成tensor
        image_tensor = torch.tensor(np.array(img).astype(np.float32))
        # 5. 范围scale
        image_tensor = image_tensor / 255 - .5
        if not self.use_color:
            # gray 图
            image_tensor = torch.stack((image_tensor, image_tensor, image_tensor))   # (H,W)
        else:
            # RGB图
            image_tensor = image_tensor.permute(2, 0, 1)      # (3,H,W)
        del img
        return image_tensor

    def preprocess_depth(self, depth: np.ndarray, crop_box=None):
        '''
        返回target size的depth
        '''
        if crop_box:
            if crop_box[1] >= 0 and crop_box[3] <= depth.shape[0]:
                depth = depth[int(crop_box[1]):int(crop_box[3]), :]
            else:
                depth_ = np.ones((crop_box[3] - crop_box[1], depth.shape[1]))
                depth_[-crop_box[1]:-crop_box[1]+depth.shape[0], :] = depth
                depth = depth_
            if crop_box[0] >= 0 and crop_box[2] <= depth.shape[1]:
                depth = depth[:, int(crop_box[0]):int(crop_box[2])]
            else:
                depth_ = np.ones((depth.shape[0], crop_box[2] - crop_box[0]))
                depth_[:, -crop_box[0]:-crop_box[0]+depth.shape[1]] = depth
                depth = depth_
        if self.target_image_size:
            depth = resize(depth, self.target_image_size, order=0)
        return torch.tensor(1 / depth)

    def preprocess_depth_dso(self, depth: Image.Image, dso_depth_parameters, crop_box=None):
        '''
        返回target size的depth
        '''
        h, w, f_x = dso_depth_parameters
        depth = np.array(depth, dtype=np.float)
        indices = np.array(np.nonzero(depth), dtype=np.float)
        indices[0] = np.clip(indices[0] / depth.shape[0] * h, 0, h-1)
        indices[1] = np.clip(indices[1] / depth.shape[1] * w, 0, w-1)

        depth = depth[depth > 0]
        depth = (w * depth / (0.54 * f_x * 65535))

        data = np.concatenate([indices, np.expand_dims(depth, axis=0)], axis=0)

        if crop_box:
            data = data[:, (crop_box[1] <= data[0, :]) & (data[0, :] < crop_box[3]) & (crop_box[0] <= data[1, :]) & (data[1, :] < crop_box[2])]
            data[0, :] -= crop_box[1]
            data[1, :] -= crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            crop_width = crop_box[2] - crop_box[0]
        else:
            crop_height = h
            crop_width = w

        data[0] = np.clip(data[0] / crop_height * self.target_image_size[0], 0, self.target_image_size[0]-1)
        data[1] = np.clip(data[1] / crop_width * self.target_image_size[1], 0, self.target_image_size[1]-1)

        depth = np.zeros(self.target_image_size)
        depth[np.around(data[0]).astype(np.int), np.around(data[1]).astype(np.int)] = data[2]

        return torch.tensor(depth, dtype=torch.float32)

    def preprocess_depth_annotated_lidar(self, depth: Image.Image, crop_box=None):
        '''
        给depth map处理成对应的size
        Input:
            depth: Image (from png) - (H,W)
            crop_box: 裁剪的image的坐标
        Output:
            depth - target size 的depth map
        '''
        depth = np.array(depth, dtype=np.float)
        h, w = depth.shape
        indices = np.array(np.nonzero(depth), dtype=np.float)   # (2,num) - 2是(x,y), num是depth ≠ 0的pixel个数。

        depth = depth[depth > 0]   # 选取depth>0的depth value
        depth = 256.0 / depth

        data = np.concatenate([indices, np.expand_dims(depth, axis=0)], axis=0)

        if crop_box:
            data = data[:, (crop_box[1] <= data[0, :]) & (data[0, :] < crop_box[3]) & (crop_box[0] <= data[1, :]) & (
                        data[1, :] < crop_box[2])]
            data[0, :] -= crop_box[1]
            data[1, :] -= crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            crop_width = crop_box[2] - crop_box[0]
        else:
            crop_height = h
            crop_width = w

        data[0] = np.clip(data[0] / crop_height * self.target_image_size[0], 0, self.target_image_size[0] - 1)
        data[1] = np.clip(data[1] / crop_width * self.target_image_size[1], 0, self.target_image_size[1] - 1)

        # 生成最后的depth map
        depth = np.zeros(self.target_image_size)
        depth[np.around(data[0]).astype(np.int), np.around(data[1]).astype(np.int)] = data[2]  # 获取x,y的depth赋值到depth(x,y)

        return torch.tensor(depth, dtype=torch.float32)

    def __getitem__(self, index: int):
        # 1. 获取image的index
        dataset_index, index = self.get_dataset_index(index)  # 返回sequence index，和在这个seq中的index
        if dataset_index is None:
            raise IndexError()
        
        ## 判断是否使用index mask - 只考虑包含mvobj的samples
        if self.use_index_mask is not None:  # 如果使用了index mask
            index = self._indices[dataset_index][index] - self._offset
        
        ## 获取seq的folder和下属的GT depth folder
        sequence_folder = self.dataset_dir / "sequences" / self.sequences[dataset_index]  # 获取seq文件夹
        depth_folder = sequence_folder / self.depth_folder          # 获取保存GT的文件夹 - {kitti}/seq/image_depth_annonated
        
        ## 判断是否color aug
        if self.use_color_augmentation:
            # 使用color aug
            self.color_transform.fix_transform()

        # 2. 这个获取了KITTI的dataset （有很多信息），这个image所在的seq
        dataset = self._datasets[dataset_index]                # '''获取dataset <class 'pykitti.odometry.odometry'>'''
        # 3. 获取相机内参K (preprocess过了)
        keyframe_intrinsics = self._intrinsics[dataset_index]  # 获取seqid的相机内参K

        # 4. 获取GT for different training
        if not (self.lidar_depth or self.dso_depth):
            # 如果既没有使用lidar，也没有使用dso的depth，就直接使用现成的npy (在image_depth_annotated).
            # 这其实就是在把return的GT换成mask
            # 但其实这没有运行过，author把mask的返回在最后边加了一段代码执行。
            # 所以这个if 删了也没啥影响
            keyframe_depth = self.preprocess_depth(np.load(depth_folder / f"{(index + self._offset):06d}.npy"), self._depth_crop_boxes[dataset_index]).type(torch.float32).unsqueeze(0)
        else:
            # 一般lidar_depth和dso_depth只会设置一个True
            if self.lidar_depth:
                # 使用lidar depth, eval使用的是这个！！！！！！
                if not self.annotated_lidar: # 如果不使用annotated,但是是lidar数据，就转化成depth形式 depth(1~400)。应当是target sized image
                    lidar_depth = 1 / torch.tensor(sparse.load_npz(depth_folder / f"{(index + self._offset):06d}.npz").todense()).type(torch.float32).unsqueeze(0)
                    lidar_depth[torch.isinf(lidar_depth)] = 0
                    keyframe_depth = lidar_depth
                else: # 使用lidar数据，并且annotated过了。 eval使用的是这个！！！！！！
                    # lidar_depth=True; annotated=True
                    # 预处理 - 转化成target sized depth map
                    keyframe_depth = self.preprocess_depth_annotated_lidar(Image.open(depth_folder / f"{(index + self._offset):06d}.png"), self._crop_boxes[dataset_index]).unsqueeze(0)
            else:
                # 没有使用lidar depth. 设置为全0 depth矩阵 - target sized depth map
                keyframe_depth = torch.zeros(1, self.target_image_size[0], self.target_image_size[1], dtype=torch.float32)

            if self.dso_depth:
                # dso_depth的优先级 > lidar_depth，如果使用了dso，则优先dso
                # dso_depth=True；使用了DSVO系统生成的image_sparse_depth
                # 转成
                dso_depth = self.preprocess_depth_dso(Image.open(depth_folder / f"{(index + self._offset):06d}.png"), self.dso_depth_parameters[dataset_index], self._crop_boxes[dataset_index]).unsqueeze(0)
                mask = dso_depth == 0
                dso_depth[mask] = keyframe_depth[mask]
                keyframe_depth = dso_depth

        # 5. 获取key frame RGB，使用dataset[index+offset]的get_cam2
        keyframe = self.preprocess_image(
            (dataset.get_cam0 if not self.use_color else dataset.get_cam2)(index + self._offset),
            self._crop_boxes[dataset_index])

        # 6. 获取kf的pose。(第index+一开始不要的几帧)
        keyframe_pose = torch.tensor(dataset.poses[index + self._offset], dtype=torch.float32)  # cam0

        # 7. 获取src frames - list
        ## 取frame_count张
        frames = [self.preprocess_image((dataset.get_cam0 if not self.use_color else dataset.get_cam2)(index + self._offset + i + self.offset_d),
                                        self._crop_boxes[dataset_index]) for i in
                  range(-(self.frame_count // 2) * self.dilation, ((self.frame_count + 1) // 2) * self.dilation + 1, self.dilation) if i != 0]
        # 8. 获取src frames的内参K - list - 其实元素都一样，只是重复了frame_count个
        intrinsics = [self._intrinsics[dataset_index] for _ in range(self.frame_count)]
        # 9. 获取src frame的外参poses => cam0的GT pose
        poses = [torch.tensor(dataset.poses[index + self._offset + i + self.offset_d], dtype=torch.float32) for i in
                 range(-(self.frame_count // 2) * self.dilation, ((self.frame_count + 1) // 2) * self.dilation + 1, self.dilation) if i != 0]

        data = {
            "keyframe": keyframe,
            "keyframe_pose": keyframe_pose,
            "keyframe_intrinsics": keyframe_intrinsics,
            "frames": frames,
            "poses": poses,
            "intrinsics": intrinsics,
            "sequence": torch.tensor([int(self.sequences[dataset_index])], dtype=torch.int32),
            "image_id": torch.tensor([int(index + self._offset)], dtype=torch.int32)
        }

        if self.return_stereo: # kf的另一个view的frame - static stereo frame
            stereoframe = self.preprocess_image(
                (dataset.get_cam1 if not self.use_color else dataset.get_cam3)(index + self._offset),
                self._crop_boxes[dataset_index])
            stereoframe_pose = torch.tensor(dataset.poses[index + self._offset], dtype=torch.float32) @ self._stereo_transform[dataset_index]
            data["stereoframe"] = stereoframe
            data["stereoframe_pose"] = stereoframe_pose
            data["stereoframe_intrinsics"] = keyframe_intrinsics

        ############################# 在这返回data和mvobj_mask，应当是为了train mask module设置的 ####################################
        if self.return_mvobj_mask > 0:
            # 在这返回data和mvobj_mask，应当是为了train mask module设置的
            mask = torch.tensor(np.load(sequence_folder / "mvobj_mask" / f"{index + self._offset:06d}.npy"), dtype=torch.float32).unsqueeze(0)
            data["mvobj_mask"] = mask
            if self.return_mvobj_mask == 2:
                return data, mask
        #################################################################
        '''
        data - {
            keyframe - tensor (3,256,512) - RGB
            keyframe_pose - tensor (4,4) - [R|t] - [R3 t3; 0 0 0 1] kf的内参
            keyframe_intrinsics - tensor (4,4) - K - kf camera intrinsics kf的外参
            frames - list - 是src frame的图片(3,256,512) 多张RGB
            poses - list - 是src frame的相机pose [R|t] (4,4)
            intrinsics - list - 是src frame的内参矩阵 K (4,4)
            sequence - tensor - 属于哪个sequence e.g. seq07 - 7
            image_id - tensor - 这个seq里的哪张图片 - 169
            ############ [optional] 补充stereo frame ###############
            stereoframe - tensor (3,256,512) - RGB - 对应kf time的另一个view的image
            stereoframe_pose - tensor (4,4) - [R|t] - [R3 t3; 0 0 0 1] kf的内参
            stereoframe_intrinsics - tensor (4,4) - K - kf camera intrinsics kf的外参
            ############################################
        }
        keyframe_depth - tensor (1,256,512)
        '''
        return data, keyframe_depth

    def __len__(self) -> int:
        return self.length

    def compute_depth_crop(self, depth_folder):
        # This function is only used for dense gt depth maps.
        example_dm = np.load(depth_folder / "000000.npy")  # 获取给出的depth map
        ry = example_dm.shape[0] / self.target_image_size[0]  # 放缩系数
        rx = example_dm.shape[1] / self.target_image_size[1]
        if ry < 1 or rx < 1:
            if ry >= rx:
                o_w = example_dm.shape[1]
                w = int(np.ceil(ry * self.target_image_size[1]))
                h = example_dm.shape[0]
                return ((o_w - w) // 2, 0, (o_w - w) // 2 + w, h)
            else:
                o_h = example_dm.shape[0]
                h = int(np.ceil(rx * self.target_image_size[0]))
                w = example_dm.shape[1]
                return (0, (o_h - h) // 2, w, (o_h - h) // 2 + h)
        if ry >= rx:
            o_h = example_dm.shape[0]
            h = rx * self.target_image_size[0]
            w = example_dm.shape[1]
            return (0, (o_h - h) // 2, w, (o_h - h) // 2 + h)
        else:
            o_w = example_dm.shape[1]
            w = ry * self.target_image_size[1]
            h = example_dm.shape[0]
            return ((o_w - w) // 2, 0, (o_w - w) // 2 + w, h)

    def compute_target_intrinsics(self, dataset, target_image_size, use_color):
        '''
        每个
        Input: 
            dataset - <class 'pykitti.odometry.odometry'>
            target_image_size - (256,512)
            use_color - true: gray; false: RGB
        Output:
            box - 取原图的中心位置。代表的是4个点的坐标 (w左,h上,w右,h下)
            intrinsic - 新的K矩阵的4个系数(f_x, f_y, c_x, c_y)
        '''
        # Because of cropping and resizing of the frames, we need to recompute the intrinsics
        P_cam = dataset.calib.P_rect_00 if not use_color else dataset.calib.P_rect_20       # 1 获取 image02 的3*4 K矩阵
        orig_size = tuple(reversed((dataset.cam0 if not use_color else dataset.cam2).__next__().size)) # 2 初始的image02 的原始size
        # 3 我们已经有target image的size

        # 4 通过两个size进行计算，得出intrinsics和box
        r_orig = orig_size[0] / orig_size[1] # 原图 H/W
        r_target = target_image_size[0] / target_image_size[1] # target H/W

        if r_orig >= r_target: # 原图比target 竖直方向更长
            new_height = r_target * orig_size[1]   # new height for 原图
            box = (0, (orig_size[0] - new_height) // 2, orig_size[1], orig_size[0] - (orig_size[0] - new_height) // 2) # 0，新h起点，W，新h终点

            c_x = P_cam[0, 2] / orig_size[1]    # 新K中的tx => 原始的tx/新图的总W
            c_y = (P_cam[1, 2] - (orig_size[0] - new_height) / 2) / new_height  # 新K中的tx => 原始的tx/新图的总W

            rescale = orig_size[1] / target_image_size[1]  # 放缩系数

        else:  # 原图比target 更宽
            new_width = orig_size[0] / r_target
            box = ((orig_size[1] - new_width) // 2, 0, orig_size[1] - (orig_size[1] - new_width) // 2, orig_size[0])  # 新w起点，0，新w终点,h

            c_x = (P_cam[0, 2] - (orig_size[1] - new_width) / 2) / new_width
            c_y = P_cam[1, 2] / orig_size[0]

            rescale = orig_size[0] / target_image_size[0]

        f_x = P_cam[0, 0] / target_image_size[1] / rescale  # fx * w/target_h = 初始的fx/w
        f_y = P_cam[1, 1] / target_image_size[0] / rescale

        intrinsics = (f_x, f_y, c_x, c_y)

        return intrinsics, box

    def get_dso_depth_parameters(self, dataset):
        # Info required to process d(v)so depths
        P_cam =  dataset.calib.P_rect_20
        orig_size = tuple(reversed(dataset.cam2.__next__().size))
        return orig_size[0], orig_size[1], P_cam[0, 0]  # H, W, 原始fx

    def get_index(self, sequence, index):
        for i in range(len(self.sequences)):
            if int(self.sequences[i]) != sequence:
                index += self._dataset_sizes[i]
            else:
                break
        return index


def format_intrinsics(intrinsics, target_image_size):
    '''
    input
        intrinsics - (f_x, f_y, c_x, c_y)
        target_image_size - (256,512)
    output
        intrinsics_mat - 4*4的新K矩阵
    '''
    intrinsics_mat = torch.zeros(4, 4)
    intrinsics_mat[0, 0] = intrinsics[0] * target_image_size[1]
    intrinsics_mat[1, 1] = intrinsics[1] * target_image_size[0]
    intrinsics_mat[0, 2] = intrinsics[2] * target_image_size[1]
    intrinsics_mat[1, 2] = intrinsics[3] * target_image_size[0]
    intrinsics_mat[2, 2] = 1
    intrinsics_mat[3, 3] = 1
    return intrinsics_mat


class ColorJitterMulti(torchvision.transforms.ColorJitter):
    # 就是个color aug的function
    def fix_transform(self):
        self.transform = self.get_params(self.brightness, self.contrast,
                                         self.saturation, self.hue)

    def __call__(self, x):
        return map_fn(x, self.transform)
