from ast import Try
import json
from pathlib import Path
from re import A
from socket import SO_RCVTIMEO
from cv2 import randShuffle
import random
import os
import os.path as osp
# pip install nuscenes-devkit &> /dev/null
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from torch.nn.functional import max_pool2d


import numpy as np
import pykitti
import torch
import torchvision
from PIL import Image
from scipy import sparse
from skimage.transform import resize
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes

# from utils import map_fn
def map_fn(batch, fn):
    # 把dict或者list或者tensor本身，使用fn function操作，进行返回。
    # 最终return的类型和开始保持一致
    if isinstance(batch, dict):
        # 传入的是batch
        for k in batch.keys():
            batch[k] = map_fn(batch[k], fn)
        return batch
    elif isinstance(batch, list):
        # 传入的是list
        return [map_fn(e, fn) for e in batch]
    else:
        return fn(batch)


class NuscenesDataset(Dataset):

    def __init__(self, version: str = 'v1.0-mini',dataset_dir: str = '../../data/nuscenes-mini/v1.0-mini', 
                pointsensor_channel: str = 'LIDAR_TOP', camera_channel: str = 'CAM_FRONT',
                frame_count=4,target_image_size=(256, 512), dilation=1, offset_d=0,
                 use_color_augmentation=False, return_mvobj_mask=False):
        """
        Dataset implementation for KITTI Odometry.
        - dataset_dir: Top level folder for KITTI Odometry (should contain folders sequences, poses, poses_dvso (if available)
        - frame_count: Number of frames used per sample (excluding the keyframe). By default, the keyframe is in the middle of those frames. (Default=2)
                        一般是偶数, 因为要确保key frame在中间
        - target_image_size: Desired image size (correct processing of depths is only guaranteed for default value). (Default=(256, 512))
        - dilation: Spacing between the frames (Default 1)
        - offset_d: Index offset for frames (offset_d=0 means keyframe is centered). (Default=0)
        - use_color_augmentation: Use color jitter augmentation. The same transformation is applied to all frames in a sample. (Default=False)
        - return_mvobj_mask: Return additional moving object mask. Only used during training. If return_mvobj_mask=2, then the mask is returned as target instead of the depthmap. (Default=False)
        """
        # 1. 准备工作
        self.dataset_dir = Path(dataset_dir)   # 下边有poses, sequences, poses_dvso等文件夹
        self.frame_count = frame_count         # 周围的src view image个数
        # self.depth_folder = depth_folder       # 标注的lidar depth map数据
        self.target_image_size = target_image_size   # 最后要给出的image size
        self.offset_d = offset_d                     # index的offset
        # 创建nuscene dataset
        self.nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=True)
        '''
        nusc.sample 是 所有的keyframe metadata - 包含token
        '''
        # sensor 来源
        self.pointsensor_channel = pointsensor_channel   # LIDAR_TOP
        self.camera_channel = camera_channel             # CAM_FRONT
        

        # select参数
        self._offset = (frame_count // 2) * dilation    # index的前后range,例如 使用 kf-offset ~ kf+offset index的数据
        extra_frames = frame_count * dilation     # 不是逐帧计算的，所以有额外的frame张数
        ## additional set: 最多跨度为前后10张frame 作 src image
        extra_frames = max(extra_frames, 8)
        self._offset = max(self._offset, 4)

        ''' 
        计算dataset的len这里有个小修改:
        以前的KITTIDataset会有扔掉;
        现在一个简单的处理方法是让首尾的frame也count进length,不过是第二帧和倒数第二帧多count了一次
        '''
        
        ## 获取总length
        self.length = len(self.nusc.sample)
        
        ## dilation最大是2
        self.dilation = max(dilation,2)
        self.use_color_augmentation = use_color_augmentation   # 是否使用color aug
        ## 颜色增强
        if self.use_color_augmentation:
            # 使用color aug
            self.color_transform = ColorJitterMulti(brightness=.2, contrast=.2, saturation=.2, hue=.1)

        self.return_mvobj_mask = return_mvobj_mask   # 是否返回动态物体的mask

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
        if self.use_color_augmentation:  # default false
            # 3.增强
            img = self.color_transform(img)
        # 4. 化成tensor
        image_tensor = torch.tensor(np.array(img).astype(np.float32))
        # 5. 范围scale
        image_tensor = image_tensor / 255 - .5

        image_tensor = image_tensor.permute(2, 0, 1)      # (3,H,W)
        del img
        return image_tensor

    def get_key_sample(self,nusc,index,offset):
        '''
        前提: 默认一个scene中至少能取出一个sample
        Return the right key frame for this index (ensure the key frame is centered and there are enough neibor frames)
        Input:
            nusc - nusc dataset
            index - index of sample
            offset - range that neighbor frame reaches
        Return:
            key_sample - the right key frame sample {['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns']}
        '''
        sample = nusc.sample[index]
        scene = nusc.get("scene",sample["scene_token"])
        nbr_samples = scene["nbr_samples"]
        if nbr_samples < (2*offset+1):
            raise FileNotFoundError("Can't generate one sample in this scene because of too large frame range")

        temp_sample = sample
        prev_count = 0
        next_count = 0

        # ensure prev has enough frames
        for i in range(offset):
            if temp_sample["prev"] == "":
                # 触到prev的边界，需要向后移动offset-i个，来确保prev 有 enough frames
                for j in range(offset-i):
                    # 调整到相应位置
                    sample = nusc.get('sample', token=sample['next'])
                # 调整完之后应该跳出for循环
                break
            else:
                # 仍有prev, 再向前探索
                temp_sample = nusc.get('sample',temp_sample['prev'])
                prev_count += 1
            # 如果prev时调整了，就无需再next调整
        if prev_count < 4:
            # 向前搜索已经调整过了，无需向后调整
            return sample

        # ensure next has enough frames
        ## 重新校定temp sample到sample一样的位置
        temp_sample = sample
        for i in range(offset):
            if temp_sample["next"] == "":
                # 触到prev的边界，需要向后移动offset-i个，来确保prev 有 enough frames
                for j in range(offset-i):
                    # 调整到相应位置
                    sample = nusc.get('sample', token=sample['prev'])
                # 调整完之后应该跳出for循环
                break
            else:
                # 仍有next, 再向后探索
                temp_sample = nusc.get('sample',temp_sample['next'])
                next_count += 1
        return sample

    def map_pointcloud_to_image(self,nusc,
                            pointsensor_token: str,
                            camera_token: str,
                            sample_rec,
                            nsweeps=3,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False):
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidar intensity instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """

        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            # Ensure that lidar pointcloud is from a keyframe.
            assert pointsensor['is_key_frame'], \
                'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
            # pc = LidarPointCloud.from_file(pcl_path)
            # 融合一下lidar信息 # 
            chan="LIDAR_TOP"
            ref_chan = "LIDAR_TOP"
            pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(nusc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                                'not %s!' % pointsensor['sensor_modality']
            # Retrieve the color from the intensities.
            # Performs arbitary scaling to achieve more visually pleasing results.
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities

        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring, im

    def get_depth(self, nusc, sample, do_flip=False):
        '''
        return depth map for the original image

        '''
        pointsensor_token = sample['data'][self.pointsensor_channel]
        camsensor_token = sample['data'][self.camera_channel]
        pts, depth, img = self.map_pointcloud_to_image(nusc, pointsensor_token, camsensor_token)
        depth_gt = np.zeros((img.size[0], img.size[1]))
        # 转成坐标形式
        pts_int = np.array(pts, dtype=int)
        depth_gt[pts_int[0,:], pts_int[1,:]] = depth

        return np.transpose(depth_gt, (1,0))

    def format_intrinsics(self,intrinsics, target_image_size):
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
        intrinsics_mat[0, 2] = intrinsics[2] * target_image_size[1]   # cx*w
        intrinsics_mat[1, 2] = intrinsics[3] * target_image_size[0]   # cy*h
        intrinsics_mat[2, 2] = 1
        intrinsics_mat[3, 3] = 1
        return intrinsics_mat

    def preprocess_depth_points(self,pts, depth, orig_size, key_frame_box, target_image_size):
        '''
        把lidar points warp到
        input
            pts - (3,n1) - coordinates of sparse lidar points
            depth - (n1,) - depth value
            orig_size - (900,1600) - 原图size
            key_frame_box - (新w起点,新h起点,新w终点,新h终点)
            target_image_size - (256,512) - target image size
        Return
            new_pts - (3,n2) - n2≤n1
            new_depth - (n2,)
        '''
        n1 = depth.shape[0]

        # get box area
        w_st, h_st, w_end, h_end= key_frame_box

        if w_st == 0:
            # 原图的h更长, w部分局限, 按照w放缩
            # 900,1600 -> 256,512
            rescale = orig_size[1] / target_image_size[1]
        else:
            # 原图的w更长, h部分局限, 按照h放缩
            # 700,1600 -> 256,512
            rescale = orig_size[0] / target_image_size[0]

        new_pts = []
        new_depth = []

        for i in range(n1):
            xi = pts[0,i]
            yi = pts[1,i]
            if xi<=w_st or xi>=w_end or yi<=h_st or yi>=h_end:
                # not in the box area, throw it out
                continue
            else:
                # keep it and warp
                new_xi = (xi-w_st) / rescale
                new_yi = (yi-h_st) / rescale
                new_pts.append(np.array([new_xi,new_yi,1]))
                new_depth.append(depth[i])
        # change list to array
        new_pts = np.array(new_pts).T
        new_depth = np.array(new_depth)

        return new_pts,new_depth

    def get_depth_map(self,new_pts,new_depth,target_image_size):
        '''
        Input
            new_pts - (3,n)
            new_depth - (n,)
            target_image_size - tuple - (256,512)
        Return
            depth_map - tensor - (1,256,512)
        '''
        depth_map = np.zeros(target_image_size)
        for i in range(new_depth.shape[0]):
            h = int(new_pts[1,i])
            w = int(new_pts[0,i])
            depth_map[h,w] = 1/new_depth[i]

        # numpy -> tensor (256,512) and -> (1,256,512)
        depth_map = torch.from_numpy(depth_map).unsqueeze(0)

        return depth_map

    # def get_pose_mat(self, frame_pose_data):
    #     '''
    #     frame_pose_data: Quaternions
    #     e.g. - 
    #         {'token': 'ddbc1befa70f4b49a0824f63a920676b',
    #         'timestamp': 1532402930112460,
    #         'rotation': [0.5859061687572851,
    #         0.00046748170180213036,
    #         0.01238901832158085,
    #         -0.8102840582771244],
    #         'translation': [404.3748127067429, 1161.3378286421344, 0.0]}
    #     return:
    #         tensor - 4*4 matrix [R|t] - [R3 t3; 0 0 0 1]
    #     '''
    #     pose = torch.zeros(4,4)
    #     rotation = frame_pose_data["rotation"]
    #     w,x,y,z = rotation
    #     translation = frame_pose_data["translation"]

    #     # R
    #     pose[0,0] = 1 - 2*(y**2) - 2*(z**2)
    #     pose[0,1] = 2*x*y + 2*w*z
    #     pose[0,2] = 2*x*z - 2*w*y
    #     pose[1,0] = 2*x*y - 2*w*z
    #     pose[1,1] = 1 - 2*(x**2) - 2*(z**2)
    #     pose[1,2] = 2*y*z + 2*w*x
    #     pose[2,0] = 2*x*z + 2*w*y
    #     pose[2,1] = 2*y*z - 2*w*x
    #     pose[2,2] = 1 - 2*(x**2) - 2*(y**2)

    #     # t
    #     pose[0,3] = translation[0]
    #     pose[1,3] = translation[1]
    #     pose[2,3] = translation[2]

    #     pose[3,3] = 1

    #     return pose

    def get_ego_pose(self, frame_pose_data):
        '''
        frame_pose_data: Quaternions
        e.g. - 
            {'token': 'ddbc1befa70f4b49a0824f63a920676b',
            'timestamp': 1532402930112460,
            'rotation': [0.5859061687572851,
            0.00046748170180213036,
            0.01238901832158085,
            -0.8102840582771244],
            'translation': [404.3748127067429, 1161.3378286421344, 0.0]}
        return:
            tensor - 4*4 matrix [R|t] - [R3 t3; 0 0 0 1]
        '''
        pose = torch.zeros(4,4)
        rotation = frame_pose_data["rotation"]
        w,x,y,z = rotation
        translation = frame_pose_data["translation"]

        # 获取rotation matrix from quaternion
        r_mat = Quaternion(rotation).rotation_matrix  

        # R
        # pose[0,0] = 1 - 2*(y**2) - 2*(z**2)
        # pose[0,1] = 2*x*y + 2*w*z
        # pose[0,2] = 2*x*z - 2*w*y
        # pose[1,0] = 2*x*y - 2*w*z
        # pose[1,1] = 1 - 2*(x**2) - 2*(z**2)
        # pose[1,2] = 2*y*z + 2*w*x
        # pose[2,0] = 2*x*z + 2*w*y
        # pose[2,1] = 2*y*z - 2*w*x
        # pose[2,2] = 1 - 2*(x**2) - 2*(y**2)
        pose[:3,:3] = torch.tensor(r_mat)

        # t
        pose[0,3] = translation[0]
        pose[1,3] = translation[1]
        pose[2,3] = translation[2]

        pose[3,3] = 1

        return pose

    def __getitem__(self, index: int):
        # 1. 获取key frame的正确sample
        key_sample = self.get_key_sample(self.nusc,index,self._offset)

        # Here we just grab the front camera and the point sensor.
        key_pointsensor_token = key_sample['data'][self.pointsensor_channel]   # lidar data token
        key_camera_token = key_sample['data'][self.camera_channel]    # cam data token
        
        ## 判断是否color aug
        if self.use_color_augmentation:
            # 使用color aug
            self.color_transform.fix_transform()
        
        # 在这都是原始的大小
        ## 1.获取image (未处理的原图)
        # keyimg_meta = self.nusc.get("sample_data",key_pointsensor_token)
        # keyimg_filename = keyimg_meta["filename"]
        # keyimg_path = osp.join(self.nusc.dataroot,keyimg_filename)
        # keyimg = Image.open(keyimg_path)
        #############################  待完成  ##########################################
        ## 获取depth的lidar points
        ## 在这由于太稀疏了，所以不使用之前的preprocess了 
        pts, depth, keyimg = self.map_pointcloud_to_image(self.nusc, key_pointsensor_token, key_camera_token, key_sample, nsweeps=3)
        '''
        pts - [3,n] - 第一行是height coor, 第二行是width coor, 第三行是1
        depth - [n,] - depth value
        '''

        #######################################################################
        ## 2. 获取原始intrinsic matrix - K
        key_camData = self.nusc.get('sample_data', key_camera_token)
        '''
        token, sample_token, ego_pose_token, calibrated_sensor_token, filename, channel, is_key_frame, prev, next - 这指的就是sweep了
        '''
        key_intrinsic_data = self.nusc.get('calibrated_sensor', key_camData['calibrated_sensor_token'])
        key_intrinsic_K = np.array(key_intrinsic_data["camera_intrinsic"])   # 3*3 K
        
        ## 3. 根据target size，预处理原图，K，depth map
        ### 3.1 获取K新系数，crop box
        orig_size = tuple((key_camData["height"],key_camData["width"]))   # (900,1600)
        new_key_intrinsics_coff, key_frame_box = self.compute_target_intrinsics(key_intrinsic_K, orig_size, self.target_image_size)

        ### 3.2 预处理原图
        keyimg_tensor = self.preprocess_image(keyimg, key_frame_box)    # (3,H,W)

        ### 3.3 预处理K
        key_K = self.format_intrinsics(new_key_intrinsics_coff, self.target_image_size) # 4*4 K

        ### 3.4 预处理depth map
        new_pts, new_depth = self.preprocess_depth_points(pts, depth, orig_size, key_frame_box, self.target_image_size)
        keyframe_depth = self.get_depth_map(new_pts,new_depth,self.target_image_size)

        # 4. 获取keyframe的pose mat
        key_ego_pose_token = key_camData["ego_pose_token"]
        keyframe_pose_data = self.nusc.get("ego_pose", key_ego_pose_token)  # ego pose metadata
        keyframe_pose = self.get_ego_pose(keyframe_pose_data)

        
        
        # 5.获取src的数据
        ### 5.0 准备工作
        src_samples = []  # 按照顺序的src samples
        frames = []
        poses = []
        intrinsics = []

        # 向前向后取出sample
        ## 向前
        prev_sample = key_sample
        for i in range(self.frame_count//2):
            for j in range(self.dilation):
                prev_sample = self.nusc.get("sample",prev_sample["prev"])
            src_samples.insert(0,prev_sample)  # 插在头部
        ## 向后
        next_sample = key_sample
        for i in range(self.frame_count//2):
            for j in range(self.dilation):
                next_sample = self.nusc.get("sample",next_sample["next"])
            src_samples.append(next_sample)  # 插在尾部
        ### 5.1 获取src的frames
        for src_sample in src_samples:
            # 5.1 获取src的frames
            src_camtoken = src_sample['data'][self.camera_channel]
            src_camData = self.nusc.get("sample_data",src_camtoken)
            srcimg_filename = src_camData["filename"]
            srcimg_path = osp.join(self.nusc.dataroot,srcimg_filename)
            srcimg = Image.open(srcimg_path)
            srcimg_tensor = self.preprocess_image(srcimg, key_frame_box)  # box是和key frame共享的
            frames.append(srcimg_tensor)

            # 5.2 由于一个sequence里的K都是一样的，所以我们加入处理过的key_K
            intrinsics.append(key_K)

            # 5.3 加入poses
            src_ego_pose_token = src_camData["ego_pose_token"]
            srcframe_pose_data = self.nusc.get("ego_pose", src_ego_pose_token)
            poses.append(self.get_ego_pose(srcframe_pose_data))

        data = {
            "keyframe": keyimg_tensor,
            "keyframe_pose": keyframe_pose,
            "keyframe_intrinsics": key_K,
            "frames": frames,
            "poses": poses,
            "intrinsics": intrinsics,
            "sequence": torch.tensor([0]),  # default
            "image_id": torch.tensor([index + self._offset]) # default
        }
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

    def compute_target_intrinsics(self, key_intrinsic_K, orig_size, target_image_size):
        '''
        每个
        Input: 
            key_intrinsic_K - ndarray 3*3 原始K矩阵
            orig_size - 初始的image02 的原始size - tuple(900,1600)
            target_image_size - (256,512)
        Output:
            intrinsic - 新的K矩阵的4个系数(f_x, f_y, c_x, c_y)
            box - 取原图的中心位置。代表的是4个点的坐标 (w左,h上,w右,h下)
        '''
        # Because of cropping and resizing of the frames, we need to recompute the intrinsics
        P_cam = key_intrinsic_K   # 获取原始K 3*3
        orig_size = orig_size # 2 初始的image02 的原始size
        # 3 我们已经有target image的size

        # 4 通过两个size进行计算，得出intrinsics和box
        r_orig = orig_size[0] / orig_size[1] # 原图 H/W
        r_target = target_image_size[0] / target_image_size[1] # target H/W

        if r_orig >= r_target: # 原图比target 竖直方向更长
            new_height = r_target * orig_size[1]   # # 原图被scale的区域的height - 800
            box = (0, (orig_size[0] - new_height) // 2, orig_size[1], orig_size[0] - (orig_size[0] - new_height) // 2) # 0，新h起点，W，新h终点

            c_x = P_cam[0, 2] / orig_size[1]    # 新K中的tx => 原始的tx/新图的总W
            c_y = (P_cam[1, 2] - (orig_size[0] - new_height) / 2) / new_height  # 新K中的tx => 原始的tx/新图的总W

            rescale = orig_size[1] / target_image_size[1]  # 放缩系数  1600/512     ～      800/256

        else:   # 原图比target 更宽
            new_width = orig_size[0] / r_target  # 原图被scale的区域的width - 1600
            box = ((orig_size[1] - new_width) // 2, 0, orig_size[1] - (orig_size[1] - new_width) // 2, orig_size[0])  # 新w起点，0，新w终点,h

            c_x = (P_cam[0, 2] - (orig_size[1] - new_width) / 2) / new_width
            c_y = P_cam[1, 2] / orig_size[0]

            rescale = orig_size[0] / target_image_size[0]     # 即800/256

        f_x = P_cam[0, 0] / target_image_size[1] / rescale  # fx * w/target_h = 初始的fx/w
        f_y = P_cam[1, 1] / target_image_size[0] / rescale     # 即fy/800

        intrinsics = (f_x, f_y, c_x, c_y)

        return intrinsics, box

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
