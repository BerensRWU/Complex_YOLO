from __future__ import division
import os
import glob
import numpy as np
import cv2
import torch.utils.data as torch_data
import utils.astyx_utils as astyx_utils

class AstyxDataset(torch_data.Dataset):

    def __init__(self, root_dir, split='valid'):
        self.split = split

        self.imageset_dir = os.path.join(root_dir, 'dataset_astyx_hires2019')
        
        self.lidar_path = os.path.join(self.imageset_dir, "lidar_vlp16")
        self.radar_path = os.path.join(self.imageset_dir, "radar_6455")
        self.image_path = os.path.join(self.imageset_dir, "camera_front")
        self.calib_path = os.path.join(self.imageset_dir, "calibration")
        self.label_path = os.path.join(self.imageset_dir, "groundtruth_obj3d")

        split_dir = os.path.join(root_dir, "split", split+'.txt')
        self.image_idx_list = [np.int(x.strip()) for x in open(split_dir).readlines()]

        self.num_samples = self.image_idx_list.__len__()

    def get_image(self, idx):
        img_file = os.path.join(self.image_path, '%06d.jpg' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file) # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_path, '%06d.jpg' % idx)
        assert  os.path.exists(img_file)
        img = cv2.imread(img_file)
        width, height, channel = img.shape
        return width, height, channel

    def get_radar(self, idx):
        radar_file = os.path.join(self.radar_path, '%06d.txt' % idx)
        assert os.path.exists(radar_file)
        point_cloud = np.loadtxt(radar_file, dtype=np.float32, skiprows = 2)
        point_cloud = point_cloud[:,0:4]
        return point_cloud
        
    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_path, '%06d.txt' % idx)
        assert os.path.exists(lidar_file)
        point_cloud = np.loadtxt(lidar_file, dtype=np.float32, skiprows = 1)
        point_cloud = point_cloud[:,0:4]
        return point_cloud

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, '%06d.json' % idx)
        assert os.path.exists(calib_file)
        return astyx_utils.calib_astyx(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_path, '%06d.json' % idx)
        assert os.path.exists(label_file)
        return astyx_utils.read_label(label_file)

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented
