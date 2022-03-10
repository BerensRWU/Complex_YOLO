import numpy as np
import cv2
import torch

import utils.utils as utils

import utils.astyx_bev_utils as bev_utils
import utils.config as cnf

def visualize_func(bev_maps, targets, img_detections, sample_id, estimate_bb):
        bev_map = bev_maps.numpy() # torch to numpy

        # Generation of a depth map that can be visualized by cv2
        # Because cv2 saves BGR instead of RGB
        RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        RGB_Map[:, :, 2] = bev_map[0, :, :]  # height -> r_map 
        RGB_Map[:, :, 1] = bev_map[1, :, :]  # density -> g_map
        RGB_Map[:, :, 0] = bev_map[2, :, :]  # intensity/velocity -> b_map
        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)
        
        # loop over all targets
        for _,cls,x,y,w,l,im,re in targets:
            # get the yaw from the euler angle value
            yaw = np.arctan2(im,re)
            # Draw green groundtruth
            drawRotatedBox(RGB_Map, x, y, w, l, yaw, [0, 255, 0])

        # same visualization as for targets
        if estimate_bb:    
            for detections in img_detections:
                if detections is None:
                    continue
                for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw red predicted box
                    drawRotatedBox(RGB_Map, x, y, w, l, yaw, [0, 0, 255])
        
        cv2.imwrite("output/%06d.png" % sample_id, RGB_Map) # note cv2 RGB->BGR
        
        
def drawRotatedBox(img,x,y,w,l,yaw,color):
    # get the corners of the rotated bounding box
    bev_corners = bev_utils.get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    # draw box
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    # draw line that shows where the front of the car is
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)
