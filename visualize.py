import numpy as np
import os
import argparse
import cv2
import torch

import utils.utils as utils
from models import *
import torch.utils.data as torch_data

import utils.astyx_bev_utils as bev_utils
from utils.astyx_yolo_dataset import AstyxYOLODataset
import utils.config as cnf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="network/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--split", type=str, default="valid", help="text file having image lists in dataset")
    parser.add_argument("--radar", default=False, action="store_true" , help="Use Radar Data")
    parser.add_argument("--estimate_bb", default=False, action="store_true", help="Whether to estimate Bounding Boxes")
    opt = parser.parse_args()
    print(opt)
    
    if not os.path.exists("output"):
        os.makedirs("output")
        
    if opt.estimate_bb:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        weights_path = os.path.join(opt.weights_path, "weights_RADAR.pth" if opt.radar else "weights_LIDAR.pth")
        # Set up model
        model = Darknet(opt.model_def, img_size=cnf.BEV_WIDTH).to(device)
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path, map_location = device))
        # Eval mode
        model.eval()
    
    dataset = AstyxYOLODataset(cnf.root_dir, split=opt.split, mode="EVAL", radar=opt.radar)
    data_loader = torch_data.DataLoader(dataset, batch_size=1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for index, (sample_id, bev_maps, targets) in enumerate(data_loader):
        if opt.estimate_bb:
            # Configure bev image
            input_imgs = Variable(bev_maps.type(Tensor))
            # Get detections 
            with torch.no_grad():
                detections = model(input_imgs)
                detections = utils.non_max_suppression_rotated_bbox(detections, opt.conf_thres, opt.nms_thres) 
            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

        bev_maps = torch.squeeze(bev_maps).numpy()

        RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        # Because cv2 saves BGR instead of RGB
        RGB_Map[:, :, 2] = bev_maps[0, :, :]  # height -> r_map 
        RGB_Map[:, :, 1] = bev_maps[1, :, :]  # density -> g_map
        RGB_Map[:, :, 0] = bev_maps[2, :, :]  # intensity/velocity -> b_map
        
        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)
        
        targets = targets[0]
        targets[:, 2:] *= cnf.BEV_WIDTH
        for _,cls,x,y,w,l,im,re in targets:
            yaw = np.arctan2(im,re)
            bev_utils.drawRotatedBox(RGB_Map, x, y, w, l, yaw, [0, 255, 0])
            
        if opt.estimate_bb:    
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = utils.rescale_boxes(detections, cnf.BEV_WIDTH, RGB_Map.shape[:2])
                
                for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    bev_utils.drawRotatedBox(RGB_Map, x, y, w, l, yaw, [0, 0, 255])
        
        cv2.imwrite("output/%06d.png" % sample_id, RGB_Map) # note cv2 RGB->BGR
