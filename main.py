import numpy as np
import os
import argparse
import cv2
import torch
import torch.utils.data as torch_data

from models import Darknet
from detector import detector, setup_detector
from visualize import visualize_func
from evaluation import get_batch_statistics_rotated_bbox

from utils.astyx_yolo_dataset import AstyxYOLODataset
import utils.config as cnf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="network/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou thresshold for evaluation")
    parser.add_argument("--split", type=str, default="valid", help="text file having image lists in dataset")
    parser.add_argument("--radar", default=False, action="store_true" , help="Use Radar Data")
    parser.add_argument("--estimate_bb", default=False, action="store_true", help="Whether to estimate Bounding Boxes")
    parser.add_argument("--visualize", default=False, action="store_true", help="Whether to visualize the data")
    parser.add_argument("--evaluate", default=False, action="store_true", help="Whether to evaluate the detection")
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists("output"):
        os.makedirs("output")

    if opt.estimate_bb:
        # if we want to detect objects we have to setup the model for our purpose
        model = setup_detector(opt)
        if opt.evaluate:
            ngt = 0 # number of all targets
            true_positives = []
            pred_scores = []
        
    # Load the Astyx dataset
    dataset = AstyxYOLODataset(cnf.root_dir, split=opt.split, mode="EVAL", radar=opt.radar)
    data_loader = torch_data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    # loop over all frames from the split file
    for index, (sample_id, bev_maps, targets) in enumerate(data_loader):
        # Stores detections for each image index
        img_detections = []
        
        # Targets position and dimension values are between 0 - 1, so that they
        # have to be transformed to pixel coordinates
        targets[:, 2:] *= cnf.BEV_WIDTH
        
        if opt.estimate_bb:
            # detects objects
            predictions = detector(model, bev_maps, opt)
            img_detections.extend(predictions)
            # Calculate if the prediction is a true detection
            if opt.evaluate:
                ngt += len(targets)
                true_positive, pred_score = get_batch_statistics_rotated_bbox(predictions, targets, opt.iou_thres)
                """
                Concatenate all true_positives and pred_scores to two long true_positives and pred_scores lists.
                """
                
        # Visualization of the ground truth and if estimated the predicted boxes
        if opt.visualize:
            visualize_func(bev_maps[0], targets, img_detections, sample_id, opt.estimate_bb)
        
    if opt.estimate_bb and opt.evaluate:
        AP = calculate_ap(true_positives, pred_scores, ngt)
