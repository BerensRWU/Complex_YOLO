import torch
import numpy as np
from shapely.geometry import Polygon
import utils.astyx_bev_utils as bev_utils

def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    The IoU is the fraction between the area of the intersection and the union 
        the bounding boxes.
    box: a polygon
    boxes: a vector of polygons
    """
    
    raise NotImplementedError # this is you job to write this Function.

def to_cpu(tensor):
    return tensor.detach().cpu()

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes

def rotated_bbox_iou_polygon(box1, box2):
    """
        Calculate the IoU between box1 (list of only one box) and 
            box2 (list of boxes).
        First the elements of both lists are converted to shapley 
            Polygon (convert_format).
        After this the IoU has to be computed. This is your job.
        
    """
    box1 = to_cpu(box1).numpy()
    box2 = to_cpu(box2).numpy()

    x,y,w,l,im,re = box1
    angle = np.arctan2(im, re)
    bbox1 = np.array(bev_utils.get_corners(x, y, w, l, angle)).reshape(-1,4,2)
    bbox1 = convert_format(bbox1)

    bbox2 = []
    for i in range(box2.shape[0]):
        x,y,w,l,im,re = box2[i,:]
        angle = np.arctan2(im, re)
        bev_corners = bev_utils.get_corners(x, y, w, l, angle)
        bbox2.append(bev_corners)
    bbox2 = convert_format(np.array(bbox2))
    return compute_iou(bbox1[0], bbox2)

def non_max_suppression_rotated_bbox(prediction, conf_thres=0.95, nms_thres=0.4):
    """ 
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x, y, w, l, im, re, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 6] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 6] * image_pred[:, 7:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 7:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :7].float(), class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = rotated_bbox_iou_polygon(detections[0, :6], detections[:, :6]) > nms_thres
            large_overlap = torch.from_numpy(large_overlap)
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 6:7]
            # Merge overlapping bboxes by order of confidence
            detections[0, :6] = (weights * detections[invalid, :6]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output
