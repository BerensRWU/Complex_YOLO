import numpy as np
import torch

from utils.utils import rotated_bbox_iou_polygon

def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample.
    Input:
        predictions: numpy array of all predictied bounding boxes, after non maximum supression
        targets: numpy array of all ground truth bounding boxes
        iou_threshodl: int, threshold which prediction is a positive detection or not
    """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :6]
        pred_scores = output[:, 6]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                #iou, box_index = rotated_bbox_iou(pred_box.unsqueeze(0), target_boxes, 1.0, False).squeeze().max(0)
                ious = rotated_bbox_iou_polygon(pred_box, target_boxes)
                iou, box_index = torch.from_numpy(ious).max(0)

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores])
    return batch_metrics
                
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap_all = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    ap_11 = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap_11 = ap_11 + p / 11.
        
    return ap_all, ap_11


def evaluate(tp, conf, n_gt):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        n_gt: Number of all ground truht objects
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]

    # Create Precision-Recall curve and compute AP 
    ap, p, r = [], [], []
    # Accumulate FPs and TPs
    fpc = (1 - tp).cumsum()
    tpc = (tp).cumsum()

    # Recall
    recall_curve = tpc / (n_gt + 1e-16)

    # Precision
    precision_curve = tpc / (tpc + fpc)

    # AP from recall-precision curve
    ap_all, ap_11 = compute_ap(recall_curve, precision_curve)

    return ap_all, ap_11    
