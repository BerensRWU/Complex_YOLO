import numpy as np
import torch

from utils.utils import rotated_bbox_iou_polygon

def get_batch_statistics_rotated_bbox(predictions, targets, iou_threshold):
    """ Compute true positives and predicted scores per sample.
    Input:
        predictions: numpy array of all predictied bounding boxes, after non maximum supression
        targets: numpy array of all ground truth bounding boxes
        iou_threshodl: int, threshold which prediction is a positive detection or not
        
    Output:
        bool list of true positive and false positive detections
        list of the score values of the predictions
    """
    batch_metrics = []
    
    # loop over all batches
    for sample_i, prediction in enumerate(predictions):

        if prediction is None:
            continue

        pred_boxes = prediction[:, :6] # predicted boundig box
        pred_scores = prediction[:, 6] # predicted score / confidence
        pred_labels = prediction[:, -1] # predicted label

        true_positives = np.zeros(pred_boxes.shape[0])
        
        # get the ground truth of the batch sample_i
        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]
            
            # loop over all predictions
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                #iou, box_index = rotated_bbox_iou(pred_box.unsqueeze(0), target_boxes, 1.0, False).squeeze().max(0)
                ious = rotated_bbox_iou_polygon(pred_box, target_boxes)
                
                raise NotImplementedError
                """
                Get the maximum iou value for all predictions and the corresponding index.
                Check if the maxIOU is greater or equal than  iou_threshold and the index not in detected_boxes.
                    If true set true_positives[pred_i] equal 1 and add the index to detected_boxes.
                After all loops return the list of true positives and the list of the scores"""
                
def calculate_ap(true_positives, pred_scores, ngt):
        """
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        return: ap
        
        Steps:
        1. Calculate a list of False positves (1 - true positves)
        2. Sort the lists according to the scores
        3. Calculate the Cumulative sums of the sorted Lists
        4. Calculate the precision and recall for every entry of the cumulative sums
            precision = cumsumTP/(cumsumTP + cumsumFP)
            recall = cumsumTP / ngt
        5. Concatenate 0 / 1 to the start/ending of the precision list
           Concatenate 0 / 0 to the start/ending of the recall list
        6. Calculate the approximated AP as described in the Lecture ( AP-11, AP-all)
        """
        
        pass
