import numpy as np
import math
import cv2
import utils.config as cnf

def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
            PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]
    
    # Set the minmum height to zero
    PointCloud[:, 2] = PointCloud[:, 2] - minZ

    return PointCloud

def makeBVFeature(PointCloud_, Discretization, bc):
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map, Intensity Map & DensityMap
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # Points that map to the same pixel, only that with the largest height should be used 
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_frac = PointCloud[indices]
    
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(bc['maxZ'] - bc['minZ']))
    # get height map
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height
    # get intensity map
    intensityMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 3]
    # get density map
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    densityMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = normalizedCounts
    
    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
    return RGB_Map

def read_labels_for_bevbox(objects):
    # Read all bounding boxes
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox = []
            bbox.append(obj.cls_id)
            bbox.extend([obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry])
            bbox_selected.append(bbox)
    if (len(bbox_selected) == 0):
        return np.zeros((1, 8), dtype=np.float32), True
    else:
        bbox_selected = np.array(bbox_selected).astype(np.float32)
        return bbox_selected, False

def get_corners(x, y, w, l, yaw):
    # bev image coordinates format
    bev_corners = np.zeros((4, 2), dtype=np.float32)

    # front left
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    # rear left
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # rear right
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # front right
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    return bev_corners

def build_yolo_target(labels):
    # transform the labels such that they fit the yolo method
    bc = cnf.boundary
    target = np.zeros([50, 7], dtype=np.float32)
    
    index = 0
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]

        # ped and cyc labels are very small, so lets add some factor to height/width
        l = l + 0.3
        w = w + 0.3
        
        yaw = np.pi * 2 - yaw
        # check if the bounding box fits in the range we are interested
        if (x > bc["minX"]) and (x < bc["maxX"]) and (y > bc["minY"]) and (y < bc["maxY"]):
            y1 = (y - bc["minY"]) / (bc["maxY"]-bc["minY"])  # we should put this in [0,1], so divide max_size  80 m
            x1 = (x - bc["minX"]) / (bc["maxX"]-bc["minX"])  # we should put this in [0,1], so divide max_size  40 m
            w1 = w / (bc["maxY"] - bc["minY"])
            l1 = l / (bc["maxX"] - bc["minX"])
            
            # build the target
            target[index][0] = cl
            target[index][1] = y1 
            target[index][2] = x1
            target[index][3] = w1
            target[index][4] = l1
            target[index][5] = math.sin(float(yaw))
            target[index][6] = math.cos(float(yaw))

            index = index+1

    return target
