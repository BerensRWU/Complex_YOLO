import torch
import numpy as np

root_dir = "dataset/" # Where do you have saved the data

class_list = ["Car"]

CLASS_NAME_TO_ID = {
            'Car': 			0
        }

# Front side (of vehicle) Point Cloud boundary for BEV
boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

BEV_WIDTH = 608 # across y axis -25m ~ 25m
BEV_HEIGHT = 608 # across x axis 0m ~ 50m

DISCRETIZATION = (boundary["maxX"] - boundary["minX"])/BEV_HEIGHT
