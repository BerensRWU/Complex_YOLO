import numpy as np
import json
import utils.config as cnf
import math

class Object3d(object):
    ''' 3d object label '''
    def __init__(self, obj):
        # extract label, truncation, occlusion
        self.type = obj["classname"]
        self.cls_id = self.cls_type_to_id(self.type)

        self.occlusion = obj["occlusion"] 
        self.quat = obj["orientation_quat"]
        self.ry = self.qaut_to_angle()[2] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        
        # extract 3d bounding box information
        self.h = obj["dimension3d"][2] # box height
        self.w = obj["dimension3d"][1] # box width
        self.l = obj["dimension3d"][0] # box length (in meters)
        self.t = (obj["center3d"][0],obj["center3d"][1],obj["center3d"][2]) # location (x,y,z) in camera coord.
        self.dis_to_cam = np.linalg.norm(self.t)
        self.score = obj["score"]

    def qaut_to_angle(self):
        w = self.quat[0]
        x = self.quat[1]
        y = self.quat[2]
        z = self.quat[3]
        return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)), math.asin(2*(w*y-z*x)),
                math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))  

    def cls_type_to_id(self, cls_type):
        if cls_type not in cnf.CLASS_NAME_TO_ID.keys():
            return -1
        return cnf.CLASS_NAME_TO_ID[cls_type]

class calib_astyx():
    """Calibration class"""
    def __init__(self, file):
        with open(file) as json_file:
            data = json.load(json_file)
            
        self.radar2ref = np.array(data["sensors"][0]["calib_data"]["T_to_ref_COS"])
        self.lidar2ref_cos = np.array(data["sensors"][1]["calib_data"]["T_to_ref_COS"])
        self.camera2ref = np.array(data["sensors"][2]["calib_data"]["T_to_ref_COS"])
        self.K = np.array(data["sensors"][2]["calib_data"]["K"])
        
        self.ref2radar = self.inv_trans(self.radar2ref)
        self.ref2lidar = self.inv_trans(self.lidar2ref_cos)
        self.ref2camera = self.inv_trans(self.camera2ref)
        
    @staticmethod
    def inv_trans(T):
        rotation = np.linalg.inv(T[0:3, 0:3])
        translation = T[0:3, 3]
        translation = -1 * np.dot(rotation, translation.T)
        translation = np.reshape(translation, (3, 1))
        Q = np.hstack((rotation, translation))

        return Q
    
    def lidar2ref(self, points):
        n = points.shape[0]
        
        points_hom = np.hstack((points, np.ones((n,1))))
        points_ref = np.dot(points_hom, np.transpose(self.lidar2ref_cos))
        
        return points_ref[:,0:3]

def read_label(label_filename):
    with open(label_filename) as json_file:
        data = json.load(json_file)

    objects = [Object3d(obj) for obj in data["objects"]]

    return objects
