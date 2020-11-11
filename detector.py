import os
import torch
from torch.autograd import Variable

from models import Darknet
from utils import utils
import utils.config as cnf


def setup_detector(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = os.path.join(opt.weights_path, "weights_RADAR.pth" if opt.radar else "weights_LIDAR.pth")
    # Set up model
    model = Darknet(opt.model_def, img_size=cnf.BEV_WIDTH).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path, map_location = device))
    # Eval mode
    model.eval()

    return model

def detector(model, bev_maps, opt):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Configure bev image
    input_imgs = Variable(bev_maps.type(Tensor))
    # Get detections 
    with torch.no_grad():
        detections = model(input_imgs)
        detections = utils.non_max_suppression_rotated_bbox(detections, opt.conf_thres, opt.nms_thres) 

    return detections
