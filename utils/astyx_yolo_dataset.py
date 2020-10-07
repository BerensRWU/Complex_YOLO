import os
import numpy as np
from utils.astyx_dataset import AstyxDataset
import utils.astyx_bev_utils as bev_utils
import utils.config as cnf
import torch
import torch.nn.functional as F

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class AstyxYOLODataset(AstyxDataset):

    def __init__(self, root_dir, split='train', mode ='EVAL', radar = False):
        super().__init__(root_dir=root_dir, split=split)

        self.split = split
        self.max_objects = 100
        self.radar = radar

        assert mode == 'EVAL', 'Invalid mode: %s' % mode
        self.mode = mode

        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

        print('Load %s samples from %s' % (mode, self.imageset_dir))
        print('Done: total %s samples %d' % (mode, len(self.sample_id_list)))

    def __getitem__(self, index):
        
        sample_id = int(self.sample_id_list[index])

        if self.mode in ['TRAIN', 'EVAL']:
            
                
            objects = self.get_label(sample_id)   
            calib = self.get_calib(sample_id)
            
            if self.radar:
                pcData = self.get_radar(sample_id)
            else:
                pcData = self.get_lidar(sample_id)
                intensity = pcData[:,3].reshape(-1,1)
                pcData = calib.lidar2ref(pcData[:,0:3])
                pcData = np.concatenate([pcData,intensity],1)
            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)
            

            b = bev_utils.removePoints(pcData, cnf.boundary)
            rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            target = bev_utils.build_yolo_target(labels)

            ntargets = 0
            for i, t in enumerate(target):
                if t.sum(0):
                    ntargets += 1            
            targets = torch.zeros((ntargets, 8))
            for i, t in enumerate(target):
                if t.sum(0):
                    targets[i, 1:] = torch.from_numpy(t)
            
            img = torch.from_numpy(rgb_map).type(torch.FloatTensor)
            
            return sample_id, img, targets

    def __len__(self):
        return len(self.sample_id_list)
