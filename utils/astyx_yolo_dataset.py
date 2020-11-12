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
        self.radar = radar

        assert mode == 'EVAL', 'Invalid mode: %s' % mode
        self.mode = mode

        self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

        print('Load %s samples from %s' % (mode, self.dataset_dir))
        print('Done: total %s samples %d' % (mode, len(self.sample_id_list)))

    def __getitem__(self, index):
        
        sample_id = int(self.sample_id_list[index])

        if self.mode in ['TRAIN', 'EVAL']:
            
                
            objects = self.get_label(sample_id)   
            calib = self.get_calib(sample_id)
            
            if self.radar:
                # If we use RADAR we do only load the data
                pcData = self.get_radar(sample_id)
            else:
                # If we use LiDAR we have to transform the point cloud to the RADAR coordinate system
                pcData = self.get_lidar(sample_id)
                intensity = pcData[:,3].reshape(-1,1) # save the intensity
                pcData = calib.lidar2ref(pcData[:,0:3]) # transformation from lidar coordinatesystem to radar coordinatesystem
                pcData = np.concatenate([pcData,intensity],1) # concatenate the transformed the point cloud with the intensity
            
            # Read all bounding boxes
            labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects)
            
            # Remove points of the point cloud that are not in the range we are focusing
            b = bev_utils.removePoints(pcData, cnf.boundary)
            # Generate the BEV map
            rgb_map = bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            # Transform the groundtruth such that it fits for the model
            target = bev_utils.build_yolo_target(labels)

            ntargets = 0
            # count the number of ground truth objects, because in build_yolo_target an array that is lager than the number of objects is generated
            for i, t in enumerate(target):
                if t.sum(0):
                    ntargets += 1            
            targets = torch.zeros((ntargets, 8))
            # store the targets now in an array that fits the number of ground truth objects
            for i, t in enumerate(target):
                if t.sum(0):
                    targets[i, 1:] = torch.from_numpy(t)
            
            img = torch.from_numpy(rgb_map).type(torch.FloatTensor) # cast to torch.tensor
            
            return sample_id, img, targets

    def __len__(self):
        return len(self.sample_id_list)

    def collate_fn(self, batch):
        # this function defines how batches should be concatenated
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Resize images to input shape
        imgs = torch.stack([resize(img, cnf.BEV_WIDTH) for img in imgs])
        return paths, imgs, targets
