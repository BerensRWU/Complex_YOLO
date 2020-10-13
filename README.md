# ComplexYOLO
This repository contains a PyTorch implementation of [ComplexYOLO](https://arxiv.org/pdf/1803.06199.pdf). It is build to be applied on the data from the Astyx Dataset.

## Installation
#### Clone the project and install requirements
    $ git clone https://github.com/BerensRWU/ComplexYOLO/
    
### Requirements
To run the program you need to install those libraries with dependencies:
  * torch (Tested with pytorch version 1.3.0)
  * cv2 (only for visualization)
  * json
  * numpy
  * shapely
  
An easy way to install this is with [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then you can install e.g. torch:
```
conda install pytorch torchvision cpuonly -c pytorch
```
  
## Start
With visualize.py you have a script that can be used to visualize the point clouds in bird's eye view and the labeld objects in the corresponding scene.
```
python visualize.py
```
#### Arguments:
  * ```estimate_bb```: Flag, to estimate bounding boxes. Default ```False```.
  * ```radar```: Flag, if True use radar data otherwise lidar. Default ```False```.
  * ```model_def```: Path to model definition file. Default ```config/yolov3-custom.cfg```.
  * ```weights_path```: Path where the weights are saved. Default ```checkpoints```.
  * ```conf_thres```: If bounding boxes are estimated, only those with a confidence greater than thethreshold will be used. Default ```0.5```.
  * ```nms_thres```: If estimated bounding boxes overlap with an IoU greater than the ```nms_thres``` only the bounding box with highest confidence remains. Default ```0.5```.
  * ```split```: Which split to use ```valid```, ```train```. Default ```valid```.
  
To use the ComplexYOLO bounding box estimator we need non-maximum-supression. For non-maximum-supression we need the IoU.

# Astyx HiRes
The Astyx HiRes is a dataset from Astyx for object detection for autonomous driving. Astyx has a sensor setup consisting of camera, LiDAR, RADAR. Additional information can be found here: [Dataset Paper](https://www.astyx.com/fileadmin/redakteur/dokumente/Automotive_Radar_Dataset_for_Deep_learning_Based_3D_Object_Detection.PDF) and [Specification](https://www.astyx.com/fileadmin/redakteur/dokumente/Astyx_Dataset_HiRes2019_specification.pdf)

```
└── dataset/
       ├── dataset_astyx_hires2019    <-- 546 data
       |   ├── calibration 
       |   ├── camera_front
       |   ├── groundtruth_obj3d
       |   ├── lidar_vlp16
       |   ├── radar_6455
       └── split
           ├── train.txt
           └── valid.txt
```
# Evaluation
To evaluate the valid split we need a function for IoU and AP. For an confidence threshold of 0.5, non-maximum-supression threshold of 0.5 and a IoU threshold of 0.5 we get an average precision (AP) in the range of:

 Model - Sensor/Class              | Car     | 
| ----------------------- |:--------|
| ComplexYOLO LiDAR       | 65 - 75 |
| ComplexYOLO RADAR       | 65 - 75   |
