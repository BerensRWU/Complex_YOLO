# ComplexYOLO
This repository contains a PyTorch implementation of [ComplexYOLO](https://arxiv.org/pdf/1803.06199.pdf). It is build to be applied on the data from the Astyx Dataset.

## Installation
#### Clone the project and install requirements
    $ git clone https://github.com/BerensRWU/ComplexYOLO/
    
### Requirements
The script was tested on the following systems with the following versions:
#### Ubuntu 18.04.5 LTS
To run the program you need to install those libraries with dependencies:
  * torch (Tested with pytorch version 1.6.0, torchvision 0.7.0)
  * cv2 
  * json
  * numpy 
  * shapely 
  
  
#### Windows 10
To run the program you need to install those libraries with dependencies:
  * torch (Tested with pytorch version 1.5.1, torchvision 0.6.1)
  * cv2 
  * json
  * numpy 
  * shapely 
  
An easy way to install this is with [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then you can install e.g. torch:
```
conda install pytorch torchvision cpuonly -c pytorch
```
  
## Start
main.py.
```
python main.py
```
#### Arguments:
  * ```estimate_bb```: Flag, to estimate bounding boxes. Default ```False```.
  * ```radar```: Flag, if True use radar data otherwise lidar. Default ```False```.
  * ```model_def```: Path to model definition file. Default ```config/yolov3-custom.cfg```.
  * ```weights_path```: Path where the weights are saved. Default ```checkpoints```.
  * ```conf_thres```: If bounding boxes are estimated, only those with a confidence greater than thethreshold will be used. Default ```0.5```.
  * ```iou_thres```: If bounding boxes are estimated, threshold for a correct detection. Default ```0.5```.
  * ```nms_thres```: If estimated bounding boxes overlap with an IoU greater than the ```nms_thres``` only the bounding box with highest confidence remains. Default ```0.2```.
  * ```split```: Which split to use ```valid```, ```train```. Default ```valid```.
  * ```visualize```: Whether to visualize the data.
  * ```evaluate```: Whether to evaluate the data.
  
To use the ComplexYOLO bounding box estimator we need non-maximum-supression. For non-maximum-supression we need the IoU.

### visualize.py
A script for the visualization of the data.

### detector.py
A script for the detection of bounding boxes from BEV.

### evaluation.py
A script for the evaluation of the predicted bounding boxes. Here we need to calculate whether the prediction is a true positive or not and the average precision.

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
To evaluate the valid split we need a function for IoU and AP. For an confidence threshold of 0.5, non-maximum-supression threshold of 0.2 and a IoU threshold of 0.5 we get an average precision (AP) in the range of:

 Model - Sensor/Class | Car     | 
| ------------------- |:--------|
| ComplexYOLO LiDAR   | 65 - 75 |
| ComplexYOLO RADAR   | 65 - 75 |

# Google Colab
The visualize.ipynb is for those who want to run the program in google colab.

You have to upload the other scripts to colab. Also you have to upload the data files to colab or to your google drive(if you upload it to google drive you must give colab acces to your drive, and change in utils->config->root_dir to "drive/My Drive/dataset" or wherever you saved the dataset). 
