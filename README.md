# OverView of 3D object detection method

In order to facilitate the papers that I have already seen, I will organize the paper related to 3D object detection.This will include an algorithm based on deep learning and multimode fusion algorithms.

![流程图 drawio](https://user-images.githubusercontent.com/44192081/157437988-2a034f4e-5902-4634-ae39-44d8e31fab05.png)



# papper list 
## object detection without fusion


 
<center>The time axis of method of Single mode depth learning detection </center>
<div style="align: center">
<img src="https://user-images.githubusercontent.com/44192081/157589861-dc6e6a48-195a-4002-b0b5-f8493c6e7c7e.png"/>
</div>
 
Method | Title | Input  | Pub. | Author
--------- | ------------- | ------------- | ------------- | -------------
Monocular based | Deep3DBox: [3D Bounding Box Estimation Using Deep Learning and Geometry ](https://openaccess.thecvf.com/content_cvpr_2017/html/Mousavian_3D_Bounding_Box_CVPR_2017_paper.html) |  Monocular Image | CVPR 2017 |Chen et al.
Monocular based |  MonoCon : [Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection](https://xianpeng-liu.com/publication/learning-auxiliary-monocular-contexts-helps-monocular-3d-object-detection/learning-auxiliary-monocular-contexts-helps-monocular-3d-object-detection.pdf) | Monocular Image | arXiv 2021 |Liu et al.
Monocular based |Mono3D-PLiDAR : [Monocular 3d object detection with pseudo-lidar point cloud](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Weng_Monocular_3D_Object_Detection_with_Pseudo-LiDAR_Point_Cloud_ICCVW_2019_paper.html) |Monocular Image | ICCV 2019 | Weng et al.
Monocular based | M3DSSD: [onocular 3D Single Stage Object Detector](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_M3DSSD_Monocular_3D_Single_Stage_Object_Detector_CVPR_2021_paper.pdf) |Monocular Image | CVPR 2021 | Luo et al.
Monocular based | MonoRUn: [Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_MonoRUn_Monocular_3D_Object_Detection_by_Reconstruction_and_Uncertainty_Propagation_CVPR_2021_paper.html)|Monocular Image | CVPR 2021 | chen et al.
Stereo based | 3DOP: [3D Object Proposals using Stereo Imagery for Accurate Object Class Detection](https://ieeexplore.ieee.org/abstract/document/7932113) |Monocular Image | NIPS 2015 | Chen et al.
Stereo based | Liga-stereo: [Learning lidar geometry aware representations for stereo-based 3d detector](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_LIGA-Stereo_Learning_LiDAR_Geometry_Aware_Representations_for_Stereo-Based_3D_Detector_ICCV_2021_paper.html)|Stereo Image| CVPR 2021| Guo et al.
Stereo based |CG-Stereo : [Confidence guided stereo 3D object detection with split depth estimation](https://ieeexplore.ieee.org/abstract/document/9341188)| Stereo Image|| CVPR 2021| Guo et al.
Stereo based | Stereo R-CNN :[Stereo R-CNN Based 3D Object Detection for Autonomous Driving](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Stereo_R-CNN_Based_3D_Object_Detection_for_Autonomous_Driving_CVPR_2019_paper.html)|Stereo Image| CVPR 2019| Li et al.
MultiView based | VeloFCN : [Vehicle detection from 3d lidar using fully convolutional network](https://arxiv.org/abs/1608.07916)|Front View,FV| CVPR 2016| Li et al.
MultiView based | BirdNet : [Birdnet: a 3d object detection framework from lidar information](https://ieeexplore.ieee.org/abstract/document/8569311)|Bird’s Eye of View,BEV | CVPR 2018| Jorge et al.
MultiView based | Pixor: [Real-time 3d object detection from point clouds](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.html)|BEV| CVPR 2018| Yang et al.
MultiView based | Hdnet: [Exploiting hd maps for 3d object detection](http://proceedings.mlr.press/v87/yang18b)|BEV| PMLR 2018| Yang et al.



## multimodel object detection
![流程图 drawio](https://user-images.githubusercontent.com/44192081/157562681-b7d4d5cb-ac9f-490f-b3a6-45b03c459505.png)

Title | Pub. | Author
--------- | ------------- | -------------
MV3D :[ Multi-View 3D Object Detection Network for Autonomous Driving](https://ieeexplore.ieee.org/document/8100174)| CVPR 2017  |  chen et al.
AVOD : [Joint 3D Proposal Generation and Object Detection from View Aggregation](https://ieeexplore.ieee.org/abstract/document/8594049)| IROS 2018  | Ku et al.
SCANet:[ Spatial-channel attention network for 3D object detection](https://ieeexplore.ieee.org/abstract/document/8682746)| ICASSP 2019| Lu et al.
MVX-net:[ Multimodal voxelnet for 3d object detection](https://ieeexplore.ieee.org/abstract/document/8794195) | ICRA 2019 | Sindagi et al.
MMF : [Multi-task multi-sensor fusion for 3d object detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Liang_Multi-Task_Multi-Sensor_Fusion_for_3D_Object_Detection_CVPR_2019_paper.html) | CVPR 2019 | liang et al.
CLOCs: [Camera-LiDAR Object Candidates Fusion for 3D Object Detection](https://ieeexplore.ieee.org/abstract/document/9341791/) | IROS 2020| Peng et al.
ContFusion : [Deep continuous fusion for multi-sensor 3d object detection](https://openaccess.thecvf.com/content_ECCV_2018/html/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.html)| ECCV 2018 |Liang et al. 
Pointfusion: [Deep sensor fusion for 3d bounding box estimation](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_PointFusion_Deep_Sensor_CVPR_2018_paper.html) | CVPR 2018 | Xu et al.
Pointpainting: [Sequential fusion for 3d object detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.html) | CVPR 2020 |Lang et al.
Epnet: [Enhancing point features with image semantics for 3d object detection](https://arxiv.org/pdf/2007.08856.pdf) | ECCV 2020 | Huang et al.
PI-RCNN: [An efficient multi-sensor 3D object detector with point-based attentive cont-conv fusion module](https://ojs.aaai.org/index.php/AAAI/article/view/6933)| AAAI 2020 | Xiang et al.
MoCa : [Multi-Modality Cut and Paste for 3D Object Detection](https://ui.adsabs.harvard.edu/abs/2020arXiv201212741Z/abstract) | arXiv 2020 | Zhang et al.
PointAugmenting: [Cross-Modal Augmentation for 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_PointAugmenting_Cross-Modal_Augmentation_for_3D_Object_Detection_CVPR_2021_paper.html?utm_campaign=Akira%27s%20Machine%20Learning%20News%20%28ja%29&utm_medium=email&utm_source=Revue%20newsletter) | CVPR 2021 | Wang et al.
Imvotenet: [Boosting 3d object detection in point clouds with image votes](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_ImVoteNet_Boosting_3D_Object_Detection_in_Point_Clouds_With_Image_CVPR_2020_paper.html) | CVPR 2020| Charles Qi et al.
Pseudo-LiDAR From Visual Depth Estimation: [Bridging the Gap in 3D Object Detection for Autonomous Driving](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Pseudo-LiDAR_From_Visual_Depth_Estimation_Bridging_the_Gap_in_3D_CVPR_2019_paper.html) | CVPR 2019 |Wang et al.
Roarnet: [A robust 3d object detection based on region approximation refinement](https://ieeexplore.ieee.org/abstract/document/8813895)| IEEE.IV 2019|Shin et al.
Frustum PointNet : [Frustum pointnets for 3d object detection from rgb-d data](https://openaccess.thecvf.com/content_cvpr_2018/html/Qi_Frustum_PointNets_for_CVPR_2018_paper.html) | CVPR 2018 | Qi et al.
Frustum ConvNet : [Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection](https://ieeexplore.ieee.org/abstract/document/8968513) | IROS 2019 | Wang et al.

 
## Selfsupervised Learning
Title | Pub. | Author
--------- | ------------- | -------------
Data2vec: [A General Framework for Self-supervised Learning in Speech, Vision and Language](https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language)|  2022| MetaAI

## Unsupervised Learning
Title | Pub. | Author
--------- | ------------- | -------------
[Unsupervised Learning of Depth from Monocular Videos Using 3D-2D Corresponding Constraints ](https://www.mdpi.com/2072-4292/13/9/1764) |Remote Sensing 2021| Jin et al.


## Point Cloud Local Feature Description
Title | Pub. | Author
--------- | ------------- | -------------
2D Shape Context:[ Shape Context: A new descriptor for shape matching and object recognition](https://en.wikipedia.org/wiki/Shape_context#Step_6:_Computing_the_shape_distance) | NeurIPS 2000 | Serge Belongie et al.
3D Shape Context:[Recognizing Objects in Range Data Using Regional Point Descriptors](https://link.springer.com/chapter/10.1007/978-3-540-24672-5_18) | ECCV 2004|Andrea et al.
ROI-cloud: [A Key Region Extraction Method for LiDAR Odometry and Localization](https://ieeexplore.ieee.org/abstract/document/9197059) | ICRA 2020 | Zhou et al.
PointSIFT: [A sift-like network module for 3D point cloud semantic segmentation](https://arxiv.org/abs/1807.00652) | CVPR 2018 | Jiang et al.

## DataSet
DataSet | Size | Categories / Remarks |Sensing Modalities
--------- | ------------- | ------------- | -------------
[ScanNet](http://www.scan-net.org/) | | |
[SUN RGB-D](http://rgbd.cs.princeton.edu)
[SUN3D](http://sun3d.cs.princeton.edu)
[KITTI](http://www.cvlibs.net/datasets/kitti/)|  7481 frames (training) 80.256 objects| Car, Van, Truck, Pedestrian,Person (sitting), Cyclist, Tram,Misc| Visual (Stereo) camera, 3D LiDAR, GNSS, and inertial sensors
[nuScense](https://www.nuscenes.org/download) |1000 scenes, 1.4M frames (camera, Radar), 390k frames (3D LiDAR)| 25 Object classes, such as Car /Van / SUV, different Trucks,Buses, Persons, Animal, Traffic Cone, Temporary Traffic Barrier, Debris, etc.|Visual cameras (6), 3D LiDAR, and Radars (5)|
