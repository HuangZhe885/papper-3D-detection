

# OverView of 3D object detection method

In order to facilitate the papers that I have already seen, I will organize the paper related to 3D object detection.This will include an algorithm based on deep learning and multimode fusion algorithms.

 

![流程图 drawio](https://user-images.githubusercontent.com/44192081/157608778-48803592-9386-4f7e-9948-3bd4dca1927a.png)


# papper list 
 
   * [Survey](#survey) 
   * [object detection without fusion](#object-detection-without-fusion)
   * [multimodel object detection](#multimodel-object-detection)
   * [Selfsupervised Learning](#selfsupervised-learning)
   * [Unsupervised Learning](#unsupervised-learning)
   * [Point Cloud Local Feature Description](#point-cloud-local-feature-description)
   * [DataSet](#dataset)


## survey
Method | Title | Author
--------- | --------- | ------------- |
object detection |[Foreground-Background Imbalance Problem in Deep Object Detectors: A Review](https://arxiv.org/abs/2006.09238)| Joya Chen, Tong Xu
object detection |[A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving](https://arxiv.org/abs/2011.10671)|Di Feng,Ali Harakeh,Steven Waslander
object detection |[An Overview Of 3D Object Detection](https://arxiv.org/abs/2010.15614)|Yilin Wang, Jiayi Ye
object detection |[3D Object Detection for Autonomous Driving: A Survey](https://arxiv.org/abs/2106.1082)|Rui Qian, Xin Lai
MultiModel |[Multi-Modal 3D Object Detection in Autonomous Driving: a Survey](https://arxiv.org/pdf/2106.12735.pdf)|Yingjie Wang,Qiuyu Mao
MultiModel |[Deep Multi-modal Object Detection and Semantic Segmentation for Autonomous Driving: Datasets,Methods, and Challenges](https://arxiv.org/pdf/1902.07830.pdf?ref=https://githubhelp.com)|Di Feng,Christian Haase-Schutz
MultiModel |[Deep Learning for Image and Point Cloud Fusion in Autonomous Driving: A Review](https://arxiv.org/pdf/2004.05224.pdf)|Yaodong Cui


## object detection without fusion


 ![非融合算法时间序列 drawio](https://user-images.githubusercontent.com/44192081/157609450-5add851e-92e6-482a-b93f-752f34c2206c.png)

Method | Title | Input  | Pub. | Author
--------- | ------------- | ------------- | ------------- | -------------
Monocular based | Deep3DBox: [3D Bounding Box Estimation Using Deep Learning and Geometry ](https://openaccess.thecvf.com/content_cvpr_2017/html/Mousavian_3D_Bounding_Box_CVPR_2017_paper.html) |  Monocular Image | CVPR 2017 |Chen et al.
Monocular based |  MonoCon : [Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection](https://xianpeng-liu.com/publication/learning-auxiliary-monocular-contexts-helps-monocular-3d-object-detection/learning-auxiliary-monocular-contexts-helps-monocular-3d-object-detection.pdf) | Monocular Image | arXiv 2021 |Liu et al.
Monocular based |Mono3D-PLiDAR : [Monocular 3d object detection with pseudo-lidar point cloud](https://openaccess.thecvf.com/content_ICCVW_2019/html/CVRSUAD/Weng_Monocular_3D_Object_Detection_with_Pseudo-LiDAR_Point_Cloud_ICCVW_2019_paper.html) |Monocular Image | ICCV 2019 | Weng et al.
Monocular based | M3DSSD: [onocular 3D Single Stage Object Detector](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_M3DSSD_Monocular_3D_Single_Stage_Object_Detector_CVPR_2021_paper.pdf) |Monocular Image | CVPR 2021 | Luo et al.
Monocular based | MonoRUn: [Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_MonoRUn_Monocular_3D_Object_Detection_by_Reconstruction_and_Uncertainty_Propagation_CVPR_2021_paper.html)|Monocular Image | CVPR 2021 | chen et al.
Stereo based | 3DOP: [3D Object Proposals using Stereo Imagery for Accurate Object Class Detection](https://ieeexplore.ieee.org/abstract/document/7932113) |Monocular Image | NIPS 2015 | Chen et al.
Stereo based | Liga-stereo: [Learning lidar geometry aware representations for stereo-based 3d detector](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_LIGA-Stereo_Learning_LiDAR_Geometry_Aware_Representations_for_Stereo-Based_3D_Detector_ICCV_2021_paper.html)|Stereo Image| CVPR 2021| Guo et al.
Stereo based |CG-Stereo : [Confidence guided stereo 3D object detection with split depth estimation](https://ieeexplore.ieee.org/abstract/document/9341188)| Stereo Image| IROS 2020| Li et al.
Stereo based | Stereo R-CNN :[Stereo R-CNN Based 3D Object Detection for Autonomous Driving](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Stereo_R-CNN_Based_3D_Object_Detection_for_Autonomous_Driving_CVPR_2019_paper.html)|Stereo Image| CVPR 2019| Li et al.
MultiView based | VeloFCN : [Vehicle detection from 3d lidar using fully convolutional network](https://arxiv.org/abs/1608.07916)|Front View,FV| CVPR 2016| Li et al.
MultiView based | BirdNet : [Birdnet: a 3d object detection framework from lidar information](https://ieeexplore.ieee.org/abstract/document/8569311)|Bird’s Eye of View,BEV | CVPR 2018| Jorge et al.
MultiView based | Pixor: [Real-time 3d object detection from point clouds](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.html)|BEV| CVPR 2018| Yang et al.
MultiView based | Hdnet: [Exploiting hd maps for 3d object detection](http://proceedings.mlr.press/v87/yang18b)|BEV| PMLR 2018| Yang et al.
MultiView based |LaserNet: [An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://openaccess.thecvf.com/content_CVPR_2019/html/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.html)|Range View,RV| CVPR 2019| Meyer et al.
Voxel based |Voxelnet: [End-to-end learning for point cloud based 3d object detection](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.html)|voxel| CVPR 2018| Zhou et al.
Voxel based |Second: [Sparsely embedded convolutional detection](https://www.mdpi.com/1424-8220/18/10/3337)|voxel| Sensors 2018| Yan et al.
Voxel based |PointPillars: [Fast Encoders for Object Detection From Point Clouds](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html)|voxel| CVPR 2019| Lang et al.
Voxel based |HVNet: [Hybrid Voxel Network for LiDAR Based 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Ye_HVNet_Hybrid_Voxel_Network_for_LiDAR_Based_3D_Object_Detection_CVPR_2020_paper.html)|voxel| CVPR 2020| Ye et al.
Voxel based |HVPR: [Hybrid Voxel-Point Representation for Single-Stage 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Noh_HVPR_Hybrid_Voxel-Point_Representation_for_Single-Stage_3D_Object_Detection_CVPR_2021_paper.html)|voxel| CVPR 2021| Noh et al.
Voxel based | SA-SSD : [Structure aware single-stage 3d object detection from point cloud](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.html)|voxel|CVPR 2020 | He et al.
Point based | PointRCNN: [3D Object Proposal Generation and Detection From Point Cloud](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_PointRCNN_3D_Object_Proposal_Generation_and_Detection_From_Point_Cloud_CVPR_2019_paper.html)| point| CVPR 2019| Shi et al.
Point based | VoteNet :[A Deep Learning Label Fusion Method for Multi-atlas Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_23) | point| ICCV 2019| Ding et al.
Point based | Part A^2 :[From Points to Parts: 3D Object Detection From Point Cloud With Part-Aware and Part-Aggregation Network](https://ieeexplore.ieee.org/abstract/document/9018080)| point| TPAMI2020| Shi et al.
Point based | PV RCNN : [Point-Voxel Feature Set Abstraction for 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.html)| point| CVPR 2020| Shi et al.
Point based | 3DSSD :[Point-based 3D Single Stage Object Detector](https://ieeexplore.ieee.org/document/9156597)| point| CVPR 2020| Yang et al.
Point based | LiDAR RCNN :[An Efficient and Universal 3D Object Detector](https://openaccess.thecvf.com/content/CVPR2021/html/Li_LiDAR_R-CNN_An_Efficient_and_Universal_3D_Object_Detector_CVPR_2021_paper.html)| point| CVPR 2021| Li et al.
Point based | 3DIoUMatch :[Leveraging IoU Prediction for Semi-Supervised 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_3DIoUMatch_Leveraging_IoU_Prediction_for_Semi-Supervised_3D_Object_Detection_CVPR_2021_paper.html)| point| CVPR 2021| Wang et al.
Point based | ST3D :[Self-Training for Unsupervised Domain Adaptation on 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_ST3D_Self-Training_for_Unsupervised_Domain_Adaptation_on_3D_Object_Detection_CVPR_2021_paper.html)| point| CVPR 2021| Yang et al.


## multimodel object detection


![算法时间序列 drawio](https://user-images.githubusercontent.com/44192081/157609652-da342339-bb40-4314-bab4-52f3deb2aee4.png)



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
[Shape Matching and Object Recognition Using Shape Contexts](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf)|2002|Belongie et al.
[3D Shape Descriptor for Objects Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8215285)|LARS and SBR 2017| Sales et al.
ROI-cloud: [A Key Region Extraction Method for LiDAR Odometry and Localization](https://ieeexplore.ieee.org/abstract/document/9197059) | ICRA 2020 | Zhou et al.
PointSIFT: [A sift-like network module for 3D point cloud semantic segmentation](https://arxiv.org/abs/1807.00652) | CVPR 2018 | Jiang et al.

## DataSet
DataSet | Size | Categories / Remarks |Sensing Modalities
--------- | ------------- | ------------- | -------------
[ScanNet](http://www.scan-net.org/) |1513 scans 2.5M frames |floor, wall, chair, cabinet, bed, sofa, table, door, window, bookself, picture, counter, desk, curtain, refrigerator, shower curtain, toilet, sink, bathtub, other furniture| 3D comera,deep Sensors
[SUN RGB-D](http://rgbd.cs.princeton.edu)
[SUN3D](http://sun3d.cs.princeton.edu)
[KITTI](http://www.cvlibs.net/datasets/kitti/)|  7481 frames (training) 80.256 objects| Car, Van, Truck, Pedestrian, Person (sitting), Cyclist, Tram,Misc| Visual (Stereo) camera, 3D LiDAR, GNSS, and inertial sensors
[nuScense](https://www.nuscenes.org/download) |1000 scenes, 1.4M frames (camera, Radar), 390k frames (3D LiDAR)| 25 Object classes, such as Car /Van / SUV, different Trucks,Buses, Persons, Animal, Traffic Cone, Temporary Traffic Barrier, Debris, etc.|Visual cameras (6), 3D LiDAR, and Radars (5)|
[BLVD](https://github.com/VCCIV/BLVD/) | 120k frames, 249,129 objects |Vehicle, Pedestrian, Rider during day and night |  Visual (Stereo) camera, 3D LiDAR
[Waymo open dataset](https://waymo.com/open)|  200k frames, 12M objects (3D LiDAR), 1.2M objects (2D camera)| Vehicles, Pedestrians, Cyclists,Signs|3D LiDAR (5), Visual cameras (5)
[H3D](https://usa.honda-ri.com/hdd/introduction/h3d)|27,721 frames, 1,071,302 objects|Car, Pedestrian, Cyclist, Truck, Misc, Animals, Motorcyclist, Bus| Visual cameras (3), 3D LiDAR
[Lyft-L5 AV dataset](https://level5.lyft.com/dataset/)| 55k frames| Semantic HD map included|3D LiDAR (5), Visual cameras (6)
[A2D2](https://www.audi-electronics-venture.de/aev/web/en/driving-dataset.html)|40k frames (semantics), 12k frames (3D objects), 390k frames unlabeled|Car,Bicycle, Pedestrian, Truck,Small vehicles, Traffic signal,Utility vehicle, Sidebars, Speed bumper, Curbstone, Solid line,Irrelevant signs, Road blocks, Tractor, Non-drivable street, Zebra crossing, Obstacles / trash, Poles,RD restricted area, Animals, Grid structure, Signal corpus, Drivable cobbleston, Electronic traffic,Slow drive area, Nature object,Parking area, Sidewalk, Ego car,Painted driv. instr., Traffic guide obj., Dashed line, RD normal street, Sky, Buildings, Blurred area, Rain dirt| Visual cameras (6); 3D LiDAR (5); Bus data
[ApolloScape](http://apolloscape.auto/scene.html)|143,906 image frames, 89,430 objects|Rover, Sky, Car, Motobicycle,Bicycle, Person, Rider, Truck,Bus, Tricycle, Road, Sidewalk,Traffic Cone, Road Pile, Fence,Traffic Light, Pole, Traffic Sign,Wall, Dustbin, Billboard,Building, Bridge, Tunnel,Overpass, Vegetation|Visual (Stereo) camera, 3D LiDAR, GNSS, and inertial sensors
[A**3D** Dataset](https://github.com/I2RDL2/ASTAR-3D)| 39k frames, 230k objects|Car, Van, Bus, Truck, Pedestrians,Cyclists, and Motorcyclists;Afternoon and night, wet and dry|Visual cameras (2); 3D LiDAR
[DBNet Dataset](http://www.dbehavior.net/)|Over 10k frames|In total seven datasets with different test scenarios, such as seaside roads, school areas,mountain roads.|3D LiDAR, Dashboard visual camera, GNSS
[KAIST multispectral dataset](http://multispectral.kaist.ac.kr)|7,512 frames, 308,913 objects|Person, Cyclist, Car during day and night, fine time slots (sunrise,afternoon,...)
[PandaSet](https://scale.com/open-datasets/pandaset)
**TO BE CONTINUE!**
