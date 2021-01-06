# PointNetGPD: Detecting Grasp Configurations from Point Sets
PointNetGPD (ICRA 2019, [arXiv](https://arxiv.org/abs/1809.06267)) is an end-to-end grasp evaluation model to address the challenging problem of localizing robot grasp configurations directly from the point cloud.

PointNetGPD is light-weighted and can directly process the 3D point cloud that locates within the gripper for grasp evaluation. Taking the raw point cloud as input, our proposed grasp evaluation network can capture the complex geometric structure of the contact area between the gripper and the object even if the point cloud is very sparse.

To further improve our proposed model, we generate a larger-scale grasp dataset with 350k real point cloud and grasps with the [YCB objects Dataset](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/) for training.

<img src="data/grasp_pipeline.svg" width="100%">

## Video

[![Video for PointNetGPD](https://img.youtube.com/vi/RBFFCLiWhRw/0.jpg )](https://www.youtube.com/watch?v=RBFFCLiWhRw)
## Before Install
在使用前，clone的代码文件夹需要放在如下的code文件夹中:
```
mkdir -p $HOME/code/
cd $HOME/code/
```
## Install all the requirements (Using a virtual environment is recommended)
1. Make sure in your Python environment do not have same package named ```meshpy``` or ```dexnet```.

2. Clone this repository:
    ```bash
    cd $HOME/code
    git clone https://github.com/lianghongzhuo/PointNetGPD.git
    ```

3. Install our requirements in `requirements.txt`
    ```bash
    cd $HOME/code/PointNetGPD
    pip install -r requirements.txt
    ```
4. Install our modified meshpy (Modify from [Berkeley Automation Lab: meshpy](https://github.com/BerkeleyAutomation/meshpy))
    ```bash
    cd $HOME/code/PointNetGPD/meshpy
    python setup.py develop
    ```

5. Install our modified dex-net (Modify from [Berkeley Automation Lab: dex-net](https://github.com/BerkeleyAutomation/dex-net))
    ```bash
    cd $HOME/code/PointNetGPD/dex-net
    python setup.py develop
    ```
6. 这里需要根据自己的夹爪来修改如下的夹爪文件；  
    你可以模仿着新建一个文件夹，也可以直接在这个文件中修改，之后离线与在线节点的夹爪姿态检测都会根据夹爪具体的尺寸来生成。
    ```bash
    vim $HOME/code/PointNetGPD/dex-net/data/grippers/robotiq_85/params.json
    ```
    离线阶段用到的参数，（默认）通过antipod采样法对CAD模型数据集中的模型进行抓取姿态的采样时候用到的参数，在作者的代码中，这部分的参数一共有两点作用：  
    >- 离线生成抓取姿态数据集时，用此参数结合antipod采样法对CAD模型数据集中的模型进行抓取姿态的采样；
    >- 离线训练PointNet时，在dataloader中，使用该参数对提取夹爪内部的部分点云，进而将该部分点云送入网络进行训练；  
  
    **但是**个人在看代码的过程中，发现第二点的作用是不合适的：
    >对PointNet训练和使用时，应该保证离线训练时的点云采样方式尽可能和在线点云采样方式相同，这样网络输出的结果将会更好；  
    如果希望提取某姿态下夹爪内部点云，就需要根据给定的夹爪尺寸对夹爪建立尺寸数学模型；  
    而原始代码中，离线提取夹爪内部点云使用的夹爪数学模型和在线提取点云时使用的夹爪数学模型是不同的，分别使用了两种构建形式，并且在线和离线的夹爪参数不太一致，本修改代码中，仅在抓取姿态数据集生成阶段使用了离线参数；  
    将在线夹爪内部提取时的夹爪数学模型替换为和离线训练PointNet时dataloader中相同的数学模型。


    ```bash
    "min_width":      夹爪的最小闭合角度
    "force_limit":      抓取力度限制
    "max_width":     夹爪最大张开距离
    "finger_radius": 用于软体手，指定软体手的弯曲角度（弧度制），一般用不到，补上去就行了
    "max_depth":     夹爪的最大深度，竖向的距离
    ```
    These parameters are used for grasp pose generation at experiment:
    ```bash
    "finger_width":    夹持器的两个夹爪的“厚度”
    "real_finger_width":   也是两个夹爪的厚度，和上面写一样就行（仅仅用于显示，影响不大，不用于姿态检测）
    "hand_height":   夹爪的另一侧厚度，一会儿看图
    "hand_height_two_finger_side":   没有用到，代码中没有找到调用，所以不用管
    "hand_outer_diameter":  夹爪最大的可以张开的距离，从最外侧量（包括两个爪子的厚度）
    "hand_depth":   夹爪的竖向深度
    "real_hand_depth":   和hand_depth保持一致，代码中两者是相同的
    "init_bite":  这个是用于在线抓取检测时，定义的一个后撤距离，主要是避免由于点云误差之类的，导致夹爪和物体碰撞，以米为单位，一般设置1cm就行了
    ```

## Genearted Grasp Dataset Download
You can download the dataset from: https://tams.informatik.uni-hamburg.de/research/datasets/PointNetGPD_grasps_dataset.zip

## Generate Your Own Grasp Dataset

1. Download YCB object set from [YCB Dataset](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/).
2. Manage your dataset here:
    ```bash
    mkdir -p $HOME/dataset/ycb_meshes_google/objects
    ```
    Every object should have a folder, structure like this:
    ```bash
    ├002_master_chef_can
    |└── google_512k
    |    ├── kinbody.xml (no use)
    |    ├── nontextured.obj
    |    ├── nontextured.ply
    |    ├── nontextured.sdf (generated by SDFGen)
    |    ├── nontextured.stl
    |    ├── textured.dae (no use)
    |    ├── textured.mtl (no use)
    |    ├── textured.obj (no use)
    |    ├── textured.sdf (no use)
    |    └── texture_map.png (no use)
    ├003_cracker_box
    └004_sugar_box
    ...


    ```
3. Install SDFGen from [GitHub](https://github.com/jeffmahler/SDFGen.git):
    ```bash
    git clone https://github.com/jeffmahler/SDFGen.git
    cd SDFGen
    sudo sh install.sh
    ```
4. Install python pcl library [python-pcl](https://github.com/strawlab/python-pcl):
    ```bash
    git clone https://github.com/strawlab/python-pcl.git
    pip install --upgrade pip
    pip install cython
    pip install numpy
    cd python-pcl
    python setup.py build_ext -i
    python setup.py develop
    ```
    - If you use **ubuntu 18.04** and/or **conda environment**, you may encounter a compile error when install python-pcl, this is because conda has a higer version of vtk, here is a work around:
        - `conda install vtk` or `pip install vtk`
        - Use my fork: https://github.com/lianghongzhuo/python-pcl.git
5. Generate sdf file for each nontextured.obj file using SDFGen by running:
    ```bash
    cd $HOME/code/PointNetGPD/dex-net/apps
    python read_file_sdf.py
    ```
6. Generate dataset by running the code:
    ```bash
    cd $HOME/code/PointNetGPD/dex-net/apps
    python generate-dataset-canny.py [prefix]
    ```
    where ```[prefix]``` is the optional, it will add a prefix on the generated files.

## Visualization tools
- Visualization grasps
    ```bash
    cd $HOME/code/PointNetGPD/dex-net/apps
    python read_grasps_from_file.py
    ```
    Note:
    - This file will visualize the grasps in `$HOME/code/PointNetGPD/PointNetGPD/data/ycb_grasp/` folder

- Visualization object normals
    ```bash
    cd $HOME/code/PointNetGPD/dex-net/apps
    python Cal_norm.py
    ```
This code will check the norm calculated by meshpy and pcl library.

## Training the network
1. Data prepare:
    ```bash
    cd $HOME/code/PointNetGPD/PointNetGPD/data
    ```

    Make sure you have the following files, The links to the dataset directory should add by yourself:
    ```
    ├── google2cloud.csv  (Transform from google_ycb model to ycb_rgbd model)
    ├── google2cloud.pkl  (Transform from google_ycb model to ycb_rgbd model)
    ├── ycb_grasp  (generated grasps)
    ├── ycb_meshes_google  (YCB dataset)
    └── ycb_rgbd  (YCB dataset)
    ```

    Generate point cloud from rgb-d image, you may change the number of process running in parallel if you use a shared host with others
    ```bash
    cd ..
    python ycb_cloud_generate.py
    ```
    Note: Estimated running time at our `Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz` dual CPU with 56 Threads is 36 hours. Please also remove objects beyond the capacity of the gripper.

1. Run the experiments:
    ```bash
    cd PointNetGPD
    ```

    Launch a tensorboard for monitoring
    ```bash
    tensorboard --log-dir ./assets/log --port 8080
    ```

    and run an experiment for 200 epoch
    ```
    python main_1v.py --epoch 200 --mode train --batch-size x (x>1)
    ```

    File name and corresponding experiment:
    ```
    main_1v.py        --- 1-viewed point cloud, 2 class
    main_1v_mc.py     --- 1-viewed point cloud, 3 class
    main_1v_gpd.py    --- 1-viewed point cloud, GPD
    main_fullv.py     --- Full point cloud, 2 class
    main_fullv_mc.py  --- Full point cloud, 3 class
    main_fullv_gpd.py --- Full point cloud, GPD
    ```

    For GPD experiments, you may change the input channel number by modifying `input_chann` in the experiment scripts(only 3 and 12 channels are available)

## Using the trained network

1. Get UR5 robot state:

    Goal of this step is to publish a ROS parameter tell the environment whether the UR5 robot is at home position or not.
    ```
    cd $HOME/code/PointNetGPD/dex-net/apps
    python get_ur5_robot_state.py
    ```
2. Run perception code:
    This code will take depth camera ROS info as input, and gives a set of good grasp candidates as output.
    All the input, output messages are using ROS messages.
    ```
    cd $HOME/code/PointNetGPD/dex-net/apps
    python kinect2grasp.py

    arguments:
    -h, --help                 show this help message and exit
    --cuda                     using cuda for get the network result
    --gpu GPU                  set GPU number
    --load-model LOAD_MODEL    set witch model you want to use (rewrite by model_type, do not use this arg)
    --show_final_grasp         show final grasp using mayavi, only for debug, not working on multi processing
    --tray_grasp               not finished grasp type
    --using_mp                 using multi processing to sample grasps
    --model_type MODEL_TYPE    selet a model type from 3 existing models
    ```

## Citation
If you found PointNetGPD useful in your research, please consider citing:

```plain
@inproceedings{liang2019pointnetgpd,
  title={{PointNetGPD}: Detecting Grasp Configurations from Point Sets},
  author={Liang, Hongzhuo and Ma, Xiaojian and Li, Shuang and G{\"o}rner, Michael and Tang, Song and Fang, Bin and Sun, Fuchun and Zhang, Jianwei},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2019}
}
```

## Acknowledgement
- [gpg](https://github.com/atenpas/gpg)
- [gpd](https://github.com/atenpas/gpd)
- [dex-net](https://github.com/BerkeleyAutomation/dex-net)
- [meshpy](https://github.com/BerkeleyAutomation/meshpy)
- [SDFGen](https://github.com/christopherbatty/SDFGen)
- [pyntcloud](https://github.com/daavoo/pyntcloud)
- [metu-ros-pkg](https://github.com/kadiru/metu-ros-pkg)
- [mayavi](https://github.com/enthought/mayavi)

