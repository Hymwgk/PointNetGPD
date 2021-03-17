#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description:
# Date       : 05/08/2018 6:04 PM
# File Name  : kinect2grasp.py

import torch

import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf
import moveit_commander
import numpy as np
#自定义pointcloud包
import pointclouds
#from pcl import PointCloud
#自定义
import voxelgrid


import pcl
#容易报错无法导入ruamel.yaml，需要使用命令  conda install ruamel.yaml 来安装，不能使用pip安装
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper
from dexnet.grasping import GpgGraspSamplerPcl
import os
from pyquaternion import Quaternion
import sys
from os import path
import time
from scipy.stats import mode
import multiprocessing as mp
try:
    from gpd_grasp_msgs.msg import GraspConfig
    from gpd_grasp_msgs.msg import GraspConfigList
except ImportError:
    print("Please install grasp msgs from https://github.com/TAMS-Group/gpd_grasp_msgs in your ROS workspace")
    exit()

try:
    from mayavi import mlab
except ImportError:
    print("Can not import mayavi")
    mlab = None
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath("__file__")))))
sys.path.append(os.environ['HOME'] + "/code/PointNetGPD/PointNetGPD")

#把读取网络模型，外部命令参数的指令写在了main_test中
from main_test import test_network, model, args

# global config:全局的配置文件
yaml_config = YamlConfig(os.environ['HOME'] + "/code/PointNetGPD/dex-net/test/config.yaml")
gripper_name = 'panda'
#加载夹爪
gripper = RobotGripper.load(gripper_name, os.environ['HOME'] + "/code/PointNetGPD/dex-net/data/grippers")
ags = GpgGraspSamplerPcl(gripper, yaml_config)

value_fc = 0.4  # no use, set a random number
num_grasps_single_worker = 20
#如果使用多线程，每个线程采样num_grasps_p_worker个抓取
num_grasps_p_worker=6
#如果使用多线程，将一共使用num_workers个线程
num_workers = 20
max_num_samples = 150
n_voxel = 500

#输入pointnet的最小点数
minimal_points_send_to_point_net = 20
marker_life_time = 8

show_bad_grasp = False
save_grasp_related_file = False

using_mp = args.using_mp
show_final_grasp = args.show_final_grasp


tray_grasp = args.tray_grasp
single_obj_testing = False  # if True, it will wait for input before get pointcloud

#指定输入点云的点数 number of points put into neural network
if args.model_type == "500":  # minimal points send for training
    input_points_num = 500
elif args.model_type == "750":
    input_points_num = 750
elif args.model_type == "3class":
    input_points_num = 500
else:
    input_points_num = 0

#去除支撑桌面点
def remove_table_points(points_voxel_, vis=False):
    xy_unique = np.unique(points_voxel_[:, 0:2], axis=0)
    new_points_voxel_ = points_voxel_
    pre_del = np.zeros([1])
    for i in range(len(xy_unique)):
        tmp = []
        for j in range(len(points_voxel_)):
            if np.array_equal(points_voxel_[j, 0:2], xy_unique[i]):
                tmp.append(j)
        print(len(tmp))
        if len(tmp) < 3:
            tmp = np.array(tmp)
            pre_del = np.hstack([pre_del, tmp])
    if len(pre_del) != 1:
        pre_del = pre_del[1:]
        new_points_voxel_ = np.delete(points_voxel_, pre_del, 0)
    print("Success delete [[ {} ]] points from the table!".format(len(points_voxel_) - len(new_points_voxel_)))

    if vis:
        p = points_voxel_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(1, 0, 0))
        p = new_points_voxel_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(0, 0, 1))
        mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))  # plot 0 point
        mlab.show()
    return new_points_voxel_


def remove_white_pixel(msg, points_, vis=False):
    points_with_c_ = pointclouds.pointcloud2_to_array(msg)
    points_with_c_ = pointclouds.split_rgb_field(points_with_c_)
    r = np.asarray(points_with_c_['r'], dtype=np.uint32)
    g = np.asarray(points_with_c_['g'], dtype=np.uint32)
    b = np.asarray(points_with_c_['b'], dtype=np.uint32)
    rgb_colors = np.vstack([r, g, b]).T
    # rgb = rgb_colors.astype(np.float) / 255
    ind_good_points_ = np.sum(rgb_colors[:] < 210, axis=-1) == 3
    ind_good_points_ = np.where(ind_good_points_ == 1)[0]
    new_points_ = points_[ind_good_points_]
    if vis:
        p = points_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(1, 0, 0))
        p = new_points_
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(0, 0, 1))
        mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))  # plot 0 point
        mlab.show()
    return new_points_


def get_voxel_fun(points_, n):
    get_voxel = voxelgrid.VoxelGrid(points_, n_x=n, n_y=n, n_z=n)
    get_voxel.compute()
    points_voxel_ = get_voxel.voxel_centers[get_voxel.voxel_n]
    points_voxel_ = np.unique(points_voxel_, axis=0)
    return points_voxel_


def cal_grasp(msg, cam_pos_):
    """根据在线采集的点云计算候选的抓取姿态
    """
    #把pointcloud2类型的消息点云，转换为ndarray  points_
    points_ = pointclouds.pointcloud2_to_xyz_array(msg)
    #复制一份points_ ndarray对象，并将所有的点坐标转换为float32类型
    points_ = points_.astype(np.float32)

    remove_white = False
    if remove_white:
        points_ = remove_white_pixel(msg, points_, vis=True)
    # begin voxel points
    n = n_voxel  # parameter related to voxel method
    # gpg improvements, highlights: flexible n parameter for voxelizing.
    #这一句话执行的时候，不能打开虚拟机，否则容易卡住
    points_voxel_ = get_voxel_fun(points_, n)

    #当点云点数小于2000时
    if len(points_) < 2000:  # should be a parameter
        while len(points_voxel_) < len(points_)-15:
            points_voxel_ = get_voxel_fun(points_, n)
            n = n + 100
            rospy.loginfo("the voxel has {} points, we want get {} points".format(len(points_voxel_), len(points_)))

    rospy.loginfo("the voxel has {} points, we want get {} points".format(len(points_voxel_), len(points_)))
    #
    points_ = points_voxel_
    remove_points = False
    #是否剔除支撑平面
    if remove_points:
        points_ = remove_table_points(points_, vis=True)
    #重新构造经过“降采样”的点云
    point_cloud = pcl.PointCloud(points_)

    print(len(points_))
    #构造法向量估计对象
    norm = point_cloud.make_NormalEstimation()
    tree=point_cloud.make_kdtree()
    norm.set_SearchMethod(tree)
    #以周边30个点作为法向量计算点
    norm.set_KSearch(10)  # critical parameter when calculating the norms
    normals = norm.compute()


    #将点云法向量转换为ndarry类型
    surface_normal = normals.to_array()

    surface_normal = surface_normal[:, 0:3]

    #每个点到  相机位置（无姿态）的向量             但是，感觉是相机到点的向量
    vector_p2cam = cam_pos_ - points_
    #print(vector_p2cam)
    #print(cam_pos_)

    """
    np.linalg.norm(vector_p2cam, axis=1) 默认求2范数，axis=1  代表按行向量处理，求多个行向量的2范数（求模）
    np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)  将其调整为m行 1列

    整句话的含义是，将vector_p2cam归一化，单位化
    """
    vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(-1, 1)
    


    #将表面法相与表面法相（都是单位向量）点乘，以备后面计算向量夹角
    tmp = np.dot(vector_p2cam, surface_normal.T).diagonal()
    #print(vector_p2cam)
    #print(surface_normal.T)
    #print(tmp)

    """
    np.clip(tmp, -1.0, 1.0)  截取函数，将tmp中的值，都限制在-1.0到1.0之间，大于1的变成1，小于-1的记为-1
    np.arccos() 求解反余弦，求夹角
    """
    angel = np.arccos(np.clip(tmp, -1.0, 1.0))
    #print(angel)

    #找到与视角向量夹角大于90度的角（认为法向量计算错误）
    wrong_dir_norm = np.where(angel > np.pi * 0.5)[0]
    #print(np.where(angel > np.pi * 0.5))
    #print(wrong_dir_norm)
    #print(len(wrong_dir_norm))

    #创建一个len(angel)行，3列的ndarry对象
    tmp = np.ones([len(angel), 3])
    #将法向量错误的行的元素都改写为-1
    tmp[wrong_dir_norm, :] = -1
    #与表面法相元素对元素相乘，作用是将"错误的"法向量的方向   扭转过来
    surface_normal = surface_normal * tmp
    #选取桌子以上2cm处的点作为检测点
    select_point_above_table = 0.020
    #modify of gpg: make it as a parameter. avoid select points near the table.
    #查看每个点的z方向，如果它们的点z轴方向的值大于select_point_above_table，就把他们抽出来
    points_for_sample = points_[np.where(points_[:, 2] > select_point_above_table)[0] ]
    print(len(points_for_sample))
    if len(points_for_sample) == 0:
        rospy.loginfo("Can not seltect point, maybe the point cloud is too low?")
        return [], points_, surface_normal
    yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc
    
    grasps_together_ = []

    if rospy.get_param("/robot_at_home") == "false":
        robot_at_home = False
    else:
        robot_at_home = True

    if not robot_at_home:
        rospy.loginfo("Robot is moving, waiting the robot go home.")
    elif not using_mp:
        rospy.loginfo("Begin cal grasps using single thread, slow!")
        grasps_together_ = ags.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps_single_worker,
                                             max_num_samples=max_num_samples, show_final_grasp=show_final_grasp)
    else:
        # begin parallel grasp:
        rospy.loginfo("Begin cal grasps using parallel!")

        def grasp_task(num_grasps_, ags_, queue_):
            ret = ags_.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps_,
                                     max_num_samples=max_num_samples, show_final_grasp=False)
            queue_.put(ret)

        queue = mp.Queue()

        #num_grasps_p_worker = int(num_grasps/num_workers)
        workers = [mp.Process(target=grasp_task, args=(num_grasps_p_worker, ags, queue)) for _ in range(num_workers)]
        [i.start() for i in workers]


        grasps_together_ = []
        for i in range(num_workers):
            grasps_together_ = grasps_together_ + queue.get()
        rospy.loginfo("Finish mp processing!")

        if show_final_grasp and using_mp:
            ags.show_all_grasps(points_, grasps_together_)
            ags.show_points(points_, scale_factor=0.002)
            mlab.show()


    rospy.loginfo("Grasp sampler finish, generated {} grasps.".format(len(grasps_together_)))
    #返回抓取     场景的点      以及点云的表面法向量
    return grasps_together_, points_, surface_normal


#检查碰撞
def check_collision_square(grasp_bottom_center, approach_normal, binormal,
                           minor_pc, points_, p, way="p_open"):
    #抓取坐标系 轴单位化
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)
    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    #构建旋转矩阵
    matrix_ = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
    #正交矩阵的逆，就是它的转置
    grasp_matrix = matrix_.T
    #center=grasp_bottom_center+approach_normal*ags.gripper.hand_depth
    points_ = points_ - grasp_bottom_center.reshape(1, 3)
    #points_ = points_ - center.reshape(1, 3)

    tmp = np.dot(grasp_matrix, points_.T)
    points_g = tmp.T

    #选择是否使用与数据集采样相同的方式来采集点云
    use_dataset_py = False
    if not use_dataset_py:
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
    # for the way of pointGPD/dataset.py:
    else:
        width = ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width
        x_limit = ags.gripper.hand_depth
        z_limit = width / 4
        y_limit = width / 2
        x1 = points_g[:, 0] > 0
        x2 = points_g[:, 0] < x_limit
        y1 = points_g[:, 1] > -y_limit
        y2 = points_g[:, 1] < y_limit
        z1 = points_g[:, 2] > -z_limit
        z2 = points_g[:, 2] < z_limit
        a = np.vstack([x1, x2, y1, y2, z1, z2])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

    vis = False
    if vis:
        p = points_g
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(0, 0, 1))
        p = points_g[points_in_area]
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.002, color=(1, 0, 0))
        p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        mlab.points3d(p[:, 0], p[:, 1], p[:, 2], scale_factor=0.005, color=(0, 1, 0))
        mlab.show()

    return has_p, points_in_area, points_g


def collect_pc(grasp_, pc):
    """
    grasp_bottom_center, normal, major_pc, minor_pc
    grasp_是一个list形式的数据
    """
    #获取抓取的数量
    grasp_num = len(grasp_)
    #将抓取转换为ndarry的形式，其实本身也就是
    grasp_ = np.array(grasp_)
    #print(grasp_)

    #变成n个，5行3列的 矩阵, 每个矩阵块，都是一个候选抓取坐标系
    grasp_ = grasp_.reshape(-1, 5, 3)  # prevent to have grasp that only have number 1
    #print(grasp_)

    #使用每个块的第五行数据，作为抓取中心点，因为，第5个是修正后的夹爪姿态中心
    grasp_bottom_center = grasp_[:,4, :]
    approach_normal = grasp_[:, 1,:]
    binormal = grasp_[:, 2,:]
    minor_pc = grasp_[:, 3,:]

    in_ind_ = []
    in_ind_points_ = []
    #获取夹爪角点
    p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

    for i_ in range(grasp_num):
        #通过实验发现，这个函数并不是只返回了夹爪open区域内部的点，而是"扩大"了一些，
        # 估计是为了返回足够多的点
        #但是，问题是这需要与训练时候的样本点相对应啊，如果训练的时候，就没有"扩大"，那网络的打分会高么？
        #所以需要去看训练部分的采样部分两点
        # 1.生成的样本点云是否是以固定夹爪为参考系的？
        # 2.训练时，采样点的提取部分，是否进行了"扩大"?
        has_p, in_ind_tmp, points_g = check_collision_square(grasp_bottom_center[i_], approach_normal[i_],
                                                             binormal[i_], minor_pc[i_], pc, p)
        #把索引添加进list中保存
        in_ind_.append(in_ind_tmp)
        #从下面这句话可以看出
        # 这些返回的夹爪内部点，是已经旋转过后的点，而不是原始的点，为什么？
        #是因为，网络输入的点要求是以固定夹爪为参考系的么？
        #猜测是的，为什么？因为，这样子的话，就可以在传入网络点云的同时，
        # 相当于也传入了夹爪的姿态



        #这一点，他想把grasp_g中的点坐标也都保存下来，in_ind_points_是一个list
        in_ind_points_.append(points_g[in_ind_tmp])
        
        
        #显示出截取的夹爪内部区域的点云（红色）&夹爪&整体点云
        if 0:
            #p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
            ags.show_grasp_3d(p)
            ags.show_points(points_g)
            ags.show_points(in_ind_points_[i_],color='r',scale_factor=0.005)
            #table_points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * 0.5
            #triangles = [(1, 2, 3), (0, 1, 3)]
            #mlab.triangular_mesh(table_points[:, 0], table_points[:, 1], table_points[:, 2],
            #triangles, color=(0.8, 0.8, 0.8), opacity=0.5)
            mlab.show()

    return in_ind_, in_ind_points_


def show_marker(marker_array_, pos_, ori_, scale_, color_, lifetime_):
    """显示标注物体
    """
    marker_ = Marker()
    marker_.header.frame_id = "/ar_marker_6"
    # marker_.header.stamp = rospy.Time.now()
    marker_.type = marker_.CUBE
    marker_.action = marker_.ADD

    marker_.pose.position.x = pos_[0]
    marker_.pose.position.y = pos_[1]
    marker_.pose.position.z = pos_[2]
    marker_.pose.orientation.x = ori_[1]
    marker_.pose.orientation.y = ori_[2]
    marker_.pose.orientation.z = ori_[3]
    marker_.pose.orientation.w = ori_[0]

    marker_.lifetime = rospy.Duration.from_sec(lifetime_)
    marker_.scale.x = scale_[0]
    marker_.scale.y = scale_[1]
    marker_.scale.z = scale_[2]
    marker_.color.a = 0.5
    red_, green_, blue_ = color_
    marker_.color.r = red_
    marker_.color.g = green_
    marker_.color.b = blue_
    marker_array_.markers.append(marker_)


def show_grasp_marker(marker_array_, real_grasp_, gripper_, color_, lifetime_):
    """
    show grasp using marker使用marker来显示抓取
    :param marker_array_: marker array
    :param real_grasp_: [0] position, [1] approach [2] binormal [3] minor pc
    :param gripper_: gripper parameter of a grasp
    :param color_: color of the gripper 显示夹爪的颜色
    :param lifetime_: time for showing the maker  marker的显示时间长短
    :return: return add makers to the maker array    

    """
    hh = gripper_.hand_height
    fw = gripper_.real_finger_width
    hod = gripper_.hand_outer_diameter
    hd = gripper_.real_hand_depth
    open_w = hod - fw * 2

    approach = real_grasp_[1]
    binormal = real_grasp_[2]
    minor_pc = real_grasp_[3]
    grasp_bottom_center = real_grasp_[4] - approach * (gripper_.real_hand_depth - gripper_.hand_depth)

    rotation = np.vstack([approach, binormal, minor_pc]).T
    qua = Quaternion(matrix=rotation)

    marker_bottom_pos = grasp_bottom_center - approach * hh * 0.5
    marker_left_pos = grasp_bottom_center - binormal * (open_w * 0.5 + fw * 0.5) + hd * 0.5 * approach
    marker_right_pos = grasp_bottom_center + binormal * (open_w * 0.5 + fw * 0.5) + hd * 0.5 * approach
    show_marker(marker_array_, marker_bottom_pos, qua, np.array([hh, hod, hh]), color_, lifetime_)
    show_marker(marker_array_, marker_left_pos, qua, np.array([hd, fw, hh]), color_, lifetime_)
    show_marker(marker_array_, marker_right_pos, qua, np.array([hd, fw, hh]), color_, lifetime_)


def check_hand_points_fun(real_grasp_):
    """该函数是计算处于夹爪内部有多少个点（只求数量）
    """
    ind_points_num = []
    for i in range(len(real_grasp_)):
        #修正后的抓取中心点坐标（位于第5个）
        grasp_bottom_center = real_grasp_[i][4]
        approach_normal = real_grasp_[i][1]
        binormal = real_grasp_[i][2]
        minor_pc = real_grasp_[i][3]
        #固定了手，而去变换点云，原因是，这样更容易计算碰撞，可以查看ags.check_collision_square函数内部
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        #检查open区域内的点数量，返回点的索引list
        has_points_tmp, ind_points_tmp = ags.check_collision_square(grasp_bottom_center, approach_normal,
                                                                    binormal, minor_pc, points,
                                                                    local_hand_points, "p_open")
        #打印list中的点数
        ind_points_num.append(len(ind_points_tmp))
    print(ind_points_num)
    #这是啥意思？
    file_name = "./generated_grasps/real_points/" + str(np.random.randint(300)) + str(len(real_grasp_)) + ".npy"
    np.save(file_name, np.array(ind_points_num))


def get_grasp_msg(real_good_grasp_, score_value_):
    """
    创建一个ROS官方的抓取配置消息
    里面涵盖了抓取坐标系的位置与姿态定义
    """
    grasp_bottom_center_modify = real_good_grasp_[4]
    approach = real_good_grasp_[1]
    binormal = real_good_grasp_[2]
    minor_pc = real_good_grasp_[3]

    grasp_config_ = GraspConfig()
    #
    top_p_ = grasp_bottom_center_modify + approach * ags.gripper.hand_depth   
    grasp_config_.bottom.x = grasp_bottom_center_modify[0]
    grasp_config_.bottom.y = grasp_bottom_center_modify[1]
    grasp_config_.bottom.z = grasp_bottom_center_modify[2]
    grasp_config_.top.x = top_p_[0]
    grasp_config_.top.y = top_p_[1]
    grasp_config_.top.z = top_p_[2]
    #抓取的三个坐标轴向量
    grasp_config_.approach.x = approach[0]
    grasp_config_.approach.y = approach[1]
    grasp_config_.approach.z = approach[2]
    grasp_config_.binormal.x = binormal[0]
    grasp_config_.binormal.y = binormal[1]
    grasp_config_.binormal.z = binormal[2]
    grasp_config_.axis.x = minor_pc[0]
    grasp_config_.axis.y = minor_pc[1]
    grasp_config_.axis.z = minor_pc[2]
    #该抓取的分数
    grasp_config_.score.data = score_value_

    return grasp_config_


def remove_grasp_outside_tray(grasps_, points_):
    x_min = points_[:, 0].min()
    x_max = points_[:, 0].max()
    y_min = points_[:, 1].min()
    y_max = points_[:, 1].max()
    valid_grasp_ind_ = []
    for i in range(len(grasps_)):
        grasp_bottom_center = grasps_[i][4]
        approach_normal = grasps_[i][1]
        major_pc = grasps_[i][2]
        hand_points_ = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
        finger_points_ = hand_points_[[1, 2, 3, 4, 9, 10, 13, 14], :]
        # aa = points_[:, :2] - finger_points_[0][:2]  # todo： work of remove outside grasp not finished.

        # from IPython import embed;embed()
        a = finger_points_[:, 0] < x_min
        b = finger_points_[:, 0] > x_max
        c = finger_points_[:, 1] < y_min
        d = finger_points_[:, 1] > y_max
        if np.sum(a) + np.sum(b) + np.sum(c) + np.sum(d) == 0:
            valid_grasp_ind_.append(i)
    grasps_inside_ = [grasps_[i] for i in valid_grasp_ind_]
    rospy.loginfo("gpg got {} grasps, after remove grasp outside tray, {} grasps left".format(len(grasps_),
                                                                                              len(grasps_inside_)))
    return grasps_inside_


if __name__ == '__main__':
    """
    definition of gotten grasps:

    grasp_bottom_center = grasp_[0]
    approach_normal = grasp_[1]
    binormal = grasp_[2]
    """
    #初始化节点，把节点名称写为grasp_tf_broadcaster  抓取发布器，anonymous参数在
    # 为True的时候会在原本节点名字的后面加一串随机数，来保证可以同时开启多个同样的
    # 节点，如果为false的话就只能开一个
    rospy.init_node('grasp_tf_broadcaster', anonymous=True)
    #创建发布器，用于显示抓取
    pub1 = rospy.Publisher('gripper_vis', MarkerArray, queue_size=1)
    #发布检测到的抓取，用于抓取
    pub2 = rospy.Publisher('/detect_grasps/clustered_grasps', GraspConfigList, queue_size=1)
    
    #
    rate = rospy.Rate(10)
    #在ros参数服务器上，设置一个参数
    rospy.set_param("/robot_at_home", "true")  # only use when in simulation test.
    rospy.loginfo("getting transform from kinect2 to table top")

    #cam_pos列表
    cam_pos = []
    #创建TF监听器
    listener = tf.TransformListener()
    #是否得到变换的标志位
    get_transform = False
    #等待tf中查找到'/table_top'与'/kinect2_ir_optical_frame'
    # 两个坐标系之间的变换关系
    while not get_transform:
        try:
            #查看kinect2相机与桌子之间的关系，确保相机已经能够看到标签ar_marker_6
            #cam_pos代表的是相机的trans，不是rot
            cam_pos, _ = listener.lookupTransform('/ar_marker_6', '/kinect2_rgb_optical_frame', rospy.Time(0))
            print(cam_pos)
            print(_)
            get_transform = True
            rospy.loginfo("got transform complete")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue


    while not rospy.is_shutdown():
        #检测机械臂是否处于home状态
        if rospy.get_param("/robot_at_home") == "false":
            robot_at_home = False
        else:
            robot_at_home = True
        #如果机械臂状态不在home（移动中）
        if not robot_at_home:
            rospy.loginfo("Robot is moving, waiting the robot go home.")
            #就跳过后面一直循环等待移动完毕
            continue
        else:
            #在home状态，可采集点云
            rospy.loginfo("Robot is at home, safely catching point cloud data.")
            if single_obj_testing:
                input("Pleas put object on table and press any number to continue!")
       
        rospy.loginfo("rospy is waiting for message: /table_top_points")
        
        """点云数据的名称是/table_top_points
        对象是kinect_data   类形是 PointCloud2类型
        """
        #kinect_data = rospy.wait_for_message("/table_top_points", PointCloud2)
        kinect_data_ = rospy.wait_for_message("/table_top_points", PointCloud2)
        kinect_data = rospy.wait_for_message("/table_top_points_subsampled", PointCloud2)

        real_good_grasp = []
        real_bad_grasp = []
        real_score_value = []
        
        
        #设置每个候选抓取的夹爪内部的点
        # 需要送入pointnet中训练几次
        repeat = 3  # speed up this try 10 time is too time consuming
        ###########################开始抓取姿态检测###############################
        # begin of grasp detection 开始抓取检测
        # if there is no point cloud on table, waiting for point cloud.
        if kinect_data.data == '':
            rospy.loginfo("There is no points on the table, waiting...")
            continue


        #获取当前文件所在目录（文件夹）的绝对路径
        path=os.path.dirname(os.path.abspath(__file__))
        #更改（确保）当前所在目录是工作目录
        os.chdir(path)

        """根据Kinect读取到的场景点云，使用gpd检测候选的抓取？
        输入：
        kinect_data读取的点云数据
        cam_pos    Kinect与桌子标签之间的距离
        grasp_sampled   生成的所有候选抓取（不与桌子以及场景物体碰撞的抓取）
        points     函数中处理后，用于计算抓取的点云，ndarry形式
        normals_cal   points对应的所有法相量 ndarry形式
        """
        grasp_sampled, points, normals_cal = cal_grasp(kinect_data, cam_pos)
        #检查生成的抓取数量是否为0
        if len(grasp_sampled)==0:
            rospy.loginfo("Nice try!")
            continue


        #托盘，如果有托盘？
        if tray_grasp:
            #grasp_sampled 去除外部托盘导致的抓取；
            grasp_sampled = remove_grasp_outside_tray(grasp_sampled, points)
        #估计一个抓取中的点数
        check_grasp_points_num = False  # evaluate the number of points in a grasp
        
        """
        等效于
        if check_grasp_points_num:
            check_hand_points_fun(grasp_sampled)
        else:
            0
        """
        check_hand_points_fun(grasp_sampled) if check_grasp_points_num else 0

        #计算，每个抓取，夹爪内部的点云
        in_ind, in_ind_points= collect_pc(grasp_sampled, points)


        #保存在线抓取抓取相关的文件
        if save_grasp_related_file:
            np.save("./generated_grasps/points.npy", points)
            np.save("./generated_grasps/in_ind.npy", in_ind)
            np.save("./generated_grasps/grasp_sampled.npy", grasp_sampled)
            np.save("./generated_grasps/cal_norm.npy", normals_cal)
        #打分
        score = []  # should be 0 or 1
        score_value = []  # should be float [0, 1]

        ind_good_grasp = []
        ind_bad_grasp = []
        ############################把检测到的抓取送到pointnet中打分####################################
        rospy.loginfo("Begin send grasp into pointnet, cal grasp score")
        
        #in_ind_points 是一个 list
        for ii in range(len(in_ind_points)):
            """
            首先需要保证机械臂处于home状态
            """
            if rospy.get_param("/robot_at_home") == "false":
                robot_at_home = False
            else:
                robot_at_home = True
            if not robot_at_home:
                rospy.loginfo("robot is not at home, stop calculating the grasp score")
                break
            """
            判断，夹爪内部的点数量，是否满足最小点数（20），如果小于20的话就别扩充啥的了
            太少了，可能就只是噪点抓取而已
            """
            if in_ind_points[ii].shape[0] < minimal_points_send_to_point_net:
                rospy.loginfo("Mark as bad grasp! Only {} points, should be at least {} points.".format(
                              in_ind_points[ii].shape[0], minimal_points_send_to_point_net))
                #第一个分数记为0
                score.append(0)
                score_value.append(0.0)
                #如果需要记下来，哪个是bad抓取的话，就把哪个抓取给记下来
                if show_bad_grasp:
                    ind_bad_grasp.append(ii)
            
            #内部点数满足要求
            else:
                predict = []
                grasp_score = []
                """for _ in range()            其中_代表临时变量，说明只关心完成循环而已，不关心循环到第几个
                给定一个抓取，找到夹爪内部的点，然后重复repeat次输入pointnet中，进行多次打分
                因为下面的采集点具有部分随机性，所以需要重复输入几次，但是稍微比较费时
                """
                for _ in range(repeat):
                    #如果在线采集的夹爪内部点云点数量是大于  指定的点数量
                    if len(in_ind_points[ii]) >= input_points_num:
                        #随机抽选其中的一些点，保证和要求的点数量是一致的
                        points_modify = in_ind_points[ii][np.random.choice(len(in_ind_points[ii]),
                                                                           input_points_num, replace=False)]
                    #如果在线采集的夹爪内部点云点数量是小于  指定的点数量
                    else:
                        #就是补上去一些没有用的点，但是怎么补的，没看清楚？？？？？？？？？？？？？？？？？？
                        points_modify = in_ind_points[ii][np.random.choice(len(in_ind_points[ii]),
                                                                           input_points_num, replace=True)]
                    if 0:
                        p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
                        ags.show_grasp_3d(p)
                        print("原始点数{},修改后点数{}".format(len(in_ind_points[ii]),len(points_modify)))
                        ags.show_points(points_modify,color='b',scale_factor=0.005)
                        ags.show_points(in_ind_points[ii],color='r',scale_factor=0.003)
                        #table_points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * 0.5
                        #triangles = [(1, 2, 3), (0, 1, 3)]
                        #mlab.triangular_mesh(table_points[:, 0], table_points[:, 1], table_points[:, 2],
                                            #triangles, color=(0.8, 0.8, 0.8), opacity=0.5)
                        mlab.show()
                        
                    
                    
                    """在这里输入网络打分！
                    返回的是？，grasp打分
                    """
                    if_good_grasp, grasp_score_tmp = test_network(model.eval(), points_modify)


                    predict.append(if_good_grasp.item())
                    #保存分数，添加到list中（这个list中都是同一个抓取的分数，重复了几次）
                    grasp_score.append(grasp_score_tmp)

                predict_vote = mode(predict)[0][0]  # vote from all the "repeat" results.
                #把list转换为np.array
                grasp_score = np.array(grasp_score)
                #如果是3class的？？？？？？？？？？？？？？？
                if args.model_type == "3class":  # the best in 3 class classification is the last column, third column
                    which_one_is_best = 2  # should set as 2
                #两分类是第二个
                else:  # for two class classification best is the second column (also the last column)
                    which_one_is_best = 1  # should set as 1

                """np.mean()函数功能：求取均值numpy.mean(a, axis, dtype, out，keepdims )
                        经常操作的参数为axis，以m * n矩阵举例：
                        axis 不设置值，对 m*n 个数求均值，返回一个实数
                        axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
                        axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
                """
                score_vote = np.mean(grasp_score[np.where(predict == predict_vote)][:, 0, which_one_is_best])
                score.append(predict_vote)
                score_value.append(score_vote)

                if score[ii] == which_one_is_best:
                    ind_good_grasp.append(ii)
                else:
                    if show_bad_grasp:
                        #存放“坏抓取”的抓取编号
                        ind_bad_grasp.append(ii)
        #记录好的抓取的数量，以及坏的抓取数量
        print("Got {} good grasps, and {} bad grasps".format(len(ind_good_grasp),
                                                             len(in_ind_points) - len(ind_good_grasp)))
        

        if len(ind_good_grasp) != 0:
            real_good_grasp = [grasp_sampled[i] for i in ind_good_grasp]
            real_score_value = [score_value[i] for i in ind_good_grasp]
            if show_bad_grasp:
                real_bad_grasp = [grasp_sampled[i] for i in ind_bad_grasp]
        # end of grasp detection抓取检测部分结束
        
        
        #real_bad_grasp = [grasp_sampled[i] for i in ind_bad_grasp]


        # get sorted ind by the score values对检测到的抓取通过打分进行排序
        sorted_value_ind = list(index for index, item in sorted(enumerate(real_score_value),
                                                                key=lambda item: item[1],
                                                                reverse=True))
        # sort grasps using the ind
        sorted_real_good_grasp = [real_good_grasp[i] for i in sorted_value_ind]
        real_good_grasp = sorted_real_good_grasp
        # get the sorted score value, from high to low
        real_score_value = sorted(real_score_value, reverse=True)



        marker_array = MarkerArray()
        marker_array_single = MarkerArray()


        grasp_msg_list = GraspConfigList()

        #按照顺序将所有的好的抓取添加显示，应该是第0个是最优的抓取
        for i in range(len(real_good_grasp)):
            grasp_msg = get_grasp_msg(real_good_grasp[i], real_score_value[i])
            grasp_msg_list.grasps.append(grasp_msg)
        #把所有的抓取都显示成为绿色的，都添加进入那个显示序列中
        for i in range(len(real_good_grasp)):
            show_grasp_marker(marker_array, real_good_grasp[i], gripper, (0, 1, 0), marker_life_time)

        if show_bad_grasp:
            for i in range(len(real_bad_grasp)):
                show_grasp_marker(marker_array, real_bad_grasp[i], gripper, (1, 0, 0), marker_life_time)

        id_ = 0
        for m in marker_array.markers:
            m.id = id_
            id_ += 1

        grasp_msg_list.header.stamp = rospy.Time.now()
        grasp_msg_list.header.frame_id = "/ar_marker_6"

        # from IPython import embed;embed()
        if len(real_good_grasp) != 0:
        #if 0:
            i = 0
            #第0个抓取得分是最高的，以红色显示出来
            single_grasp_list_pub = GraspConfigList()
            single_grasp_list_pub.header.stamp = rospy.Time.now()
            single_grasp_list_pub.header.frame_id = "/ar_marker_6"
            grasp_msg = get_grasp_msg(real_good_grasp[i], real_score_value[i])
            single_grasp_list_pub.grasps.append(grasp_msg)
            show_grasp_marker(marker_array_single, real_good_grasp[i], gripper, (1, 0, 0), marker_life_time+5)

            for m in marker_array_single.markers:
                m.id = id_
                id_ += 1
            pub1.publish(marker_array)
            rospy.sleep(4)
            pub2.publish(single_grasp_list_pub)
            pub1.publish(marker_array_single)
            print(grasp_msg.approach)
            print(grasp_msg.binormal)
            print(grasp_msg.axis)

        # pub2.publish(grasp_msg_list)
        rospy.loginfo(" Publishing grasp pose to rviz using marker array and good grasp pose")

        rate.sleep()
