import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf
import numpy as np
#自定义pointcloud包
import pointclouds
#from pcl import PointCloud
#自定义
import voxelgrid


import pcl
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

# global config:全局的配置文件
yaml_config = YamlConfig(os.environ['HOME'] + "/code/PointNetGPD/dex-net/test/config.yaml")
gripper_name = 'robotiq_85'
#加载夹爪
gripper = RobotGripper.load(gripper_name, os.environ['HOME'] + "/code/PointNetGPD/dex-net/data/grippers")
ags = GpgGraspSamplerPcl(gripper, yaml_config)

#using_mp=True
using_mp=True
show_single=True

show_mp=True

num_grasps=10
num_workers=10
max_num_samples=50
marker_life_time = 20

rospy.set_param("/robot_at_home", "true")

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
    n = 500  # parameter related to voxel method
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
    #这里，算是将点云进行了voxel降采样
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
    select_point_above_table = 0.070
    #modify of gpg: make it as a parameter. avoid select points near the table.
    #查看每个点的z方向，如果它们的点z轴方向的值大于select_point_above_table，就把他们抽出来
    points_for_sample = points_[np.where(points_[:, 2] > select_point_above_table)[0]]
    print("待抓取的点数量为{}".format(len(points_for_sample)))
    if len(points_for_sample) == 0:
        rospy.loginfo("Can not seltect point, maybe the point cloud is too low?")
        return [], points_, surface_normal
    yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = 0.4
    if not using_mp:
        rospy.loginfo("Begin cal grasps using single thread, slow!")
        """

        """
        grasps_together_ = ags.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps,
                                             max_num_samples=max_num_samples, show_final_grasp=show_single)
    else:
        # begin parallel grasp:
        rospy.loginfo("Begin cal grasps using parallel!")

        def grasp_task(num_grasps_, ags_, queue_):
            ret = ags_.sample_grasps(point_cloud, points_for_sample, surface_normal, num_grasps_,
                                     max_num_samples=max_num_samples, show_final_grasp=False)
            queue_.put(ret)

        queue = mp.Queue()
        num_grasps_p_worker = int(num_grasps/num_workers)
        workers = [mp.Process(target=grasp_task, args=(num_grasps_p_worker, ags, queue)) for _ in range(num_workers)]
        [i.start() for i in workers]




        grasps_together_ = []
        for i in range(num_workers):
            grasps_together_ = grasps_together_ + queue.get()
        rospy.loginfo("Finish mp processing!")

        #显示多线程的抓取计算结果
        if show_mp:
            ags.show_all_grasps(points_, grasps_together_)
            ags.show_points(points_, scale_factor=0.002)
            mlab.show()

    rospy.loginfo("Grasp sampler finish, generated {} grasps.".format(len(grasps_together_)))


    #返回抓取(主要是抓取坐标系)     全部场景点      以及pcl计算的点云表面法向量
    return grasps_together_, points_, surface_normal


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

def get_voxel_fun(points_, n):
    get_voxel = voxelgrid.VoxelGrid(points_, n_x=n, n_y=n, n_z=n)
    get_voxel.compute()
    points_voxel_ = get_voxel.voxel_centers[get_voxel.voxel_n]
    points_voxel_ = np.unique(points_voxel_, axis=0)
    return points_voxel_


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


if __name__ == '__main__':

    rospy.init_node('grasp_tf_broadcaster', anonymous=True)
    #创建发布器
    pub1 = rospy.Publisher('gripper_vis', MarkerArray, queue_size=1)
    #发布检测到的抓取
    pub2 = rospy.Publisher('/detect_grasps/clustered_grasps', GraspConfigList, queue_size=1)
    #
    pub3 = rospy.Publisher('/test_points', PointCloud2, queue_size=1)

    rate = rospy.Rate(10)

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
            #尝试查看kinect2相机与桌子之间的转换？
            #cam_pos, _ = listener.lookupTransform('/table_top', '/kinect2_ir_optical_frame', rospy.Time(0))

            #cam_pos代表的是相机的trans，不是rot
            cam_pos, _ = listener.lookupTransform('/ar_marker_6', '/kinect2_rgb_optical_frame', rospy.Time(0))
            get_transform = True
            rospy.loginfo("got transform complete")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue


    while not rospy.is_shutdown():
       
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
        real_grasp   猜测此时的
        points
        normals_cal
        """
        real_grasp, points, normals_cal = cal_grasp(kinect_data, cam_pos)
        #托盘，如果有托盘？
        if True:
            #real_grasp 去除外部托盘导致的抓取；
            real_grasp = remove_grasp_outside_tray(real_grasp, points)
        #估计一个抓取中的点数
        check_grasp_points_num = True  # evaluate the number of points in a grasp
        
        """
        等效于
        if check_grasp_points_num:
            check_hand_points_fun(real_grasp)
        else:
            0
        """
        #check_hand_points_fun(real_grasp) if check_grasp_points_num else 0

        #计算，每个抓取，夹爪内部的点云
        #in_ind, in_ind_points = collect_pc(real_grasp, points)


        score = []  # should be 0 or 1
        score_value = []  # should be float [0, 1]

        ind_good_grasp = []
        ind_bad_grasp = []



        #记录好的抓取的数量，以及坏的抓取数量
        print("Got {}  grasps".format(len(real_grasp)))

        real_bad_grasp = real_grasp
        # end of grasp detection抓取检测部分结束

        marker_array = MarkerArray()
        marker_array_single = MarkerArray()
        grasp_msg_list = GraspConfigList()


        for i in range(len(real_bad_grasp)):
            show_grasp_marker(marker_array, real_bad_grasp[i], gripper, (1, 0, 0), marker_life_time)

        id_ = 0
        for m in marker_array.markers:
            m.id = id_
            id_ += 1

        grasp_msg_list.header.stamp = rospy.Time.now()
        grasp_msg_list.header.frame_id = "/ar_marker_6"

        # from IPython import embed;embed()
        if True:
            i = 0

            for m in marker_array_single.markers:
                m.id = id_
                id_ += 1
            pub1.publish(marker_array)
            rospy.sleep(4)
            pub1.publish(marker_array_single)
        rospy.loginfo(" Publishing grasp pose to rviz using marker array and good grasp pose")
        rate.sleep()


