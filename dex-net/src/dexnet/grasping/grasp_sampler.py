# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import copy
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import os, IPython, sys
import random
import time
import scipy.stats as stats
try:
    import pcl
except ImportError as e:
    print("[grasp_sampler] {}".format(e))
import dexnet

from dexnet.grasping import Grasp, Contact3D, ParallelJawPtGrasp3D, PointGraspMetrics3D  # , GraspableObject3D
from autolab_core import RigidTransform
import scipy
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

USE_OPENRAVE = True
try:
    import openravepy as rave
except ImportError:
    logger.warning('Failed to import OpenRAVE')
    USE_OPENRAVE = False

try:
    import rospy
    import moveit_commander
except ImportError:
    logger.warning("Failed to import rospy, you can't grasp now.")

try:
    from mayavi import mlab
except ImportError:
    mlab = []
    logger.warning('Do not have mayavi installed, please set the vis to False')

"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Classes for sampling grasps.
Author: Jeff Mahler
"""


# class GraspSampler(metaclass=ABCMeta):
#抓取采样基类
class GraspSampler:
    """ Base class for various methods to sample a number of grasps on an object.
    Should not be instantiated directly.

    Attributes
    ----------
    gripper : :obj:`RobotGripper`
        the gripper to compute grasps for
    config : :obj:`YamlConfig`
        configuration for the grasp sampler
    """
    __metaclass__ = ABCMeta

    #通过初始化函数读取夹爪参数
    def __init__(self, gripper, config):
        self.gripper = gripper
        self._configure(config)

    def _configure(self, config):
        """ Configures the grasp generator."""
        self.friction_coef = config['sampling_friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.target_num_grasps = config['target_num_grasps']
        if self.target_num_grasps is None:
            self.target_num_grasps = config['min_num_grasps']

        self.min_contact_dist = config['min_contact_dist']
        self.num_grasp_rots = config['num_grasp_rots']
        if 'max_num_surface_points' in list(config.keys()):
            self.max_num_surface_points_ = config['max_num_surface_points']
        else:
            self.max_num_surface_points_ = 100
        if 'grasp_dist_thresh' in list(config.keys()):
            self.grasp_dist_thresh_ = config['grasp_dist_thresh']
        else:
            self.grasp_dist_thresh_ = 0

    @abstractmethod
    def sample_grasps(self, graspable, num_grasps_generate, vis, **kwargs):
        """
        Create a list of candidate grasps for a given object.
        Must be implemented for all grasp sampler classes.

        Parameters
        ---------
        graspable : :obj:`GraspableObject3D`
            object to sample grasps on
        num_grasps_generate : int
        vis : bool
        """
        grasp = []
        return grasp
        # pass

    def generate_grasps_stable_poses(self, graspable, stable_poses, target_num_grasps=None, grasp_gen_mult=5,
                                     max_iter=3, sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object, aligning the approach angles to the object stable poses.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        stable_poses : :obj:`list` of :obj:`meshpy.StablePose`
            list of stable poses for the object with ids read from the database
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        vis : bool
        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # sample dense grasps
        unaligned_grasps = self.generate_grasps(graspable, target_num_grasps=target_num_grasps,
                                                grasp_gen_mult=grasp_gen_mult,
                                                max_iter=max_iter, vis=vis)

        # align for each stable pose
        grasps = {}
        print(sample_approach_angles)  # add by Liang
        for stable_pose in stable_poses:
            grasps[stable_pose.id] = []
            for grasp in unaligned_grasps:
                aligned_grasp = grasp.perpendicular_table(grasp)
                grasps[stable_pose.id].append(copy.deepcopy(aligned_grasp))
        return grasps

    def generate_grasps(self, graspable, target_num_grasps=None, grasp_gen_mult=5, max_iter=3,
                        sample_approach_angles=False, vis=False, **kwargs):
        """Samples a set of grasps for an object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        target_num_grasps : int
            number of grasps to return, defualts to self.target_num_grasps
        grasp_gen_mult : int
            number of additional grasps to generate
        max_iter : int
            number of attempts to return an exact number of grasps before giving up
        sample_approach_angles : bool
            whether or not to sample approach angles
        vis : bool
            whether show the grasp on picture

        Return
        ------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            list of generated grasps
        """
        # get num grasps
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps
        num_grasps_remaining = target_num_grasps

        grasps = []
        k = 1
        while num_grasps_remaining > 0 and k <= max_iter:
            # SAMPLING: generate more than we need
            num_grasps_generate = grasp_gen_mult * num_grasps_remaining
            #这里调用它的子类的具体重定义的sample_grasps函数
            new_grasps = self.sample_grasps(graspable, num_grasps_generate, vis, **kwargs)

            # COVERAGE REJECTION: prune grasps by distance
            pruned_grasps = []
            for grasp in new_grasps:
                min_dist = np.inf
                for cur_grasp in grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                for cur_grasp in pruned_grasps:
                    dist = ParallelJawPtGrasp3D.distance(cur_grasp, grasp)
                    if dist < min_dist:
                        min_dist = dist
                if min_dist >= self.grasp_dist_thresh_:
                    pruned_grasps.append(grasp)

            # ANGLE EXPANSION sample grasp rotations around the axis
            candidate_grasps = []
            if sample_approach_angles:
                for grasp in pruned_grasps:
                    # construct a set of rotated grasps
                    for i in range(self.num_grasp_rots):
                        rotated_grasp = copy.copy(grasp)
                        delta_theta = 0  # add by Hongzhuo Liang
                        print("This function can not use yes, as delta_theta is not set. --Hongzhuo Liang")
                        rotated_grasp.set_approach_angle(i * delta_theta)
                        candidate_grasps.append(rotated_grasp)
            else:
                candidate_grasps = pruned_grasps

            # add to the current grasp set
            grasps += candidate_grasps
            logger.info('%d/%d grasps found after iteration %d.',
                        len(grasps), target_num_grasps, k)

            grasp_gen_mult *= 2
            num_grasps_remaining = target_num_grasps - len(grasps)
            k += 1

        # shuffle computed grasps
        random.shuffle(grasps)
        if len(grasps) > target_num_grasps:
            logger.info('Truncating %d grasps to %d.',
                        len(grasps), target_num_grasps)
            grasps = grasps[:target_num_grasps]
        logger.info('Found %d grasps.', len(grasps))
        return grasps

    def show_points(self, point, color='lb', scale_factor=.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        elif color == 'lb':  # light blue
            color_f = (0.22, 1, 1)
        else:
            color_f = (1, 1, 1)
        if point.size == 3:  # vis for only one point, shape must be (3,), for shape (1, 3) is not work
            point = point.reshape(3, )
            mlab.points3d(point[0], point[1], point[2], color=color_f, scale_factor=scale_factor)
        else:  # vis for multiple points
            mlab.points3d(point[:, 0], point[:, 1], point[:, 2], color=color_f, scale_factor=scale_factor)

    def show_line(self, un1, un2, color='g', scale_factor=0.0005):
        if color == 'b':
            color_f = (0, 0, 1)
        elif color == 'r':
            color_f = (1, 0, 0)
        elif color == 'g':
            color_f = (0, 1, 0)
        else:
            color_f = (1, 1, 1)
        mlab.plot3d([un1[0], un2[0]], [un1[1], un2[1]], [un1[2], un2[2]], color=color_f, tube_radius=scale_factor)

    def show_grasp_norm_oneside(self, grasp_bottom_center,
                                grasp_normal, grasp_axis, minor_pc, scale_factor=0.001):

        # un1 = grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width
        un2 = grasp_bottom_center
        # un3 = grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width
        # un4 = grasp_bottom_center
        # un5 = grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width
        # un6 = grasp_bottom_center
        #把当前选择的点画成一个球
        self.show_points(grasp_bottom_center, color='g', scale_factor=scale_factor * 4)
        # self.show_points(un1, scale_factor=scale_factor * 4)
        # self.show_points(un3, scale_factor=scale_factor * 4)
        # self.show_points(un5, scale_factor=scale_factor * 4)
        # self.show_line(un1, un2, color='g', scale_factor=scale_factor)  # binormal/ major pc
        # self.show_line(un3, un4, color='b', scale_factor=scale_factor)  # minor pc
        # self.show_line(un5, un6, color='r', scale_factor=scale_factor)  # approach normal
        #画箭头，起始点坐标xyz为un2[0], un2[1], un2[2]
        #终止点坐标xyz 为grasp_axis[0], grasp_axis[1], grasp_axis[2]
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_axis[0], grasp_axis[1], grasp_axis[2],
                      scale_factor=.03, line_width=0.25, color=(0, 1, 0), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], minor_pc[0], minor_pc[1], minor_pc[2],
                      scale_factor=.03, line_width=0.1, color=(0, 0, 1), mode='arrow')
        mlab.quiver3d(un2[0], un2[1], un2[2], grasp_normal[0], grasp_normal[1], grasp_normal[2],
                      scale_factor=.03, line_width=0.05, color=(1, 0, 0), mode='arrow')


    #这个函数，主要是根据yaml文件中的夹爪参数，将夹爪近似为方形的结构，并返回每个角点的坐标,这些点就完全定义了夹爪的X形状和尺寸
    #hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

    def get_hand_points(self, grasp_bottom_center, approach_normal, binormal):
        hh = self.gripper.hand_height
        fw = self.gripper.finger_width
        hod = self.gripper.hand_outer_diameter
        hd = self.gripper.hand_depth
        #open_w是夹爪内部的宽度
        open_w = hod - fw * 2
        #计算minor_pc，并单位化
        minor_pc = np.cross(approach_normal, binormal)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)

        p5_p6 = minor_pc * hh * 0.5 + grasp_bottom_center
        p7_p8 = -minor_pc * hh * 0.5 + grasp_bottom_center
        p5 = -binormal * open_w * 0.5 + p5_p6
        p6 = binormal * open_w * 0.5 + p5_p6
        p7 = binormal * open_w * 0.5 + p7_p8
        p8 = -binormal * open_w * 0.5 + p7_p8
        p1 = approach_normal * hd + p5
        p2 = approach_normal * hd + p6
        p3 = approach_normal * hd + p7
        p4 = approach_normal * hd + p8

        p9 = -binormal * fw + p1
        p10 = -binormal * fw + p4
        p11 = -binormal * fw + p5
        p12 = -binormal * fw + p8
        p13 = binormal * fw + p2
        p14 = binormal * fw + p3
        p15 = binormal * fw + p6
        p16 = binormal * fw + p7

        p17 = -approach_normal * hh + p11
        p18 = -approach_normal * hh + p15
        p19 = -approach_normal * hh + p16
        p20 = -approach_normal * hh + p12
        p = np.vstack([np.array([0, 0, 0]), p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                       p11, p12, p13, p14, p15, p16, p17, p18, p19, p20])
        return p

    def show_grasp_3d(self, hand_points, color=(0.003, 0.50196, 0.50196)):
        # for i in range(1, 21):
        #     self.show_points(p[i])
        if color == 'd':
            color = (0.003, 0.50196, 0.50196)
        triangles = [(9, 1, 4), (4, 9, 10), (4, 10, 8), (8, 10, 12), (1, 4, 8), (1, 5, 8),
                     (1, 5, 9), (5, 9, 11), (9, 10, 20), (9, 20, 17), (20, 17, 19), (17, 19, 18),
                     (14, 19, 18), (14, 18, 13), (3, 2, 13), (3, 13, 14), (3, 6, 7), (3, 6, 2),
                     (3, 14, 7), (14, 7, 16), (2, 13, 15), (2, 15, 6), (12, 20, 19), (12, 19, 16),
                     (15, 11, 17), (15, 17, 18), (6, 7, 8), (6, 8, 5)]
        mlab.triangular_mesh(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2],
                             triangles, color=color, opacity=0.5)
    
    #检查当前姿态下的夹爪和夹爪内部的点有没有碰撞
    """
    open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, all_points,
                                                                     hand_points, "p_open")
    """
    def check_collision_square(self, grasp_bottom_center, approach_normal, binormal,
                               minor_pc, graspable, p, way, vis=False):
        """检查点云与夹爪指定部位的碰撞情况
        """
        #单位化
        approach_normal = approach_normal.reshape(1, 3)
        approach_normal = approach_normal / np.linalg.norm(approach_normal)

        binormal = binormal.reshape(1, 3)
        binormal = binormal / np.linalg.norm(binormal)

        minor_pc = minor_pc.reshape(1, 3)
        minor_pc = minor_pc / np.linalg.norm(minor_pc)

        #将三个列向量  水平拼接
        matrix = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
        #转置
        grasp_matrix = matrix.T  # same as cal the inverse

        #这个graspable就是  allpoints（全部点）
        #isinstance  判断，如果graspable的类型和dexnet.grasping.graspable_object.GraspableObject3D是相同的，那么就返回true
        #
        if isinstance(graspable, dexnet.grasping.graspable_object.GraspableObject3D):
            points = graspable.sdf.surface_points(grid_basis=False)[0]
        else:
            points = graspable
        #获取所有的点相对于夹爪底部中心点的向量
        points = points - grasp_bottom_center.reshape(1, 3)
        # points_g = points @ grasp_matrix
        tmp = np.dot(grasp_matrix, points.T)
        points_g = tmp.T
        #p_open 代表，检查闭合区域内部有没有点，不包含夹爪自身碰撞，只是看夹爪内部有没有点云
        if way == "p_open": 
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        #判断右侧夹爪本身，有没有与点云产生碰撞
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        #判断左侧夹爪本身，有没有与点云产生碰撞
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        #判断顶端夹爪本身，有没有与点云产生碰撞
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        #查找points_g中所有y坐标大于p1点的y坐标
        a1 = s1[1] < points_g[:, 1]    #y
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]    #z
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]    #x
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            #不存在点
            has_p = False
        else:
            has_p = True

        if vis:
            print("points_in_area", way, len(points_in_area))
            mlab.clf()
            # self.show_one_point(np.array([0, 0, 0]))
            self.show_grasp_3d(p)
            #self.show_points(grasp_bottom_center, color='b', scale_factor=.008)
            #注意这一点，在检查的时候，参考系是相机坐标系，夹爪的相对相机的位姿并没有改变，
            # 他们反而是对点云进行了变换，搞不懂这样有什么好处
            self.show_points(points_g)
            # 画出抓取坐标系
            #self.show_grasp_norm_oneside(grasp_bottom_center,approach_normal, binormal,minor_pc, scale_factor=0.001)

            if len(points_in_area) != 0:
                self.show_points(points_g[points_in_area], color='r')
            
            mlab.show()
        # print("points_in_area", way, len(points_in_area))
        #返回是否有点has_p，以及，内部点的索引list
        return has_p, points_in_area
    
    """自添加：检测夹爪在给定姿态下是否会与桌面碰撞
    """
    def check_collision_hand_table(self, grasp_bottom_center, approach_normal,
        binormal,minor_pc,  safe_dis=0.01 ,vis=False):
        #获取夹爪各个点
        hand_points=self.get_hand_points(grasp_bottom_center,
                                                                approach_normal,
                                                                binormal)
        hand_points_=hand_points[1:]
        #找夹爪上的所有点的z坐标最小值对应的点就是夹爪的最低点
        min_finger_end = hand_points_[:, 2].min()
        #返回夹爪最低点的索引
        min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0]

        if min_finger_end < safe_dis:
            collision_table =True
        else:
            collision_table=False

        return collision_table, min_finger_end_pos_ind





    def show_all_grasps(self, all_points, grasps_for_show):
        """此时的grasp坐标系不是固定的，是以桌面标签坐标系为参考坐标系
        """
        print("Show {} grasps".format(len(grasps_for_show)))
        for grasp_ in grasps_for_show:
            grasp_bottom_center = grasp_[4]  # new feature: ues the modified grasp bottom center
            approach_normal = grasp_[1]
            binormal = grasp_[2]
            hand_points = self.get_hand_points(grasp_bottom_center, approach_normal, binormal)
            self.show_grasp_3d(hand_points)
        # self.show_points(all_points)
        # mlab.show()

    def check_collide(self, grasp_bottom_center, approach_normal, binormal, minor_pc, graspable, hand_points):
        """仅仅检查点云是否与夹爪碰撞，而不检查桌面
        """
        bottom_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                    binormal, minor_pc, graspable, hand_points, "p_bottom")
        if bottom_points[0]:
            return True

        left_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                  binormal, minor_pc, graspable, hand_points, "p_left")
        if left_points[0]:
            return True

        right_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                   binormal, minor_pc, graspable, hand_points, "p_right")
        if right_points[0]:
            return True
        return False

    def check_collide_(self, grasp_bottom_center, approach_normal, binormal, minor_pc, graspable, hand_points):
        """
        检查点云是否和夹爪，以及桌面碰撞
        不同于check_collide函数
        """
        bottom_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                    binormal, minor_pc, graspable, hand_points, "p_bottom")
        if bottom_points[0]:
            return True

        left_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                  binormal, minor_pc, graspable, hand_points, "p_left")
        if left_points[0]:
            return True

        right_points = self.check_collision_square(grasp_bottom_center, approach_normal,
                                                   binormal, minor_pc, graspable, hand_points, "p_right")
        if right_points[0]:
            return True

        table_points=self.check_collision_hand_table(grasp_bottom_center,approach_normal,
                                                    binormal,minor_pc,safe_dis=0.01)
        if table_points[0]:
            return True

        return False

    def cal_surface_property(self, graspable, selected_surface, r_ball,
                             point_amount, max_trial, vis=False):
        tmp_count = 0
        M = np.zeros((3, 3))
        trial = 0
        old_normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(selected_surface))
        if old_normal is None:
            logger.warning("The selected point has no norm according to meshpy!")
            return None
        while tmp_count < point_amount and trial < max_trial:
            trial += 1
            neighbor = selected_surface + 2 * (np.random.rand(3) - 0.5) * r_ball
            normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(neighbor))
            if normal is None:
                continue
            normal = normal.reshape(-1, 1)
            if np.linalg.norm(normal) != 0:
                normal /= np.linalg.norm(normal)
            if vis:
                # show the r-ball performance
                neighbor = neighbor.reshape(-1, 1)
                # self.show_line(neighbor, normal * 0.05 + neighbor)
            M += np.matmul(normal, normal.T)
            tmp_count = tmp_count + 1

        if trial == max_trial:
            logger.warning("rball computation failed over %d", max_trial)
            return None

        eigval, eigvec = np.linalg.eig(M)  # compared computed normal
        minor_pc = eigvec[np.argmin(eigval)]  # minor principal curvature
        minor_pc /= np.linalg.norm(minor_pc)
        new_normal = eigvec[np.argmax(eigval)]  # estimated surface normal
        new_normal /= np.linalg.norm(new_normal)
        major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
        if np.linalg.norm(major_pc) != 0:
            major_pc = major_pc / np.linalg.norm(major_pc)
        return old_normal, new_normal, major_pc, minor_pc


class UniformGraspSampler(GraspSampler):
    """ Sample grasps by sampling pairs of points on the object surface uniformly at random.
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]
        i = 0
        grasps = []

        # get all grasps
        while len(grasps) < num_grasps and i < max_num_samples:
            # get candidate contacts
            indices = np.random.choice(num_surface, size=2, replace=False)
            c0 = surface_points[indices[0], :]
            c1 = surface_points[indices[1], :]

            gripper_distance = np.linalg.norm(c1 - c0)
            if self.gripper.min_width < gripper_distance < self.gripper.max_width:
                # compute centers and axes
                grasp_center = ParallelJawPtGrasp3D.center_from_endpoints(c0, c1)
                grasp_axis = ParallelJawPtGrasp3D.axis_from_endpoints(c0, c1)
                g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center,
                                                                                        grasp_axis,
                                                                                        self.gripper.max_width))
                # keep grasps if the fingers close
                if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                    angle_candidates = np.arange(-90, 120, 30)
                    np.random.shuffle(angle_candidates)
                    for grasp_angle in angle_candidates:
                        g.approach_angle_ = grasp_angle
                        # get true contacts (previous is subject to variation)
                        success, contacts = g.close_fingers(graspable, vis=vis)
                        if not success:
                            continue
                        break
                    else:
                        continue
                else:
                    success, contacts = g.close_fingers(graspable, vis=vis)

                if success:
                    grasps.append(g)
            i += 1

        return grasps


class GaussianGraspSampler(GraspSampler):
    """ Sample grasps by sampling a center from a gaussian with mean at the object center of mass
    and grasp axis by sampling the spherical angles uniformly at random.
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, sigma_scale=2.5, **kwargs):
        """
        Returns a list of candidate grasps for graspable object by Gaussian with
        variance specified by principal dimensions.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        sigma_scale : float
            the number of sigmas on the tails of the Gaussian for each dimension
        vis : bool
            visualization

        Returns
        -------
        :obj:`list` of obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        # get object principal axes
        center_of_mass = graspable.mesh.center_of_mass
        principal_dims = graspable.mesh.principal_dims()
        sigma_dims = principal_dims / (2 * sigma_scale)

        # sample centers
        grasp_centers = stats.multivariate_normal.rvs(
            mean=center_of_mass, cov=sigma_dims ** 2, size=num_grasps)

        # samples angles uniformly from sphere
        u = stats.uniform.rvs(size=num_grasps)
        v = stats.uniform.rvs(size=num_grasps)
        thetas = 2 * np.pi * u
        phis = np.arccos(2 * v - 1.0)
        grasp_dirs = np.array([np.sin(phis) * np.cos(thetas), np.sin(phis) * np.sin(thetas), np.cos(phis)])
        grasp_dirs = grasp_dirs.T

        # convert to grasp objects
        grasps = []
        for i in range(num_grasps):
            grasp = ParallelJawPtGrasp3D(
                ParallelJawPtGrasp3D.configuration_from_params(grasp_centers[i, :], grasp_dirs[i, :],
                                                               self.gripper.max_width))

            if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                angle_candidates = np.arange(-90, 120, 30)
                np.random.shuffle(angle_candidates)
                for grasp_angle in angle_candidates:
                    grasp.approach_angle_ = grasp_angle
                    # get true contacts (previous is subject to variation)
                    success, contacts = grasp.close_fingers(graspable, vis=vis)
                    if not success:
                        continue
                    break
                else:
                    continue
            else:
                success, contacts = grasp.close_fingers(graspable, vis=vis)

            # add grasp if it has valid contacts
            if success and np.linalg.norm(contacts[0].point - contacts[1].point) > self.min_contact_dist:
                grasps.append(grasp)

        # visualize
        if vis:
            for grasp in grasps:
                plt.clf()
                plt.gcf()
                plt.ion()
                grasp.close_fingers(graspable, vis=vis)
                plt.show(block=False)
                time.sleep(0.5)

            grasp_centers_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_centers.T)
            grasp_centers_grid = grasp_centers_grid.T
            com_grid = graspable.sdf.transform_pt_obj_to_grid(center_of_mass)

            plt.clf()
            ax = plt.gca(projection='3d')
            # graspable.sdf.scatter()
            ax.scatter(grasp_centers_grid[:, 0], grasp_centers_grid[:, 1], grasp_centers_grid[:, 2], s=60, c='m')
            ax.scatter(com_grid[0], com_grid[1], com_grid[2], s=120, c='y')
            ax.set_xlim3d(0, graspable.sdf.dims_[0])
            ax.set_ylim3d(0, graspable.sdf.dims_[1])
            ax.set_zlim3d(0, graspable.sdf.dims_[2])
            plt.show()

        return grasps


class AntipodalGraspSampler(GraspSampler):
    """ Samples antipodal pairs using rejection sampling.
    The proposal sampling ditribution is to choose a random point on
    the object surface, then sample random directions within the friction cone,
    then form a grasp axis along the direction,
    close the fingers, and keep the grasp if the other contact point is also in the friction cone.
    """

    def sample_from_cone(self, n, tx, ty, num_samples=1):
        """ Samples directoins from within the friction cone using uniform sampling.

        Parameters
        ----------
        n : 3x1 normalized :obj:`numpy.ndarray`
            surface normal
        tx : 3x1 normalized :obj:`numpy.ndarray`
            tangent x vector
        ty : 3x1 normalized :obj:`numpy.ndarray`
            tangent y vector
        num_samples : int
            number of directions to sample

        Returns
        -------
        v_samples : :obj:`list` of 3x1 :obj:`numpy.ndarray`
            sampled directions in the friction cone
       """
        v_samples = []
        for i in range(num_samples):
            theta = 2 * np.pi * np.random.rand()
            r = self.friction_coef * np.random.rand()
            v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
            v = -v / np.linalg.norm(v)
            v_samples.append(v)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Checks whether or not a direction is in the friction cone.
        This is equivalent to whether a grasp will slip using a point contact model.

        Parameters
        ----------
        cone : 3xN :obj:`numpy.ndarray`
            supporting vectors of the friction cone
        n : 3x1 :obj:`numpy.ndarray`
            outward pointing surface normal vector at c1
        v : 3x1 :obj:`numpy.ndarray`
            direction vector

        Returns
        -------
        in_cone : bool
            True if alpha is within the cone
        alpha : float
            the angle between the normal and v
        """
        if (v.dot(cone) < 0).any():  # v should point in same direction as cone
            v = -v  # don't worry about sign, we don't know it anyway...
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point """
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def sample_grasps(self, graspable, num_grasps, vis=False, **kwargs):
        """在这里完成对点抓取的具体采样过程
        Returns a list of candidate grasps for graspable object.

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            number of grasps to sample
        vis : bool
            whether or not to visualize progress, for debugging

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
            the sampled grasps
        """
        # get surface points
        grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        np.random.shuffle(surface_points)
        shuffled_surface_points = surface_points[:min(self.max_num_surface_points_, len(surface_points))]
        logger.info('Num surface: %d' % (len(surface_points)))

        for k, x_surf in enumerate(shuffled_surface_points):
            # print("k:", k, "len(grasps):", len(grasps))
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(self.num_samples):
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)

                # compute friction cone faces
                c1 = Contact3D(graspable, x1, in_direction=None)
                _, tx1, ty1 = c1.tangents()
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(n1, tx1, ty1, num_samples=1)
                sample_time = time.clock()

                for v in v_samples:
                    if vis:
                        x1_grid = graspable.sdf.transform_pt_obj_to_grid(x1)
                        cone1_grid = graspable.sdf.transform_pt_obj_to_grid(cone1, direction=True)
                        plt.clf()
                        plt.gcf()
                        plt.ion()
                        ax = plt.gca(projection=Axes3D)
                        for j in range(cone1.shape[1]):
                            ax.scatter(x1_grid[0] - cone1_grid[0], x1_grid[1] - cone1_grid[1],
                                       x1_grid[2] - cone1_grid[2], s=50, c='m')

                    # random axis flips since we don't have guarantees on surface normal directoins
                    if random.random() > 0.5:
                        v = -v

                    # start searching for contacts  在这里,是根据
                    #此时这里的grasp并不是一个完全定义的抓取，此时他们把旋转angle使用了0来代替
                    grasp, c1, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(
                        graspable, x1, v, self.gripper.max_width,
                        min_grasp_width_world=self.gripper.min_width, vis=vis)

                    if grasp is None or c2 is None:
                        continue

                    #为grasp  配置一个绕接触点c1  c2为轴的旋转角度
                    if 'random_approach_angle' in kwargs and kwargs['random_approach_angle']:
                        angle_candidates = np.arange(-90, 120, 30)
                        np.random.shuffle(angle_candidates)
                        for grasp_angle in angle_candidates:
                            grasp.approach_angle_ = grasp_angle
                            # get true contacts (previous is subject to variation)
                            success, c = grasp.close_fingers(graspable, vis=vis)
                            if not success:
                                continue
                            break
                        else:
                            continue
                    else:
                        success, c = grasp.close_fingers(graspable, vis=vis)
                        if not success:
                            continue
                    c1 = c[0]
                    c2 = c[1]

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2   计算接触点c2的摩擦锥
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        plt.figure()
                        ax = plt.gca(projection='3d')
                        c1_proxy = c1.plot_friction_cone(color='m')
                        c2_proxy = c2.plot_friction_cone(color='y')
                        ax.view_init(elev=5.0, azim=0)
                        plt.show(block=False)
                        time.sleep(0.5)
                        plt.close()  # lol

                    # check friction cone  检查所生成的抓取是否是力闭合的
                    # 即，检测接触点c1  c2是不是在摩擦锥内部
                    if PointGraspMetrics3D.force_closure(c1, c2, self.friction_coef):
                        grasps.append(grasp)

        # randomly sample max num grasps from total list
        random.shuffle(grasps)
        return grasps


class GpgGraspSampler(GraspSampler):
    """
    Sample grasps by GPG.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=30, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'debug_vis': False,
            'r_ball': self.gripper.hand_height,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 3000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        all_points = surface_points
        # construct pynt point cloud and voxel grid
        p_cloud = pcl.PointCloud(surface_points.astype(np.float32))
        voxel = p_cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(*([graspable.sdf.resolution * params['voxel_grid_ratio']] * 3))
        surface_points = voxel.filter().to_array()

        num_surface = surface_points.shape[0]
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # get candidate contacts
            ind = np.random.choice(num_surface, size=1, replace=False)
            selected_surface = surface_points[ind, :].reshape(3)

            # cal major principal curvature
            # r_ball = max(self.gripper.hand_depth, self.gripper.hand_outer_diameter)
            r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            point_amount = params['num_rball_points']
            max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.
            ret = self.cal_surface_property(graspable, selected_surface, r_ball,
                                            point_amount, max_trial, vis=params['debug_vis'])
            if ret is None:
                continue
            else:
                old_normal, new_normal, major_pc, minor_pc = ret

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.
            if np.dot(old_normal, new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc

            for normal_dir in [1.]:  # FIXME: here we can now know the direction of norm, outward
                if params['debug_vis']:
                    # example of show grasp frame
                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)
                    # self.show_points(all_points)

                # some magic number referred from origin paper
                potential_grasp = []
                for dtheta in np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy
                        # go back a bite after rotation dtheta and translation dy!
                        tmp_grasp_bottom_center = self.gripper.init_bite * (
                                -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, graspable,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, graspable,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])
                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        potential_grasp.append(dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)])
                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist / params['approach_step'])
                for ptg in potential_grasp:
                    for approach_s in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]

                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, graspable, hand_points)
                        if is_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center += (-tmp_grasp_normal) * params['approach_step']

                            # final check
                            open_points, _ = self.check_collision_square(tmp_grasp_bottom_center,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, graspable, hand_points)
                            if open_points and not is_collide:
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc, tmp_grasp_bottom_center])

                                self.show_points(selected_surface, color='r', scale_factor=.005)
                                if params['debug_vis']:
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points,
                                                                "p_open", vis=True)
                            break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            if not sampled_surface_amount % 20:  # params['debug_vis']:
                # self.show_all_grasps(all_points, processed_potential_grasp)
                # self.show_points(all_points)
                # mlab.show()
                return processed_potential_grasp
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp


class PointGraspSampler(GraspSampler):
    """
    Sample grasps by PointGraspSampler
    TODO: since gpg sampler changed a lot, this class need to totally rewrite
    """

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree
            'range_dtheta': 90,
            'debug_vis': True,
            'approach_step': 0.005,
            'max_trail_for_r_ball': 3000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        all_points = surface_points
        # construct pynt point cloud and voxel grid
        p_cloud = pcl.PointCloud(surface_points.astype(np.float32))
        voxel = p_cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(*([graspable.sdf.resolution * params['voxel_grid_ratio']] * 3))
        surface_points = voxel.filter().to_array()

        num_surface = surface_points.shape[0]
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # get candidate contacts
            # begin of modification 5: Gaussian over height, select more point in the middle
            # we can use the top part of the point clouds to generate more sample points
            min_height = min(surface_points[:, 2])
            max_height = max(surface_points[:, 2])

            selected_height = min_height + np.random.normal(3 * (max_height - min_height) / 4,
                                                            (max_height - min_height) / 6)
            ind_10 = np.argsort(abs(surface_points[:, 2] - selected_height))[:10]
            ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5
            # ind = np.random.choice(num_surface, size=1, replace=False)
            selected_surface = surface_points[ind, :].reshape(3)

            # cal major principal curvature
            r_ball = max(self.gripper.hand_depth, self.gripper.hand_outer_diameter)
            point_amount = params['num_rball_points']
            max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.
            ret = self.cal_surface_property(graspable, selected_surface, r_ball,
                                            point_amount, max_trial, vis=params['debug_vis'])
            if ret is None:
                continue
            else:
                old_normal, new_normal, major_pc, minor_pc = ret

            for normal_dir in [-1., 1.]:  # FIXME: here we do not know the direction of the object normal
                grasp_bottom_center = self.gripper.init_bite * new_normal * -normal_dir + selected_surface
                new_normal = normal_dir * new_normal
                major_pc = normal_dir * major_pc

                if params['debug_vis']:
                    # example of show grasp frame
                    self.show_grasp_norm_oneside(selected_surface, new_normal, major_pc,
                                                 minor_pc, scale_factor=0.001)
                    self.show_points(selected_surface, color='g', scale_factor=.002)

                # some magic number referred from origin paper
                potential_grasp = []
                extra_potential_grasp = []

                for dtheta in np.arange(-params['range_dtheta'],
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes
                        tmp_major_pc = np.dot(rotation, major_pc)
                        tmp_grasp_normal = np.dot(rotation, new_normal)
                        tmp_grasp_bottom_center = grasp_bottom_center + tmp_major_pc * dy

                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, graspable,
                                                                     hand_points, "p_open")
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, graspable,
                                                                       hand_points,
                                                                       "p_bottom")
                        if open_points is True and bottom_points is False:

                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points,
                                                                         "p_left")
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, graspable,
                                                                          hand_points,
                                                                          "p_right")

                            if left_points is False and right_points is False:
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])
                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        # potential_grasp += dy_potentials
                        potential_grasp.append(dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)])

                    # get more potential_grasp by moving along minor_pc
                    if len(potential_grasp) != 0:
                        self.show_points(selected_surface, color='r', scale_factor=.005)
                        for pt in potential_grasp:
                            for dz in range(-5, 5):
                                new_center = minor_pc * dz * 0.01 + pt[0]
                                extra_potential_grasp.append([new_center, pt[1], pt[2], pt[3]])
                approach_dist = self.gripper.hand_depth  # use gripper depth
                num_approaches = int(approach_dist // params['approach_step'])
                for ptg in extra_potential_grasp:
                    for _ in range(num_approaches):
                        tmp_grasp_bottom_center = ptg[1] * params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        not_collide = self.check_approach_grasp(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points)

                        if not not_collide:
                            # if collide, go back one step to get a collision free hand position
                            tmp_grasp_bottom_center = -ptg[1] * params['approach_step'] + ptg[0]
                            # final check
                            open_points, _ = self.check_collision_square(tmp_grasp_bottom_center,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, graspable,
                                                                         hand_points, "p_open")
                            not_collide = self.check_approach_grasp(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                    tmp_major_pc, minor_pc, graspable, hand_points)
                            if open_points and not_collide:
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc])

                                self.show_points(selected_surface, color='r', scale_factor=.005)
                                if params['debug_vis']:
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, graspable, hand_points,
                                                                "p_open", vis=True)
                        break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            if not sampled_surface_amount % 60:  # params['debug_vis']:
                self.show_all_grasps(all_points, processed_potential_grasp)
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp


class OldPointGraspSampler(GraspSampler):
    """
    Sample grasps by PointGraspSampler
    """

    def show_obj(self, graspable, color='b', clear=False):
        if clear:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()

        # plot the obj
        ax = plt.gca(projection='3d')
        surface = graspable.sdf.surface_points()[0]
        surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
        ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                   s=np.ones_like(surface[:, 0]) * 0.3, c=color)

    def show_grasp_norm(self, graspable, grasp_center, grasp_bottom_center,
                        grasp_normal, grasp_axis, minor_pc, color='b', clear=False):
        if clear:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()

        ax = plt.gca(projection='3d')
        grasp_center_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_center)
        ax.scatter(grasp_center_grid[0], grasp_center_grid[1], grasp_center_grid[2], marker='s', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * grasp_axis * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * grasp_axis * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='x', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * minor_pc * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='^', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * minor_pc * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='^', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center + 0.5 * grasp_normal * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='*', c=color)
        grasp_center_bottom_grid = graspable.sdf.transform_pt_obj_to_grid(
            grasp_bottom_center - 0.5 * grasp_normal * self.gripper.max_width)
        ax.scatter(grasp_center_bottom_grid[0], grasp_center_bottom_grid[1], grasp_center_bottom_grid[2],
                   marker='*', c=color)

    def sample_grasps(self, graspable, num_grasps, vis=False, max_num_samples=1000, **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        graspable : :obj:`GraspableObject3D`
            the object to grasp
        num_grasps : int
            the number of grasps to generate
        vis :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 0.3,
            'range_dtheta': 0.30,
            'max_chain_length': 20,
            'max_retry_times': 100
        }

        # get all surface points
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        num_surface = surface_points.shape[0]

        i = 0
        self.grasps = []

        # ____count = 0
        # get all grasps
        while len(self.grasps) < num_grasps and i < max_num_samples:
            # print('sample times:', ____count)
            # ____count += 1
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(surface_points[:, 2])
            # max_height = max(surface_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(surface_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]

            # end of modification 5
            ind = np.random.choice(num_surface, size=1, replace=False)
            grasp_bottom_center = surface_points[ind, :]
            grasp_bottom_center = grasp_bottom_center.reshape(3)

            for ind in range(params['max_chain_length']):
                # if not graspable.sdf.on_surface(graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center))[0]:
                #     print('first damn it!')
                #     from IPython import embed; embed()
                new_grasp_bottom_center = self.sample_chain(grasp_bottom_center, graspable,
                                                            params, vis)
                if new_grasp_bottom_center is None:
                    i += ind
                    break
                else:
                    grasp_bottom_center = new_grasp_bottom_center
            else:
                i += params['max_chain_length']
            print('Chain broken, length:', ind, 'amount:', len(self.grasps))
        return self.grasps

    def sample_chain(self, grasp_bottom_center, graspable, params, vis):
        grasp_success = False
        grasp_normal = graspable.sdf.surface_normal(
            graspable.sdf.transform_pt_obj_to_grid(grasp_bottom_center))
        for normal_dir in [-1., 1.]:  # FIXME: here we assume normal is outward
            grasp_center = self.gripper.max_depth * normal_dir * grasp_normal + grasp_bottom_center
            r_ball = max(self.gripper.max_depth, self.gripper.max_width)
            # cal major principal curvature
            tmp_count = 0
            M = np.zeros((3, 3))
            while tmp_count < params['num_rball_points']:
                neighbor = grasp_bottom_center + 2 * (np.random.rand(3) - 0.5) * r_ball
                normal = graspable.sdf.surface_normal(graspable.sdf.transform_pt_obj_to_grid(neighbor))
                if normal is None:
                    continue
                normal = normal.reshape(-1, 1)
                M += np.matmul(normal, normal.T)
                tmp_count = tmp_count + 1
            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            minor_pc = eigvec[np.argmin(eigval)]  # minor principal curvature
            minor_pc /= np.linalg.norm(minor_pc)
            new_normal = eigvec[np.argmax(eigval)]  # estimated surface normal
            new_normal /= np.linalg.norm(new_normal)
            # major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            # FIXME: We can not get accurate normal, so we use grasp_normal instead
            major_pc = np.cross(minor_pc, grasp_normal)
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)
            grasp_axis = major_pc

            g = ParallelJawPtGrasp3D(
                ParallelJawPtGrasp3D.configuration_from_params(
                    grasp_center,
                    grasp_axis,
                    self.gripper.max_width))
            grasp_success, _ = g.close_fingers(graspable, vis=vis)
            if grasp_success:
                self.grasps.append(g)
        if not grasp_success:
            return None

        trial = 0
        next_grasp_bottom_center = None
        while trial < params['max_retry_times'] and next_grasp_bottom_center is None:
            trial += 1
            dy = np.random.uniform(-params['num_dy'] * self.gripper.finger_width,
                                   (params['num_dy']) * self.gripper.finger_width)
            dtheta = np.random.uniform(-params['range_dtheta'], params['range_dtheta'])

            for tmp_normal_dir in [-1., 1.]:
                # get new grasp sample from a chain
                # some magic number referred from origin paper

                # compute centers and axes
                x, y, z = minor_pc
                rotation = RigidTransform.rotation_from_quaternion(
                    np.array([dtheta / 180 * np.pi, x, y, z]))
                tmp_grasp_axis = np.dot(rotation, grasp_axis)
                tmp_grasp_normal = np.dot(rotation, grasp_normal)

                tmp_grasp_bottom_center = grasp_bottom_center + tmp_grasp_axis * dy
                # TODO: find contact
                # FIXME: 0.2 is the same as close_finger()
                approach_dist = 0.2
                approach_dist_grid = graspable.sdf.transform_pt_obj_to_grid(approach_dist)
                num_approach_samples = int(Grasp.samples_per_grid * approach_dist_grid / 2)
                approach_loa = ParallelJawPtGrasp3D.create_line_of_action(tmp_grasp_bottom_center,
                                                                          -tmp_grasp_normal * tmp_normal_dir,
                                                                          approach_dist,
                                                                          graspable,
                                                                          num_approach_samples,
                                                                          min_width=0)
                contact_found, contact = ParallelJawPtGrasp3D.find_contact(approach_loa,
                                                                           graspable, vis=vis)
                if not contact_found:
                    continue
                else:
                    if not graspable.sdf.on_surface(graspable.sdf.transform_pt_obj_to_grid(contact.point))[0]:
                        # print('damn it!')
                        pass
                    else:
                        next_grasp_bottom_center = contact.point
                        break
        print('amount:', len(self.grasps), 'next center:', next_grasp_bottom_center)
        return next_grasp_bottom_center


class GpgGraspSamplerPcl(GraspSampler):
    """
    从点云中直接采样候选抓取，一般用于在线阶段的抓取采样中
    Sample grasps by GPG with pcl directly.
    http://journals.sagepub.com/doi/10.1177/0278364917735594
    """
    def sample_grasps(self, point_cloud,points_for_sample, all_normal, num_grasps=20, max_num_samples=200, show_final_grasp=False,
                      **kwargs):
        """
        Returns a list of candidate grasps for graspable object using uniform point pairs from the SDF

        Parameters
        ----------
        point_cloud :
        all_normal :
        num_grasps : int
            the number of grasps to generate

        show_final_grasp :
        max_num_samples :

        Returns
        -------
        :obj:`list` of :obj:`ParallelJawPtGrasp3D`
           list of generated grasps
        """
        params = {
            'num_rball_points': 27,  # FIXME: the same as meshpy..surface_normal()
            'num_dy': 10,  # number
            'dtheta': 10,  # unit degree   绕轴旋转的单位角度
            'range_dtheta': 90,            #一共旋转 +- 90度
            'debug_vis': False,
            'r_ball': self.gripper.hand_height,
            'approach_step': 0.005,     #前推后退的单位值
            'max_trail_for_r_ball': 1000,
            'voxel_grid_ratio': 5,  # voxel_grid/sdf.resolution
        }

        # get all surface points获取所有桌面点云，转换到ndarry（区别于points_for_sample）
        all_points = point_cloud.to_array()
        sampled_surface_amount = 0
        grasps = []
        processed_potential_grasp = []

        #以参考系原点为夹爪中心，以X正向为approach轴，主曲率方向设置为Y轴
        hand_points = self.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
        """
        if params['debug_vis']:
            for x in range(1,21):
                self.show_grasp_3d(hand_points)
                #self.show_points(np.array([0, 0, 0]), color='b', scale_factor=.008)
                self.show_points(hand_points[x,:],color='r', scale_factor=.008)
                mlab.show()
        """

        # get all grasps
        while len(grasps) < num_grasps and sampled_surface_amount < max_num_samples:
            # begin of modification 5: Gaussian over height
            # we can use the top part of the point clouds to generate more sample points
            # min_height = min(all_points[:, 2])
            # max_height = max(all_points[:, 2])
            # selected_height = max_height - abs(np.random.normal(max_height, (max_height - min_height)/3)
            #                                    - max_height)
            # ind_10 = np.argsort(abs(all_points[:, 2] - selected_height))[:10]
            # ind = ind_10[np.random.choice(len(ind_10), 1)]
            # end of modification 5

            # for ros, we neded to judge if the robot is at HOME

            if rospy.get_param("/robot_at_home") == "false":
                robot_at_home = False
            else:
                robot_at_home = True
            if not robot_at_home:
                rospy.loginfo("robot is moving! wait untill it go home. Return empty gpg!")
                return []
            
            scipy.random.seed()  # important! without this, the worker will get a pseudo-random sequences.如果不做这一步，就会产生一个伪-随机序列
            """
            points_for_sample.shape[0]  返回ndarry的行数
            points_for_sample.shape[1]  返回ndarry的列数
            size=1  随机抽出一个数
            np.random.choice(points_for_sample.shape[0], size=1, replace=False) 从[0,points_for_sample.shape[0]]中随机抽出一个值（索引）
            """
            ind = np.random.choice(points_for_sample.shape[0], size=1, replace=False)

            """
            points_for_sample[ind, :]   就是获取刚才随机抽取的索引对应的点的坐标（是其中的一行）
            points_for_sample[ind, :].reshape(3, )     就是将刚才1x3的行向量，变为3x1的列向量？？？？？
            """
            selected_surface = points_for_sample[ind, :].reshape(3, )
            if show_final_grasp:
                #把选出来的点画出来，坐标为selected_surface[0], selected_surface[1], selected_surface[2]
                #颜色为红色球体，直径为5mm
                mlab.points3d(selected_surface[0], selected_surface[1], selected_surface[2],
                              color=(1, 0, 0), scale_factor=0.005)

            # cal major principal curvature
            # r_ball = params['r_ball']  # FIXME: for some relative small obj, we need to use pre-defined radius
            """
            看起来像是构造了一个球的半径，从三个值中挑出一个最大的
            self.gripper.hand_outer_diameter - self.gripper.finger_width    
            self.gripper.hand_depth   
            self.gripper.hand_height / 2.0
            """
            r_ball = max(self.gripper.hand_outer_diameter - self.gripper.finger_width, self.gripper.hand_depth,
                         self.gripper.hand_height / 2.0)

            # point_amount = params['num_rball_points']
            # max_trial = params['max_trail_for_r_ball']
            # TODO: we can not directly sample from point clouds so we use a relatively small radius.

            M = np.zeros((3, 3))

            # neighbor = selected_surface + 2 * (np.random.rand(3) - 0.5) * r_ball

            #构造只包含刚才随机抽选的一个点selected_surface的点云！
            selected_surface_pc = pcl.PointCloud(selected_surface.reshape(1, 3))
            #以全部点云point_cloud构建一个快速kdtree
            kd = point_cloud.make_kdtree_flann()
            #以selected_surface_pc这个一个点为中心，以r_ball为半径，搜索最近的100个点，返回索引和对应的欧氏距离
            kd_indices, sqr_distances = kd.radius_search_for_cloud(selected_surface_pc, r_ball, 100)
            #print(kd_indices)
            #print(kd_indices[0])

            for _ in range(len(kd_indices[0])):
                if sqr_distances[0, _] != 0:
                    # neighbor = point_cloud[kd_indices]
                    #读取搜索到的点的法向量读取出来
                    normal = all_normal[kd_indices[0, _]]
                    #把法向量转换为列向量
                    normal = normal.reshape(-1, 1)
                    #如果该向量内容不全是0（2范数非零）
                    if np.linalg.norm(normal) != 0:
                        #法向量单位化
                        normal /= np.linalg.norm(normal)
                    #法向量相乘，之后，叠加   得到M矩阵   
                    M += np.matmul(normal, normal.T)
            if sum(sum(M)) == 0:
                print("M matrix is empty as there is no point near the neighbour")
                print("Here is a bug, if points amount is too little it will keep trying and never go outside.")
                continue
            else:
                logger.info("Selected a good sample point.")

            """
            计算M矩阵的特征值eigval与特征向量eigvec
            """
            eigval, eigvec = np.linalg.eig(M)  # compared computed normal
            #这个点minor_pc 和下面的major_pc 与原始点selected_surface构成的向量与该店法向量之间是两两垂直的（用于后来画坐标系）
            minor_pc = eigvec[:, np.argmin(eigval)].reshape(3)  # minor principal curvature !!! Here should use column!
            minor_pc /= np.linalg.norm(minor_pc)
            #计算得到一个新的法向量（比较准）
            new_normal = eigvec[:, np.argmax(eigval)].reshape(3)  # estimated surface normal !!! Here should use column!
            new_normal /= np.linalg.norm(new_normal)

            major_pc = np.cross(minor_pc, new_normal)  # major principal curvature
            if np.linalg.norm(major_pc) != 0:
                major_pc = major_pc / np.linalg.norm(major_pc)

            # Judge if the new_normal has the same direction with old_normal, here the correct
            # direction in modified meshpy is point outward.判断新的法向量的方向是否和旧的法向量方向相同（正确的是向外）
            #如果新旧向量之间夹角小于0  就将新的向量反向
            if np.dot(all_normal[ind], new_normal) < 0:
                new_normal = -new_normal
                minor_pc = -minor_pc
            # FIXED: we know the direction of norm is outward as we know the camera pos这个感觉是备选-1和1，可以手动选择方向，
            # 不过normal_dir已经被选为1，因为刚才已经调整了新的向量的方向了（与相机位置一致）
            for normal_dir in [1]:
                if params['debug_vis']:
                    # example of show grasp frame;new_normal * normal_dir就是为了改变方向，现在已经固定了，
                    # 画出抓取坐标系
                    self.show_grasp_norm_oneside(selected_surface, new_normal * normal_dir, major_pc * normal_dir,
                                                 minor_pc, scale_factor=0.001)

                    #这句话又重新画了一遍随机选中的那个原始点
                    self.show_points(selected_surface, color='g', scale_factor=.002)
                    self.show_points(all_points)
                    # show real norm direction: if new_norm has very diff than pcl cal norm, then maybe a bug.
                    #使用两点selected_surface，selected_surface + all_normal[ind]*0.05来画出一个线段长度是5cm
                    self.show_line(selected_surface, (selected_surface + all_normal[ind]*0.05).reshape(3))
                    mlab.show()

                """
                刚才上面的代码完成如下的工作：
                1.随机选择了一个场景点（采样点，夹爪的底部位置？？？）
                2.根据这个场景点重新计算了一个精确一些的法向量，并根据采样点具体形状，计算了其他的两个相互垂直的轴，构建了一个抓取局部固定坐标系

                下面的代码主要完成抓取检测的工作，来自论文
                """
                # some magic number referred from origin paper  从原始的论文中
                potential_grasp = []
                #构建离散的旋转角度，从-90到+90度  间隔为10度，这一部分是，先离散的旋转角度部分
                for dtheta in np.arange(-params['range_dtheta'],    
                                        params['range_dtheta'] + 1,
                                        params['dtheta']):
                    dy_potentials = []
                    x, y, z = minor_pc
                    #得到当前的旋转角度
                    dtheta = np.float64(dtheta)
                    #dtheta是绕minor_pc轴旋转的，将这个绕轴旋转quaternion  转换为刚体旋转
                    rotation = RigidTransform.rotation_from_quaternion(np.array([dtheta / 180 * np.pi, x, y, z]))


                    #沿着y轴的偏移，这里是以夹爪的厚度为单位移动的
                    for dy in np.arange(-params['num_dy'] * self.gripper.finger_width,
                                        (params['num_dy'] + 1) * self.gripper.finger_width,
                                        self.gripper.finger_width):
                        # compute centers and axes  计算此时的中心点
                        tmp_major_pc = np.dot(rotation, major_pc * normal_dir)#计算一下，旋转后的主曲率的向量
                        tmp_grasp_normal = np.dot(rotation, new_normal * normal_dir)#计算一下，旋转后的表面法相方向
                        tmp_grasp_bottom_center = selected_surface + tmp_major_pc * dy#  因为偏移是沿着y轴的，所以叠加偏移就好

                        #translation=RigidTransform.translation(tmp_major_pc * dy)

                        # go back a bite after rotation dtheta and translation dy! 沿着旋转后的表面法相方向，撤回一点点距离
                        tmp_grasp_bottom_center = self.gripper.init_bite * (
                                -tmp_grasp_normal * normal_dir) + tmp_grasp_bottom_center

                        #if  params['debug_vis']:
                        if 0:
                            #hand_points_=np.dot(rotation,hand_points)
                            #hand_points_=np.dot(translation,hand_points_)
                            #self.show_grasp_3d(hand_points_)
                            print("绕z旋转{}度".format(dtheta))
                            print("沿y偏移距离{}".format(dy))
                            self.show_grasp_norm_oneside(tmp_grasp_bottom_center,tmp_grasp_normal,
                                         tmp_major_pc,minor_pc, scale_factor=0.001)
                            self.show_points(all_points)
                            mlab.show()

                        #完成了一次采样之后，就要查看夹爪张开状态下，夹爪内部是否存在点云
                        open_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                     tmp_major_pc, minor_pc, all_points,
                                                                     hand_points, "p_open")
                        
                        bottom_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                       tmp_major_pc, minor_pc, all_points,
                                                                       hand_points,
                                                                       "p_bottom")

                        #如果夹爪工作范围内有点云，且夹爪顶端没有与场景点云进行碰撞
                        if open_points is True and bottom_points is False:
                            #检测左侧夹爪是否和场景发生碰撞
                            left_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points,
                                                                         "p_left")
                            #检测右侧夹爪是否和场景发生了碰撞
                            right_points, _ = self.check_collision_square(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                          tmp_major_pc, minor_pc, all_points,
                                                                          hand_points,
                                                                          "p_right")
                            #如果夹爪左右都没有与点云发生碰撞
                            if left_points is False and right_points is False:
                                #就把当前的夹爪坐标系，添加到当前采样点的候选列表中
                                dy_potentials.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                      tmp_major_pc, minor_pc])
                    #如果当前采样点的当前角度下的偏移候选列表中存在候选抓取
                    if len(dy_potentials) != 0:
                        # we only take the middle grasp from dy direction.
                        center_dy = dy_potentials[int(np.ceil(len(dy_potentials) / 2) - 1)]
                        # we check if the gripper has a potential to collide with the table检查是否会与桌子碰撞
                        # by check if the gripper is grasp from a down to top direction检查，夹爪是不是从下向上抓的（穿桌子）
                        finger_top_pos = center_dy[0] + center_dy[1] * self.gripper.hand_depth
                        
                        """
                        self.show_grasp_norm_oneside(center_dy[0],center_dy[1],
                                         center_dy[2],center_dy[3], scale_factor=0.001)
                        self.show_points(finger_top_pos,color='r',scale_factor=0.008)
                        self.show_points(all_points)
                        mlab.show()
                        """

                        # [- self.gripper.hand_depth * 0.5] means we grasp objects as a angel larger than 30 degree
                        #这个就是检测一下姿态，不要
                        if finger_top_pos[2] < center_dy[0][2] - self.gripper.hand_depth * 0.5:
                            potential_grasp.append(center_dy)
                ###########完成了一个采样点的所有角度，以及所有的偏移###############
                #定义从多远开始靠近抓取物体
                approach_dist = self.gripper.hand_depth  # use gripper depth
                #计算分多少步靠近物体
                num_approaches = int(approach_dist / params['approach_step'])

                #显示一下，当前采样点，产生的所有可能的候选抓
                if 0:
                    print("当前采样点的候选抓取姿态数量{}".format(len(potential_grasp)))
                    for ptg in potential_grasp:
                        tmp_center = ptg[0]
                        tmp_normal = ptg[1]
                        tmp_pc = ptg[2]
                        pc = ptg[3]
                        
                        show_hand_points = self.get_hand_points(tmp_center,
                                                                    tmp_normal,
                                                                    tmp_pc)

                        self.show_grasp_3d(show_hand_points)
                        self.show_points(tmp_center,color='b',scale_factor=0.005)

                    self.show_grasp_norm_oneside(selected_surface, new_normal, major_pc,
                                                    minor_pc, scale_factor=0.001)
                    self.show_points(all_points)
                    #画桌子所在的平面
                    table_points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * 0.5
                    triangles = [(1, 2, 3), (0, 1, 3)]
                    mlab.triangular_mesh(table_points[:, 0], table_points[:, 1], table_points[:, 2],
                                         triangles, color=(0.8, 0.8, 0.8), opacity=0.5)
                    mlab.show()






                """对上述刚生成的所有的可能抓取抓取（还没有进一步做全体点云碰撞检测）进行桌面及全体点云碰撞检测和姿态调整
                1.沿着approach轴，前推所有的抓取（为了找到极限值），而且对前推的抓取进行全体点云碰撞检测
                2.前推找到极限的抓取之后，后撤一段距离，并进行桌面碰撞检测
                3.如果桌面检测没通过，就进一步调整夹爪，
                4.如果调整后的夹爪姿态通过了碰撞检查，就加入到processed_potential_grasp中
                注意点: 
                1.并不是所有的抓取都进行了调整，只有那些和全部点云产生了碰撞的抓取才会进行调整
                2.调整后的抓取抽了出来，调整后又放在了processed_potential_grasp中
                3.桌面水平面的设置，是按照桌面标签为参考系的，检测是否与桌面发生碰撞的安全距离也是以标签为准，
                    而不是，再次从场景点云中计算出来一个平面
                4.
                """
                for ptg in potential_grasp:
                    for approach_s in range(num_approaches):
                        #沿着抓取的approach轴向物体推进
                        tmp_grasp_bottom_center = ptg[1] * approach_s * params['approach_step'] + ptg[0]
                        tmp_grasp_normal = ptg[1]
                        tmp_major_pc = ptg[2]
                        minor_pc = ptg[3]
                        #推进之后，检查此时的夹爪是否与全部场景点云碰撞，没有碰撞时，将返回false
                        #疑问：这里就只是检查了与全部点云的一个碰撞，并没有检测与桌面的碰撞，他应该
                        # 是默认了输入的点云都是没有剔除平面的，这就导致一个问题，如果输入的点云已经
                        # 是剔除过桌子的，如果此时虽然与点云没有碰撞，但是实际上与桌子有碰撞，此时
                        # 是检测不出来的，并且会跳过下面的姿态调整环节，

                        #现在已经做了修改
                        is_collide = self.check_collide_(tmp_grasp_bottom_center, tmp_grasp_normal,
                                                        tmp_major_pc, minor_pc, point_cloud, hand_points)
                        
                        #如果有碰撞（必有碰撞，因为上面会循环着不断推进，每个原始的抓取都会不断推进，直到碰撞）
                        if is_collide:
                            # if collide, go back one step to get a collision free hand position，这点实际上不需要，因为后面会调整姿态直到满足安全距离
                            #tmp_grasp_bottom_center += (-tmp_grasp_normal) * params['approach_step'] * 3
                            # minus 3 means we want the grasp go back a little bitte more.退回的多一些

                            # here we check if the gripper collide with the table.检查是不是和桌子碰撞了
                            #在这里，变换了夹爪的位姿！相当于扭转了夹爪的点云
                            hand_points_ = self.get_hand_points(tmp_grasp_bottom_center,
                                                                tmp_grasp_normal,
                                                                tmp_major_pc)[1:]

                            #找夹爪上的所有点的z坐标最小值对应的点就是夹爪的最低点
                            min_finger_end = hand_points_[:, 2].min()
                            #返回夹爪最低点的索引
                            min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0]
                            #min_finger_end_pos_ind = np.where(hand_points_[:, 2] == min_finger_end)[0][0]


                            #设定，距离桌面的最小安全距离为1cm
                            safety_dis_above_table = 0.01
                            

                            #如果距离已经小于最小安全距离了，就修正它一下
                            if min_finger_end < safety_dis_above_table:
                                #读取最低点三维坐标
                                min_finger_pos = hand_points_[min_finger_end_pos_ind]  # the lowest point in a gripper
                                min_finger_pos_=min_finger_pos.reshape(-1,1)
                                #print(min_finger_pos)
                                #先找到，该夹爪最低点沿着此时grasp_normal在桌面上的投射点
                                x = -min_finger_pos_[2]*tmp_grasp_normal[0]/tmp_grasp_normal[2]+min_finger_pos_[0]
                                y = -min_finger_pos_[2]*tmp_grasp_normal[1]/tmp_grasp_normal[2]+min_finger_pos_[1]
                                p_table = np.array([x, y, 0])  # the point that on the table夹爪最低点，沿着当前的grasp_normal射向桌子的投影点
                                #计算桌子投射点与夹爪最低点的向量，准备用于计算欧氏距离
                                p_hand=p_table-min_finger_pos
                                
                                #
                                if min_finger_end >=0:
                                    #dis_go_back = np.linalg.norm([min_finger_pos, p_table]) + safety_dis_above_table
                                    dis_go_back = (safety_dis_above_table-min_finger_pos_[2])*np.linalg.norm(p_hand)/min_finger_pos_[2]
                                    tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center-tmp_grasp_normal*dis_go_back
                                elif min_finger_end <0:
                                    #dis_go_back = np.linalg.norm([min_finger_pos, p_table]) + safety_dis_above_table
                                    dis_go_back = (min_finger_pos_[2]-safety_dis_above_table)*np.linalg.norm(p_hand)/min_finger_pos_[2]
                                    tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center-tmp_grasp_normal*dis_go_back


                            else:
                                # if the grasp is not collide with the table, do not change the grasp
                                tmp_grasp_bottom_center_modify = tmp_grasp_bottom_center

                            # final check最终检查一下修正后的抓取的碰撞情况，原始的没有检查是否与桌子碰撞
                            _, open_points = self.check_collision_square(tmp_grasp_bottom_center_modify,
                                                                         tmp_grasp_normal,
                                                                         tmp_major_pc, minor_pc, all_points,
                                                                         hand_points, "p_open")
                            is_collide = self.check_collide_(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                            tmp_major_pc, minor_pc, all_points, hand_points)
                            #如果夹爪内部的点云数量大于10了，并且没有碰撞
                            if (len(open_points) > 10) and not is_collide:
                                # here 10 set the minimal points in a grasp, we can set a parameter later
                                #将修整后的抓取存起来
                                processed_potential_grasp.append([tmp_grasp_bottom_center, tmp_grasp_normal,
                                                                  tmp_major_pc, minor_pc,
                                                                  tmp_grasp_bottom_center_modify])
                                if params['debug_vis']:
                                    self.show_points(selected_surface, color='r', scale_factor=.005)
                                    logger.info('usefull grasp sample point original: %s', selected_surface)
                                    self.check_collision_square(tmp_grasp_bottom_center_modify, tmp_grasp_normal,
                                                                tmp_major_pc, minor_pc, all_points, hand_points,
                                                                "p_open", vis=True)
                                break
                logger.info("processed_potential_grasp %d", len(processed_potential_grasp))

            sampled_surface_amount += 1
            #打印出来已经进行的采样次数
            logger.info("current amount of sampled surface %d", sampled_surface_amount)
            #print("current amount of sampled surface:", sampled_surface_amount)

            if params['debug_vis']:  # not sampled_surface_amount % 5:
                if len(all_points) > 10000:
                    #把全部点ndarry 转换到pointcloud
                    pc = pcl.PointCloud(all_points)
                    #设置降采样
                    voxel = pc.make_voxel_grid_filter()
                    #设置降采样格子为1cm
                    voxel.set_leaf_size(0.01, 0.01, 0.01)
                    point_cloud = voxel.filter()
                    all_points = point_cloud.to_array()
                self.show_all_grasps(all_points, processed_potential_grasp)
                self.show_points(all_points, scale_factor=0.008)
                mlab.show()
            #print("The grasps number got by modified GPG:", len(processed_potential_grasp))

            #如果采样到的抓取已经满足自己要求找的数量了，或者是  
            # 采样的次数，已经超过了最大允许的采样次数，就停止采样
            if len(processed_potential_grasp) >= num_grasps or sampled_surface_amount >= max_num_samples:
                if show_final_grasp:
                    #显示点云中所有的点； 以及生成的抓取
                    self.show_all_grasps(all_points, processed_potential_grasp)
                    self.show_points(all_points, scale_factor=0.002)

                    mlab.points3d(0, 0, 0, scale_factor=0.01, color=(0, 1, 0))

                    table_points = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * 0.5
                    triangles = [(1, 2, 3), (0, 1, 3)]
                    mlab.triangular_mesh(table_points[:, 0], table_points[:, 1], table_points[:, 2],
                                         triangles, color=(0.8, 0.8, 0.8), opacity=0.5)
                    mlab.show()
                return processed_potential_grasp
            #
            # g = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
            #     tmp_grasp_center,
            #     tmp_major_pc,
            #     self.gripper.max_width))
            # grasps.append(g)

        return processed_potential_grasp
