import os
import glob
import pickle

import pcl
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

import math


#导入夹爪相关，是想，从离线训练的过程中就使用自己的夹爪构型
from dexnet.grasping import RobotGripper
from dexnet.grasping import GpgGraspSamplerPcl
try:
    from mayavi import mlab
except ImportError:
    print("Can not import mayavi")
    mlab = None

# global config:全局的配置文件
from autolab_core import YamlConfig
yaml_config = YamlConfig(os.environ['HOME'] + "/code/dex-net/test/config.yaml")
gripper_name = 'robotiq_85'
#加载夹爪
gripper = RobotGripper.load(gripper_name, os.environ['HOME'] + "/code/dex-net/data/grippers")
ags = GpgGraspSamplerPcl(gripper, yaml_config)



"""
pointnetGPD的输入是点云，是夹爪内部的点云，是部分的点云
在检测完抓取之后，生成的是完整物体的点云以及某些姿态的夹爪姿态；

这个dataset.py中，利用完整的点云结合对应的夹爪姿态，进行了夹爪内部点云的提取
然后把这部分的点云投放到pointnet中

"""

class PointGraspDataset(torch.utils.data.Dataset):
    def __init__(self, obj_points_num, grasp_points_num, pc_file_used_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path, tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        self.obj_points_num = obj_points_num
        self.grasp_points_num = grasp_points_num
        self.pc_file_used_num = pc_file_used_num
        self.grasp_amount_per_file = grasp_amount_per_file
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num  = 50
        self.projection_margin = 1

        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', '*.npy'))

        self.d_pc, self.d_grasp = {}, {}
        for i in fl_pc:
            k = i.split('/')[-3]
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                self.d_pc[k] = [i]

        for i in fl_grasp:
            k = i.split('/')[-1].split('.')[0]
            self.d_grasp[k] = i
        object1 = set(self.d_grasp.keys())
        object2 = set(self.transform.keys())
        self.object = list(object1.intersection(object2))
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        center = grasp[0:3]
        axis = grasp[3:6] # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis/np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(axis, approach)

        left = center - width*axis/2
        right = center + width*axis/2
        # bottom = center - width*approach
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        # bottom = (transform @ np.array([bottom[0], bottom[1], bottom[2], 1]))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[:3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        # pc_t/left_t/right_t is in local coordinate(with center as origin)
        # other(include pc) are in pc coordinate
        pc_t = (np.dot(matrix, (pc-center).T)).T
        left_t = (-width * np.array([0,1,0]) / 2).squeeze()
        right_t = (width * np.array([0,1,0]) / 2).squeeze()

        x_limit = width/4
        z_limit = width/4
        y_limit = width/2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]

    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                        [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic-margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1)/number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert(occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check)!=0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert(np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))
        obj_grasp = self.object[obj_ind]
        obj_pc = self.transform[obj_grasp][0]
        f_grasp = self.d_grasp[obj_grasp]
        fl_pc = np.array(self.d_pc[obj_pc])
        fl_pc = fl_pc[np.random.choice(len(fl_pc), size=self.pc_file_used_num)]

        grasp = np.load(f_grasp)[grasp_ind]
        pc = np.vstack([np.load(i) for i in fl_pc])
        pc = pc[np.random.choice(len(pc), size=self.obj_points_num)]
        t = self.transform[obj_grasp][1]

        grasp_pc = self.collect_pc(grasp, pc, t)
        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            if len(grasp_pc) > self.grasp_points_num:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        score = level_score + refine_score*0.01
        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 1
        else:
            return None

        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount


class PointGraspMultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, obj_points_num, grasp_points_num, pc_file_used_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path, tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        self.obj_points_num = obj_points_num
        self.grasp_points_num = grasp_points_num
        self.pc_file_used_num = pc_file_used_num
        self.grasp_amount_per_file = grasp_amount_per_file
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num  = 50
        self.projection_margin = 1

        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', '*.npy'))

        self.d_pc, self.d_grasp = {}, {}
        for i in fl_pc:
            k = i.split('/')[-3]
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                self.d_pc[k] = [i]

        for i in fl_grasp:
            k = i.split('/')[-1].split('.')[0]
            self.d_grasp[k] = i
        object1 = set(self.d_grasp.keys())
        object2 = set(self.transform.keys())
        self.object = list(object1.intersection(object2))
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        center = grasp[0:3]
        axis = grasp[3:6] # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis/np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(axis, approach)

        left = center - width*axis/2
        right = center + width*axis/2
        # bottom = center - width*approach
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        # bottom = (transform @ np.array([bottom[0], bottom[1], bottom[2], 1]))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[:3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        # pc_t/left_t/right_t is in local coordinate(with center as origin)
        # other(include pc) are in pc coordinate
        pc_t = (np.dot(matrix, (pc-center).T)).T
        left_t = (-width * np.array([0,1,0]) / 2).squeeze()
        right_t = (width * np.array([0,1,0]) / 2).squeeze()

        x_limit = width/4
        z_limit = width/4
        y_limit = width/2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]

    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                        [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic-margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1)/number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert(occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check)!=0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert(np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))
        obj_grasp = self.object[obj_ind]
        obj_pc = self.transform[obj_grasp][0]
        f_grasp = self.d_grasp[obj_grasp]
        fl_pc = np.array(self.d_pc[obj_pc])
        fl_pc = fl_pc[np.random.choice(len(fl_pc), size=self.pc_file_used_num)]

        grasp = np.load(f_grasp)[grasp_ind]
        pc = np.vstack([np.load(i) for i in fl_pc])
        pc = pc[np.random.choice(len(pc), size=self.obj_points_num)]
        t = self.transform[obj_grasp][1]

        grasp_pc = self.collect_pc(grasp, pc, t)
        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            if len(grasp_pc) > self.grasp_points_num:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        score = level_score + refine_score*0.01
        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 2
        else:
            label = 1

        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount

"""
自定义的单视角数据集类，它的一个对象就是 一个抽象化的自己的数据集，在这里，将那个

"""
class PointGraspOneViewDataset(torch.utils.data.Dataset):
    #重写了__init__构造函数
    def __init__(self, grasp_points_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path,prefix,tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        #每个抓取内部的点数
        self.grasp_points_num = grasp_points_num
        #
        self.grasp_amount_per_file = grasp_amount_per_file
        #文件存放路径
        self.path = path
        #标注是训练还是测试
        self.tag = tag
        #离线计算出的姿态带有打分，可以在这里设定阈值
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        #这个没啥用
        self.with_obj = with_obj
        #只有在某姿态下，夹爪内部的点数量大于min_point_limit，才会返回点云，否则返回none
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        #投影通道数，这个不用管，这个是另一篇论文的
        self.project_chann = project_chann
        #保证通道位于3~12
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num  = 50
        self.projection_margin = 1
        self.minimum_point_amount = 150


        #关键是这个！  这个是ycb三维模型  与  ycb rgbd图像的转换姿态
        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        #获取 指定模式（train或者test）下的文件路径（抓取姿态的路径）
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        #获取点云对应的路径，这是什么意思，它好像只是选择了某一个角度的点云
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', 'pc_NP3_NP5*.npy'))

        #构建两个字典
        #d_pc中存放的是，被抓物体名称  - 路径list 的字典
        #d_grasp存放的是，
        self.d_pc, self.d_grasp = {}, {}
        #由于每个物体，都会有不同视角下的多个点云，下面的操作主要是，将所有点云的路径，按照所属物体名称分类到不同字典中
        for i in fl_pc:
            #对第i个点云文件路径，按照'/'符号切割，将k赋值为倒数第3个分段，是物体的名称
            #/home/wgk/win10/data/ycb_rgbd/003_cracker_box/clouds/pc_NP3_NP5_3.npy
            #k = 003_cracker_box
            k = i.split('/')[-3]
            #self.d_pc.keys()返回self.d_pc中所有的键值
            #如果存在键k(存在物体003_cracker_box)，就把地址i，存放在003_cracker_box对应的地址列表中
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                #否则直接添加键-新的列表
                self.d_pc[k] = [i]

        #d_pc[k]是一个列表，对它的内容进行sort()
        for k in self.d_pc.keys():
            self.d_pc[k].sort()
    
        #
        prefix=prefix+'_'
        for i in fl_grasp:
            #i="/home/wgk/win10/data/ycb_grasp/train/003_cracker_box.npy"
            #k="003_cracker_box"
            good_grasp_number=''
            #print(k)
            k = i.split('/')[-1].split('.')[0]
            

            #print(prefix)
            k=k.split(prefix)[-1]
            #print(k)
            good_grasp_number='_'+k.split('_')[-1]
            k=k.split(good_grasp_number)[0]
            #名称-抓取姿态文件路径
            self.d_grasp[k] = i

        #set()是构造一个无序不重复元素集合
        object0 = set(self.d_pc.keys())#object0包含的是点云文件夹中包含的物体名称集合（此时还不是列表）
        object1 = set(self.d_grasp.keys())#object1包含的是生成的抓取姿态中包含的物体名称集合（此时还不是列表）
        object2 = set(self.transform.keys())#object2包含的是变换文件中包含的物体名称集合（此时还不是列表）

        #object0.intersection(object1,object2) 代表，返回object0、object1、object2这三个元素集合的交集
        #即要求，对于物体A，要保证存在离线生成的抓取姿态、关于A的多视角点云 、模型&点云之间的变换关系（三个缺一不可）
        #在这里就是说，要保证生成抓取的物体A，在变换列表中能够找到它的变换矩阵，这个物体才是有效物体
        #否则，虽然计算出了候选抓取，但是却找不到某个视角下的点云和模型间的变换，也是白搭，没有办法计算
        #夹爪内部的点云
        self.object = list(object0.intersection(object1,object2))#self.object是有效物体的名称列表
        print(len(self.object))

        #这里，看着像是计算了在有效物体fl_grasp上的总共的抓取数量，但是有问题，感觉不太对
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        """在这里采集候选抓取姿态下，夹爪内部的点云
        grasp:这个应该是夹爪坐标系的位置？bottom_center
        pc:
        transform:
        """
        #hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        center = grasp[0:3] #注意不是bottom_center，而是两个表面接触点的  中点
        axis = grasp[3:6] # binormal  夹爪接触点构成的向量
        width = grasp[6] #夹爪的最大张开距离
        angle = grasp[7]  #夹爪沿着axis轴的旋转角度

        #axis轴  单位化
        axis = axis/np.linalg.norm(axis)
        binormal = axis
        #cal approach   计算抓取approach轴
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]

        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)

        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        #计算出来approach再单位化
        approach = approach / np.linalg.norm(approach)
        #计算那个minor轴
        minor_normal = np.cross(axis, approach)

        #计算bottom_center
        bottom=center - ags.gripper.hand_depth*approach

        left = center - width*axis/2
        right = center + width*axis/2
        # bottom = center - width*approach
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        #旋转bottom
        bottom = (transform @ np.array([bottom[0], bottom[1], bottom[2], 1]))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[:3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        # pc_t/left_t/right_t is in local coordinate(with center as origin)  
        # other(include pc) are in pc coordinate
        pc_t = (np.dot(matrix, (pc-center).T)).T
        left_t = (-width * np.array([0,1,0]) / 2).squeeze()
        right_t = (width * np.array([0,1,0]) / 2).squeeze()


        #获取夹爪姿态

        x_limit = width/4
        z_limit = width/4
        y_limit = width/2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]
    
    def collect_pc_(self, grasp, pc, transform,ply_locate):
        """在这里采集候选抓取姿态下，夹爪内部的点云
        grasp:这个应该是夹爪坐标系的位置？bottom_center
        pc:
        transform:由于离线生成的grasp  是以cad的模型为参考系，但是，我们需要在不同视角下的点云场景，并计算得到每个夹爪此时内部的点云
                            这个视角下的点云和cad模型之间是存在一种变换关系transform的，这里采用的方法是，将点云经过transform变换到
        """
        #hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        center = grasp[0:3] #注意不是bottom_center，而是两个表面接触点的  中点
        axis = grasp[3:6]      # binormal  夹爪接触点构成的向量
        width = grasp[6]      #夹爪的最大张开距离
        angle = grasp[7]      #夹爪沿着axis轴的旋转角度

        #axis轴  单位化
        axis = axis/np.linalg.norm(axis)
        binormal = axis
        #cal approach   计算抓取approach轴
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        """np.c_ 在操作一维向量时，将每个都当成列拼在一起
        [cos_t   0      -sin_t
            0         1          0
        sin_t     0      cos_t]
        """

        R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]

        axis_y = axis
        #直接取一个正交化的向量
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])

        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        #R2是一个矩阵[]
        R2 = np.c_[axis_x,np.c_[axis_y, axis_z]]

        approach = R2.dot(R1)[:, 0]
        #计算出来approach再单位化
        approach = approach / np.linalg.norm(approach)
        #print(approach)
        #计算那个minor轴
        minor_normal = np.cross(axis, approach)

        #计算bottom_center
        bottom=center - ags.gripper.hand_depth*approach

        #left = center - width*axis/2
        #right = center + width*axis/2
        # bottom = center - width*approach
        #left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        #right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]

        #旋转bottom
        #print(bottom)
        bottom_ = (np.dot(transform,np.array([bottom[0], bottom[1], bottom[2], 1]).T))[:3]
        #print(transform)
        #print(transform.shape)
        #print(bottom)

        center_ = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal_ = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach_ = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal_ = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[:3].reshape(3, 1)

        matrix = np.hstack([approach_, binormal_, minor_normal_]).T
        # pc_t/left_t/right_t is in local coordinate(with center as origin)  
        # other(include pc) are in pc coordinate


        #这里原始的方法，竟然使用了夹爪中心点来作为中心点，这点我觉的是不对的,
        #我将它修正为了bottom_center
        #pc_c2g = (np.dot(matrix, (pc-center).T)).T
        pc_c2g = (np.dot(matrix, (pc-bottom_).T)).T

        #pc_c2o=np.hstack([pc,np.ones(np.shape(pc)[0]).T]).T
        #pc_c2o=np.dot(transform,pc_c2o).T
        #pc_c2o=pc_c2o[:,:3]

        left_t = (-width * np.array([0,1,0]) / 2).squeeze()
        right_t = (width * np.array([0,1,0]) / 2).squeeze()


        #获取夹爪角点点坐标
        hand_points=ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        their_way=False
        if their_way:
            x_limit = width/4
            z_limit = width/4
            y_limit = width/2

            a1 = pc_t[:, 0] > -x_limit
            a2 = pc_t[:, 0] < x_limit
            a3 = pc_t[:, 1] > -y_limit
            a4 = pc_t[:, 1] < y_limit
            a5 = pc_t[:, 2] > -z_limit
            a6 = pc_t[:, 2] < z_limit

        else:
            s1, s2, s4, s8 = hand_points[1], hand_points[2], hand_points[4], hand_points[8]
            #查找points_g中所有y坐标大于p1点的y坐标
            a1 = s1[1] < pc_c2g[:, 1]    #y
            a2 = s2[1] > pc_c2g[:, 1]
            a3 = s1[2] > pc_c2g[:, 2]    #z
            a4 = s4[2] < pc_c2g[:, 2]
            a5 = s4[0] > pc_c2g[:, 0]    #x
            a6 = s8[0] < pc_c2g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if 0:
            mlab.clf()
            #hand_points_=ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
            hand_points_=ags.get_hand_points(bottom, approach, binormal)
            #画出物体的cad图像
            mlab.pipeline.surface(mlab.pipeline.open(ply_locate))
            # self.show_one_point(np.array([0, 0, 0]))
            score = grasp[-2] + grasp[-1]*0.01
            print("level_score{},refine_score{},finall_score{}".format(grasp[-2],grasp[-1],score))
            
            if score <=0.6:
                #good
                ags.show_grasp_3d(hand_points_,color=(0,1,0))
            elif score >= 0.6:
                #bad
                ags.show_grasp_3d(hand_points_,color=(1,0,0))
            else:
                #between  them   
                ags.show_grasp_3d(hand_points_)


            #self.show_points(grasp_bottom_center, color='b', scale_factor=.008)
            #注意这一点，在检查的时候，参考系是相机坐标系，夹爪的相对相机的位姿并没有改变，
            # 他们反而是对点云进行了变换，搞不懂这样有什么好处
            #ags.show_points(pc)
            ags.show_points(pc_c2g)
            # 画出抓取坐标系
            #self.show_grasp_norm_oneside(grasp_bottom_center,approach_normal, binormal,minor_pc, scale_factor=0.001)

            if len(self.in_ind) != 0:
                ags.show_points(pc_c2g[self.in_ind], color='r')
                #ags.show_points(pc_t[self.in_ind], color='r')
            
            mlab.show()


        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            #一般情况下，返回的是旋转后的点云，是以夹爪为参考系的点云
            return pc_c2g[self.in_ind]


    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                        [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic-margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1)/number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert(occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check)!=0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert(np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        #len(self.object)  有效物体（有抓取姿态，也有变换矩阵）的数量
        #self.grasp_amount_per_file 指定的每个物体的抓取数量
        #构建一个len(self.object)行，self.grasp_amount_per_file)列的矩阵，
        # 并展开成1行len(self.object)*self.grasp_amount_per_file列的数组
        #问，这个数组中，给定一个index，问第index个数据，处于第几行，第几列,
        #那么行obj_ind就是第几个目标物体的索引，grasp_ind指的是该物体第几个抓取
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))
        #self.object是 list
        #获取索引obj_ind代表的被抓物体名称如 026_sponge ，obj_grasp是string类型
        obj_grasp = self.object[obj_ind]

        #self.transform是一个dict字典，找到obj_grasp对应的obj_pc
        #实际上，这他妈就是一样的啊！obj_grasp是被抓物体的名称
        #obj_pc也是被抓物体的名称，打印出来，也是同样的名字啊 
        obj_pc=obj_grasp
        #obj_pc = self.transform[obj_grasp][0]

        #查询字典，获取物体obj_grasp对应的抓取姿态数据的地址f_grasp（只有一个，在train文件夹中）
        #如"/home/wgk/win10/data/ycb_grasp/train/003_cracker_box.npy"
        f_grasp = self.d_grasp[obj_grasp]
        #对应物体的ply文件路径
        ply_locate = self.path+"/objects/" + obj_grasp + "/google_512k/nontextured.ply"

        
        #注意，这里self.d_pc[obj_pc]查出来的是一个与目标物体obj_pc对应的多个视角点云文件的路径列表
        #fl_pc也就是一个存放路径的ndarray
        #写成fl_pc = np.array(self.d_pc[obj_grasp])也是一样的啊
        fl_pc = np.array(self.d_pc[obj_pc])
        #print(len(self.d_pc[obj_pc]))

        #把路径进行了打乱
        np.random.shuffle(fl_pc)

        #np.load(f_grasp)读取出来的是目标物体obj_grasp的所有抓取序列构成的ndarray, (目前的大小是140)
        #再[grasp_ind]，就可以找到index对应的那个抓取姿态

        grasp_ind=math.floor(grasp_ind/120)#向下取整
        #print(grasp_ind)
        grasp = np.load(f_grasp)[grasp_ind]

        #取fl_pc这个ndarray中的最后一个值（是一个点云的路径）
        pc = np.load(fl_pc[-1])
        #self.transform[obj_grasp][0] 是被抓目标物体的名称
        # self.transform[obj_grasp][1]是一个标准的齐次变换矩阵
        transform_ = self.transform[obj_grasp][1]
        #print(self.transform[obj_grasp])
        
        #输入夹爪姿态，获取对应姿态下夹爪内部的点云
        #grasp是一个
        grasp_pc = self.collect_pc_(grasp, pc, transform_,ply_locate)

        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            #检测夹爪内部的点云数量是不是足够多
            if len(grasp_pc) > self.grasp_points_num:
                #太多了，就抽检
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                #太少了，就添加一些
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        

        #在这里进行分数的一个加权处理
        score = level_score + refine_score*0.01

        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 1
        else:
            return None

        #如果要求返回物体？
        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            #只返回夹爪内部的点云，以及对应的标签，这里的标签还是0  1 类型的，并没有分类
            return grasp_pc, label

    def __len__(self):
        return self.amount


class PointGraspOneViewMultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, grasp_points_num, grasp_amount_per_file, thresh_good,
                 thresh_bad, path, tag, with_obj=False, projection=False, project_chann=3, project_size=60):
        self.grasp_points_num = grasp_points_num
        self.grasp_amount_per_file = grasp_amount_per_file
        self.path = path
        self.tag = tag
        self.thresh_good = thresh_good
        self.thresh_bad = thresh_bad
        self.with_obj = with_obj
        self.min_point_limit = 50

        # projection related
        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.normal_K = 10
        self.voxel_point_num  = 50
        self.projection_margin = 1
        self.minimum_point_amount = 150

        self.transform = pickle.load(open(os.path.join(self.path, 'google2cloud.pkl'), 'rb'))
        fl_grasp = glob.glob(os.path.join(path, 'ycb_grasp', self.tag, '*.npy'))
        fl_pc = glob.glob(os.path.join(path, 'ycb_rgbd', '*', 'clouds', 'pc_NP3_NP5*.npy'))

        self.d_pc, self.d_grasp = {}, {}
        for i in fl_pc:
            k = i.split('/')[-3]
            if k in self.d_pc.keys():
                self.d_pc[k].append(i)
            else:
                self.d_pc[k] = [i]
        for k in self.d_pc.keys():
            self.d_pc[k].sort()

        for i in fl_grasp:
            k = i.split('/')[-1].split('.')[0]
            self.d_grasp[k] = i
        object1 = set(self.d_grasp.keys())
        object2 = set(self.transform.keys())
        self.object = list(object1.intersection(object2))
        self.amount = len(self.object) * self.grasp_amount_per_file

    def collect_pc(self, grasp, pc, transform):
        center = grasp[0:3]
        axis = grasp[3:6] # binormal
        width = grasp[6]
        angle = grasp[7]

        axis = axis/np.linalg.norm(axis)
        binormal = axis
        # cal approach
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        R1 = np.c_[[cos_t, 0, sin_t],[0, 1, 0],[-sin_t, 0, cos_t]]
        axis_y = axis
        axis_x = np.array([axis_y[1], -axis_y[0], 0])
        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
        approach = R2.dot(R1)[:, 0]
        approach = approach / np.linalg.norm(approach)
        minor_normal = np.cross(axis, approach)

        left = center - width*axis/2
        right = center + width*axis/2
        left = (np.dot(transform, np.array([left[0], left[1], left[2], 1])))[:3]
        right = (np.dot(transform, np.array([right[0], right[1], right[2], 1])))[:3]
        center = (np.dot(transform, np.array([center[0], center[1], center[2], 1])))[:3]
        binormal = (np.dot(transform, np.array([binormal[0], binormal[1], binormal[2], 0])))[:3].reshape(3, 1)
        approach = (np.dot(transform, np.array([approach[0], approach[1], approach[2], 0])))[:3].reshape(3, 1)
        minor_normal = (np.dot(transform, np.array([minor_normal[0], minor_normal[1], minor_normal[2], 0])))[:3].reshape(3, 1)
        matrix = np.hstack([approach, binormal, minor_normal]).T
        pc_t = (np.dot(matrix, (pc-center).T)).T
        left_t = (-width * np.array([0,1,0]) / 2).squeeze()
        right_t = (width * np.array([0,1,0]) / 2).squeeze()

        x_limit = width/4
        z_limit = width/4
        y_limit = width/2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2])
        self.in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(self.in_ind) < self.min_point_limit:
            return None
        if self.projection:
            return self.project_pc(pc_t, width)
        else:
            return pc_t[self.in_ind]

    def check_square(self, point, points_g):
        dirs = np.array([[-1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, -1, 1],
                        [-1, 1, -1], [1, 1, -1], [-1, -1, -1], [1, -1, -1]])
        p = dirs * 0.5 + point  # here res * 0.5 means get half of a pixel width
        a1 = p[2][1] < points_g[:, 1]
        a2 = p[0][1] > points_g[:, 1]
        a3 = p[0][2] > points_g[:, 2]
        a4 = p[4][2] < points_g[:, 2]
        a5 = p[1][0] > points_g[:, 0]
        a6 = p[0][0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
        return points_in_area

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                  "such data, please throw it away.  -- Hongzhuo")
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic-margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1)/number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert(occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        surface_normal = surface_normal[:, 0:3]
        pc = pc.to_array()
        grasp_pc = pc[self.in_ind]
        grasp_pc_norm = surface_normal[self.in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check)!=0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert(np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def __getitem__(self, index):
        # try:
        obj_ind, grasp_ind = np.unravel_index(index, (len(self.object), self.grasp_amount_per_file))

        obj_grasp = self.object[obj_ind]
        obj_pc = self.transform[obj_grasp][0]
        f_grasp = self.d_grasp[obj_grasp]
        fl_pc = np.array(self.d_pc[obj_pc])
        np.random.shuffle(fl_pc)

        grasp = np.load(f_grasp)[grasp_ind]
        pc = np.load(fl_pc[-1])
        t = self.transform[obj_grasp][1]

        grasp_pc = self.collect_pc(grasp, pc, t)
        if grasp_pc is None:
            return None
        level_score, refine_score = grasp[-2:]

        if not self.projection:
            if len(grasp_pc) > self.grasp_points_num:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=False)].T
            else:
                grasp_pc = grasp_pc[np.random.choice(len(grasp_pc), size=self.grasp_points_num,
                                                     replace=True)].T
        else:
            grasp_pc = grasp_pc.transpose((2, 1, 0))
        score = level_score + refine_score*0.01
        if score >= self.thresh_bad:
            label = 0
        elif score <= self.thresh_good:
            label = 2
        else:
            label = 1

        if self.with_obj:
            return grasp_pc, label, obj_grasp
        else:
            return grasp_pc, label

    def __len__(self):
        return self.amount



if __name__ == '__main__':
    grasp_points_num = 1000
    obj_points_num = 50000
    pc_file_used_num = 20
    thresh_good = 0.6
    thresh_bad = 0.6

    input_size = 60
    input_chann = 12  # 12
    a = PointGraspDataset(
        obj_points_num=obj_points_num,
        grasp_points_num=grasp_points_num,
        pc_file_used_num=pc_file_used_num,
        path="../data",
        tag='train',
        grasp_amount_per_file=6500,
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
        projection=True,
        project_chann=input_chann,
        project_size=input_size,
    )
    c, d = a.__getitem__(0)
