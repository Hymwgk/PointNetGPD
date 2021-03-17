#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python3 
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 20/05/2018 2:45 PM 
# File Name  : generate-dataset-canny.py

import numpy as np
import sys
import pickle
from dexnet.grasping.quality import PointGraspMetrics3D
from dexnet.grasping import GaussianGraspSampler, AntipodalGraspSampler, UniformGraspSampler, GpgGraspSampler
from dexnet.grasping import RobotGripper, GraspableObject3D, GraspQualityConfigFactory, PointGraspSampler
import dexnet
from autolab_core import YamlConfig
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
import os

import multiprocessing
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # for the convenient of run on remote computer

#sys.path()

#输入文件夹地址，返回一个列表，其中保存的是文件夹中的文件名称
def get_file_name(file_dir_):  
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        #将下一层子文件夹的地址保存到 file_list 中
        if root.count('/') == file_dir_.count('/') + 1:
            file_list.append(root)
    #排序
    file_list.sort()
    return file_list
"""
\brief: 创建多线程，多线程调用worker函数，处理指定的模型，
"""
def do_job(i):      #处理函数  处理第i个模型
    #根据id号，截取对应的目标模型名称
    object_name = file_list_all[i][len(home_dir) + 35:] 

    """
    good_grasp = multiprocessing.Manager().list()    #列表

    # grasp_amount per friction: 20*40   共开70个子线程做这件事
    p_set = [multiprocessing.Process(target=worker, args=(i, 100, 20, good_grasp)) for _ in
             range(1)]  
    
    #开始多线程，并等待结束
    [p.start() for p in p_set]  
    [p.join() for p in p_set]
    good_grasp = list(good_grasp)   
    """
    #创建空列表
    good_grasp=[]
    #执行worker，将采样的抓取结果放到good_grasp中
    worker(i, 100, 20, good_grasp)
    #将gpg得到的候选抓取文件存放起来
    good_grasp_file_name =  "./generated_grasps/{}_{}_{}".format(filename_prefix, str(object_name), str(len(good_grasp)))
    
    #创建一个pickle文件，将good_grasp保存起来
    with open(good_grasp_file_name + '.pickle', 'wb') as f:
        pickle.dump(good_grasp, f)

    tmp = []
    for grasp in good_grasp:
        grasp_config = grasp[0].configuration
        score_friction = grasp[1]
        score_canny = grasp[2]
        tmp.append(np.concatenate([grasp_config, [score_friction, score_canny]]))
    np.save(good_grasp_file_name + '.npy', np.array(tmp))
    print("finished job ", object_name)

def do_jobs(i):
    print("tesk id", i)

def worker(i, sample_nums, grasp_amount, good_grasp):  #主要是抓取采样器以及打分    100  20
    """
    brief: 对制定的模型，利用随机采样算法，进行抓取姿态的检测和打分
    param [in]  i 处理第i个mesh模型
    param [in]  sample_nums 每个对象模型返回的目标抓取数量
    param [in]  grasp_amount
    """

    #截取目标对象名称
    object_name = file_list_all[i][len(home_dir) + 35:]  
    print('a worker of task {} start'.format(object_name))    

    #读取初始配置文件，读取并在内存中复制了一份
    yaml_config = YamlConfig(home_dir + "/code/PointNetGPD/dex-net/test/config.yaml")
    #设置夹名称
    gripper_name = 'panda' 
    #根据设置的夹爪名称加载夹爪配置
    gripper = RobotGripper.load(gripper_name, home_dir + "/code/PointNetGPD/dex-net/data/grippers") 
    #设置抓取采样的方法
    grasp_sample_method = "antipodal"
    if grasp_sample_method == "uniform":
        ags = UniformGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gaussian":
        ags = GaussianGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "antipodal":
        #使用对映点抓取，输入夹爪与配置文件
        ags = AntipodalGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "gpg":
        ags = GpgGraspSampler(gripper, yaml_config)
    elif grasp_sample_method == "point":
        ags = PointGraspSampler(gripper, yaml_config)
    else:
        raise NameError("Can't support this sampler")
    print("Log: do job", i)
    #设置obj模型文件与sdf文件路径
    if os.path.exists(str(file_list_all[i]) + "/google_512k/nontextured.obj"):
        of = ObjFile(str(file_list_all[i]) + "/google_512k/nontextured.obj")
        sf = SdfFile(str(file_list_all[i]) + "/google_512k/nontextured.sdf")
    else:
        print("can't find any obj or sdf file!")
        raise NameError("can't find any obj or sdf file!")

    #根据路径读取模型与sdf文件
    mesh = of.read()
    sdf = sf.read() 
    #构建抓取模型类
    obj = GraspableObject3D(sdf, mesh)   
    print("Log: opened object", i + 1, object_name)

#########################################
    #设置
    force_closure_quality_config = {}   #设置力闭合  字典
    canny_quality_config = {}
    #生成一个起点是2.0终点是0.75   步长为-0.4  （递减）的等距数列fc_list_sub1 (2.0, 0.75, -0.4) 
    fc_list_sub1 = np.arange(2.0, 0.75, -0.3)   
    #生成一个起点是0.5终点是0.36   步长为-0.05的等距数列fc_list_sub2  (0.5, 0.36, -0.05)
    fc_list_sub2 = np.arange(0.5, 0.36, -0.1)

    #将上面两个向量接起来，变成一个长条向量，使用不同的步长，目的是为了在更小摩擦力的时候，有更多的分辨率
    fc_list = np.concatenate([fc_list_sub1, fc_list_sub2])
    print("判断摩擦系数")
    print(fc_list)
    for value_fc in fc_list:
        #对value_fc保留2位小数，四舍五入
        value_fc = round(value_fc, 2)
        #更改内存中配置中的摩擦系数，而没有修改硬盘中的yaml文件
        yaml_config['metrics']['force_closure']['friction_coef'] = value_fc
        yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc
        #把每个摩擦力值当成键，
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['force_closure'])
        canny_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            yaml_config['metrics']['robust_ferrari_canny'])

    #####################准备开始采样############################
    #填充一个与摩擦数量相同的数组，每个对应的元素都是0
    good_count_perfect = np.zeros(len(fc_list))
    count = 0
    #设置每个摩擦值需要计算的最少抓取数量 （根据指定输入值20）
    minimum_grasp_per_fc = grasp_amount     
    #如果每个摩擦系数下，有效的抓取(满足力闭合或者其他判断标准)小于要求值，就一直循环查找，直到所有摩擦系数条件下至少都存在20个有效抓取
    while np.sum(good_count_perfect < minimum_grasp_per_fc) != 0:    
        #开始使用antipodes sample获得对映随机抓取，此时并不判断是否满足力闭合，只是先采集满足夹爪条件的抓取
        #如果一轮多次随机采样之后，发现无法获得指定数量的随机抓取，就会重复迭代计算3次，之后放弃，并把已经找到的抓取返回来
        grasps = ags.generate_grasps(obj, target_num_grasps=sample_nums, grasp_gen_mult=10,max_iter=10,
                                     vis=False, random_approach_angle=True)
        count += len(grasps)
        #循环对每个采样抓取进行判断
        for j in grasps:    
            tmp, is_force_closure = False, False
            #循环对某个采样抓取应用不同的抓取摩擦系数，判断是否是力闭合
            for ind_, value_fc in enumerate(fc_list):
                value_fc = round(value_fc, 2)
                tmp = is_force_closure
                #判断在当前给定的摩擦系数下，抓取是否是力闭合的
                is_force_closure = PointGraspMetrics3D.grasp_quality(j, obj,
                                                                     force_closure_quality_config[value_fc], vis=False)
                #假设当前,1号摩擦力为1.6 抓取不是力闭合的，但是上一个0号摩擦系数2.0 条件下抓取是力闭合的
                if tmp and not is_force_closure:
                    #当0号2.0摩擦系数条件下采样的good抓取数量还不足指定的最低数量20
                    if good_count_perfect[ind_ - 1] < minimum_grasp_per_fc:
                        #以0号摩擦系数作为边界
                        canny_quality = PointGraspMetrics3D.grasp_quality(j, obj,
                                                                          canny_quality_config[
                                                                              round(fc_list[ind_ - 1], 2)],
                                                                          vis=False)
                        good_grasp.append((j, round(fc_list[ind_ - 1], 2), canny_quality))
                        #在0号系数的good抓取下计数加1
                        good_count_perfect[ind_ - 1] += 1
                    #当前抓取j的边界摩擦系数找到了，退出摩擦循环，判断下一个抓取
                    break
                #如果当前1号摩擦系数1.6条件下，该抓取j本身就是力闭合的，且摩擦系数是列表中的最后一个（所有的摩擦系数都判断完了）
                elif is_force_closure and value_fc == fc_list[-1]:
                    if good_count_perfect[ind_] < minimum_grasp_per_fc:
                        #以当前摩擦系数作为边界
                        canny_quality = PointGraspMetrics3D.grasp_quality(j, obj,
                                                                          canny_quality_config[value_fc], vis=False)
                        good_grasp.append((j, value_fc, canny_quality))
                        good_count_perfect[ind_] += 1
                    #当前抓取j关于当前摩擦系数1.6判断完毕，而且满足所有的摩擦系数，就换到下一个摩擦系数
                    break
        print('Object:{} GoodGrasp:{}'.format(object_name, good_count_perfect))  #判断

    object_name_len = len(object_name)
    object_name_ = str(object_name) + " " * (25 - object_name_len)
    if count == 0:
        good_grasp_rate = 0
    else:
        good_grasp_rate = len(good_grasp) / count
    print('Gripper:{} Object:{} Rate:{:.4f} {}/{}'.
          format(gripper_name, object_name_, good_grasp_rate, len(good_grasp), count))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename_prefix = sys.argv[1]
    else:
        filename_prefix = "default"
    home_dir = os.environ['HOME']
    #存放CAD模型的文件夹
    file_dir = home_dir + "/dataset/ycb_meshes_google/objects"   #获取模型的路径
    file_list_all = get_file_name(file_dir)              #返回一个列表，包含物体
    object_numbers = file_list_all.__len__()       #获取文件夹中物体数量

    job_list = np.arange(object_numbers)   #返回一个长度为object_numbers的元组 0 1 2 3 ... 
    job_list = list(job_list)                                    #转换为列表
    #设置同时对几个模型进行采样
    pool_size = 47
    assert (pool_size <= len(job_list))  
    # Initialize pool
    pool = []     #创建列表
    for _ in range(pool_size):   #想多线程处理多个模型，但是实际上本代码每次只处理一个
        job_i = job_list.pop(0)   #删除掉第0号值，并把job_i赋予0号值
        pool.append(multiprocessing.Process(target=do_job, args=(job_i,)))  #在pool末尾添加元素
    [p.start() for p in pool]                  #启动多线程
    # refill
    while len(job_list) > 0:    #如果有些没处理完
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                job_i = job_list.pop(0)
                p = multiprocessing.Process(target=do_job, args=(job_i,))
                p.start()
                pool.append(p)
                break
    print('All job done.')
