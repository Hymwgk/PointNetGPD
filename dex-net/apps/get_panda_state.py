#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 主要是为了获取机械臂当前的状态，通过ros实时获取夹爪当前的姿态，看夹爪是否位于home状态
#如果使用panda等机器人，就需要直接在shell中运行这个程序，不能在anconda中运行，其实也不是不可以，但是每个都要提前运行一个
#panda_clinet.sh脚本
# Date       : 08/09/2018 12:00 AM
# File Name  : get_ur5_robot_state.py
#python 2

import rospy
import numpy as np
import moveit_commander
import sys

def get_robot_state_moveit():
    # moveit_commander.roscpp_initialize(sys.argv)
    # robot = moveit_commander.RobotCommander()
    try:
        #获取当前机械臂的每个关节的角度，构建成为一个ndarry
        current_joint_values = np.array(group.get_current_joint_values())
        #计算每个关节的角度的所有的距离差值的绝对值之和
        diff = abs(current_joint_values - home_joint_values)*180/np.pi
        #要和关节数量相同，共7个关节
        if np.sum(diff<1) ==7:  # if current joint - home position < 1 degree, we think it is at home
            return 1  # robot at home
        else:
            print(np.sum(diff<1))
            return 2  # robot is moving

    except:
        return 3  # robot state unknow
        rospy.loginfo("Get robot state failed")

if __name__ == '__main__':
    
    #joint_state_topic = ['joint_states:=/robot/joint_states']
    rospy.init_node('panda_state_checker_if_it_at_home', anonymous=True)
    rate = rospy.Rate(10)
    moveit_commander.roscpp_initialize(sys.argv)
    #想要使用moveit，就必须要提前上传ros关于机械臂的参数等等
    group = moveit_commander.MoveGroupCommander("panda_arm")
    home_joint_values = np.array([0.0399, -0.6999, 0.1798, -2.8000,  0.1897, 2.1302, 0.9199])
    while not rospy.is_shutdown():
        #获取机械臂当前状态
        at_home = get_robot_state_moveit()
        print(group.get_current_joint_values())
        if at_home == 1:
            rospy.set_param("/robot_at_home", "true")
            rospy.loginfo("robot at home")
        elif at_home == 2:
            rospy.set_param("/robot_at_home", "false")
            rospy.loginfo("robot is moving")
        elif at_home == 3:
            rospy.loginfo("robot state unknow")
        rate.sleep()
