#!/usr/bin/env python
#coding=utf-8
"""
    moveit_ik_demo.py - Version 0.1 2014-01-14
    使得机械臂，先在初始状态，然后移动一下机械臂，然后再回到初始状态，停止
    Use inverse kinemtatics to move the end effector to a specified pose
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyleft (c) 2014 Patrick Goebel.  All lefts reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.5
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
"""

import rospy, sys
import moveit_commander
import tf
import math
import numpy as np
from math import pi

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from autolab_core import RigidTransform

try:
    from gpd_grasp_msgs.msg import GraspConfig
    from gpd_grasp_msgs.msg import GraspConfigList
except ImportError:
    print("Please install grasp msgs from https://github.com/TAMS-Group/gpd_grasp_msgs in your ROS workspace")
    exit()


class MoveItDemo:
   
    def lookupTransform(self,tf_listener, target, source):
        tf_listener.waitForTransform(target, source, rospy.Time(), rospy.Duration(4.0)) #等待时间为10秒

        trans, rot = tf_listener.lookupTransform(target, source, rospy.Time())
        euler = tf.transformations.euler_from_quaternion(rot)

        source_target = tf.transformations.compose_matrix(translate = trans, angles = euler)
        return source_target
    def getTfFromMatrix(self,matrix):
        scale, shear, angles, trans, persp = tf.transformations.decompose_matrix(matrix)
        return trans, tf.transformations.quaternion_from_euler(*angles), angles

    
    def Callback(self,data): 
        """
        根据夹爪最终抓取姿态，结合后撤距离，计算预抓取夹爪的位置与姿态
        """

        #data是GraspConfigList,data.grasps[0]是GraspConfig
        grasp_config=data.grasps[0]

        self.r_flag=True
        self.grasp_pose=PoseStamped()

        #最终抓取时候的夹爪位姿
        self.grasp_pose.pose.position.x= grasp_config.bottom.x
        self.grasp_pose.pose.position.y= grasp_config.bottom.y
        self.grasp_pose.pose.position.z= grasp_config.bottom.z
        
        #从三个抓取坐标系向量轴，转换为矩阵形式
        approach=np.array([grasp_config.approach.x,grasp_config.approach.y,grasp_config.approach.z,0])
        binormal=np.array([grasp_config.binormal.x,grasp_config.binormal.y,grasp_config.binormal.z,0])
        axis=np.array([grasp_config.axis.x,grasp_config.axis.y,grasp_config.axis.z,0])
        t=np.array([0,0,0,1])
        rot_matrix=np.hstack([approach.T,binormal.T,axis.T,t.T]).reshape(4,4)
        #print(rot_matrix)


        #rot_matrix=np.array([grasp_config.approach,grasp_config.binormal,grasp_config.axis]).reshape(3,3)
        #从旋转矩阵  变换到  四元数形式的旋转
        rot_quater=tf.transformations.quaternion_from_matrix(rot_matrix)
        #print(rot_quater)
        self.grasp_pose.pose.orientation.x=rot_quater[0]
        self.grasp_pose.pose.orientation.y=rot_quater[1]
        self.grasp_pose.pose.orientation.z=rot_quater[2]
        self.grasp_pose.pose.orientation.w=rot_quater[3]


        self.grasp_pose.header.frame_id =data.header.frame_id
        self.grasp_pose.header.stamp = rospy.Time()    

        #计算预抓取的位姿
        self.pre_grasp_pose = self.grasp_pose
    

        #设定后撤距离
        retreat_dis=0.15
        #计算原夹爪中心
        grasp_bottom_center=np.array([grasp_config.bottom.x,grasp_config.bottom.y,grasp_config.bottom.z])
        #approach向量
        grasp_approach=np.array([grasp_config.approach.x,grasp_config.approach.y,grasp_config.approach.z])

        #计算预抓取中心
        pre_grasp_bottom_center=grasp_bottom_center - retreat_dis*grasp_approach
        #修正过来
        self.pre_grasp_pose.pose.position.x=pre_grasp_bottom_center[0]
        self.pre_grasp_pose.pose.position.y=pre_grasp_bottom_center[1]
        self.pre_grasp_pose.pose.position.z=pre_grasp_bottom_center[2]







    def __init__(self):
        #关于baxter无法通过moveit获取当前姿态的错误    https://github.com/ros-planning/moveit/issues/1187
        joint_state_topic = ['joint_states:=/robot/joint_states']
        #初始化moveit的 API接口
        moveit_commander.roscpp_initialize(joint_state_topic)


        #初始化ros节点，
        rospy.init_node('baxter_grasp', anonymous=True)



        #创建一个TF监听器
        self.tf_listener = tf.TransformListener()
        #一直等待接收到桌面标签和机器人base坐标系之间的变换（需要提前进行手眼标定）
        get_transform=False
        while not get_transform:
            try:
                #尝试查看机器人基座base与桌面标签之间的转换
                trans, _ = self.tf_listener.lookupTransform('/ar_marker_6', '/base', rospy.Time(0))
                get_transform = True
                rospy.loginfo("got transform complete")
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue


                
        # 创建左臂规划组对象
        self.selected_arm = moveit_commander.MoveGroupCommander('left_arm')
                
        # 获取左臂末端执行器名称
        self.end_effector_link = self.selected_arm.get_end_effector_link()
                        
        # 创建机械臂父坐标系名称字符
        reference_frame = 'torso'

        #self.tf_listener = tf.TransformListener()
        
        # 设置父坐标系名称
        self.selected_arm.set_pose_reference_frame(reference_frame)
        
        # 允许机械臂进行重规划
        #self.selected_arm.allow_replanning(True)
        
        # 允许机械臂末位姿的错误余量
        self.selected_arm.set_goal_position_tolerance(0.01)
        self.selected_arm.set_goal_orientation_tolerance(0.05)
        
        #设置初始姿态
        #Home_positions = [-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50]#移动到工作位置，使用正运动学
        # Set the arm's goal configuration to the be the joint positions
        #self.selected_arm.set_joint_value_target(Home_positions)           
        # Plan and execute the motion
        #self.selected_arm.go()
        #rospy.sleep(5)


         # 设置home姿态
        Home_positions = [-0.08, -1.0, -1.19, 1.94,  0.67, 1.7, -0.50]#移动到工作位置，使用正运动学

        #self.selected_arm.remember_joint_values('resting', joint_positions)#存储当前状态为初始状态
        self.start_state =self.selected_arm.get_current_pose(self.end_effector_link)

        # Set the arm's goal configuration to the be the joint positions
        self.selected_arm.set_joint_value_target(Home_positions)
                 
        # Plan and execute the motion，运动到Home位置
        self.selected_arm.go()

        ######################开始等待接收夹爪姿态#########################
        print("Waiting for gripper pose!")
        #等待gripper_pose这个话题的发布（目标抓取姿态，该姿态将会进行抓取）
        rospy.wait_for_message('/detect_grasps/clustered_grasps', GraspConfigList) 
        #创建消息订阅器，订阅“gripper_pose”话题，这个gripper_pose，是以桌面标签为参考系的
        #接收到之后，调用回调函数，有两件事要做
        # 1.将gripper_pose
        # 计算后撤距离
        self.r_flag=False
        rospy.Subscriber('/detect_grasps/clustered_grasps', GraspConfigList, self.Callback,queue_size=1)

        #能不能，先检查并读取 桌面标签和机器人底座之间的转换关系？（这只是其中一条路吧）
        #利用TF直接，读取grasp_pose在base坐标系下面的姿态
        #调用pose处理函数，计算预抓取姿态，



        #rospy.sleep(5)

        print("Start to Recieve Grasp Config!")




        #####################################################################
        while not rospy.is_shutdown():
            if self.r_flag:
                #目标姿态设定为预抓取位置
                target_pose = self.pre_grasp_pose
                self.r_flag=False
                #尝试利用TF直接，读取grasp_pose在base坐标系下面的姿态
                #计算预抓取姿态
            else:
                rospy.sleep(0.5)
                continue


            #以当前姿态作为规划起始点
            self.selected_arm.set_start_state_to_current_state()  
            # 对末端执行器姿态设定目标姿态
            #self.selected_arm.set_pose_target(target_pose, 'left_gripper')
            
            # 规划轨迹
            #traj = self.selected_arm.plan(target_pose.pose)
            
            # 执行轨迹，运行到预抓取位置
            #self.selected_arm.execute(traj)
            print(self.end_effector_link)


            print('Moving to pre_grasp_pose')

            self.selected_arm.plan(target_pose.pose)
            continue



            #print(target_pose.pose)
            success=self.selected_arm.go(target_pose.pose,wait=True)

            if not success:
                print('Failed to move to pre_grasp_pose!')
                continue
            
            print('Move to pre_grasp_pose succeed')
            #等待机械臂稳定
            rospy.sleep(1)
            #再设置当前姿态为起始姿态
            self.selected_arm.set_start_state_to_current_state()  
            #
            waypoints = [self.grasp_pose.pose]
            #wpose = self.selected_arm.get_current_pose().pose
            #wpose.position.z -= scale * 0.1

            #规划从当前位姿，保持姿态，转移到目标夹爪姿态的路径
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step
                0.0)         # jump_threshold

            #执行,并等待这个轨迹执行成功
            self.selected_arm.execute(plan,wait=True)


            #执行抓取

            ####################抓取完后撤####################
            waypoints = [self.pre_grasp_pose.pose]
            
            #规划从当前位姿，保持姿态，转移到目标夹爪姿态的路径
            (plan, fraction) = move_group.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step
                0.0)         # jump_threshold

            #执行,并等待后撤成功
            self.selected_arm.execute(plan,wait=True)
           

            ######################暂时设置直接回到Home############################

            #self.selected_arm.remember_joint_values('resting', joint_positions)#存储当前状态为初始状态
            self.start_state =self.selected_arm.get_current_pose(self.end_effector_link)
            
            # Set the arm's goal configuration to the be the joint positions
            self.selected_arm.set_joint_value_target(Home_positions)
                    
            # Plan and execute the motion，运动到Home位置
            self.selected_arm.go()
            rospy.sleep(5)


        # Shut down MoveIt cleanly
        moveit_commander.roscpp_shutdown()
        
        # Exit MoveIt
        moveit_commander.os._exit(0)

if __name__ == "__main__":
    try:
        MoveItDemo()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Arm tracker node terminated.")

    
    