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

import copy
from moveit_msgs.msg import RobotTrajectory,DisplayTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from autolab_core import RigidTransform,transformations
from pyquaternion import Quaternion
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
        self.grasp_config=data.grasps[0]

        self.r_flag=True
        self.grasp_pose=Pose()
        self.pre_grasp_pose=Pose()

        
        #从三个抓取坐标系向量轴，转换为矩阵形式
        approach=-1*np.array([self.grasp_config.approach.x,self.grasp_config.approach.y,self.grasp_config.approach.z])#抓取轴
        approach=approach/np.linalg.norm(approach)
        binormal=-1*np.array([self.grasp_config.binormal.x,self.grasp_config.binormal.y,self.grasp_config.binormal.z])#合并轴
        binormal=binormal/np.linalg.norm(binormal)
        axis=np.array([self.grasp_config.axis.x,self.grasp_config.axis.y,self.grasp_config.axis.z])#撸轴
        axis=axis/np.linalg.norm(axis)

        trans=np.array([self.grasp_config.bottom.x,self.grasp_config.bottom.y,self.grasp_config.bottom.z])#原点
        scale=np.array([0,0,0,1])
        #print(approach)
        #print(binormal)
        #print(axis)




        rot_trans=np.hstack([axis,binormal,approach,trans]).reshape(4,3).T
        #rot_trans=np.hstack([approach,binormal,axis,trans]).reshape(4,3).T

        marker2gripper=np.vstack([rot_trans,scale])
        #print(marker2gripper)


        test=np.array([0,0,0])
        rot_test=np.hstack([axis,binormal,approach,test]).reshape(4,3).T
        rot_test=np.vstack([rot_test,scale])

        #print(marker2gripper)
        #构成了一个标准的4*4齐次变换矩阵，代表了夹爪抓取坐标系在marker坐标系中的位置（1平移2旋转）
        #grasp_matrix=np.hstack([approach.T,binormal.T,axis.T,t.T]).reshape(4,4)

        """
        #因为上面的夹爪坐标系是marker到graspper的，现在想要grasper到marker的
        rot=np.hstack([approach.T,binormal.T,axis.T]).reshape(3,3)
        rot_=rot.T  #取逆
        print(rot_)
        #print(t_)
        t_=-t.dot(rot).reshape(3,1)  #见机器人学导论p42~p43
        scale=np.array([0,0,0,1])
        grapper2marker=np.vstack([np.hstack([rot_,t_]),scale])
        """
        base2gripper=self.base2marker.dot(marker2gripper)
        base2grasplink8=base2gripper.dot(self.gripper2link8)
        #print(self.base2marker)

        #检查最终的抓取姿态是否过于“绕”
        axis_check=base2grasplink8[0:3,0].dot(self.base2Initial_link8[0:3,0])
        #bin_check=base2grasplink8[0:3,0].dot(self.base2Initial_link8[0:3,0])
        base2grasplink8_test=copy.deepcopy(base2grasplink8)
        if axis_check>0:
            base2grasplink8[0:3,0]=-1*base2grasplink8[0:3,0]
            base2grasplink8[0:3,1]=-1*base2grasplink8[0:3,1]





        #rot_matrix=np.array([self.grasp_config.approach,self.grasp_config.binormal,self.grasp_config.axis]).reshape(3,3)
        #从旋转矩阵  变换到  四元数形式的旋转
        #rot_quater=tf.transformations.quaternion_from_matrix(base2grasplink8)

        #上下两种计算出来的结果是不一样的，这个的方向总是错的
        scale_, shear_, angles, trans_, persp_ = tf.transformations.decompose_matrix(base2grasplink8)
        rot_quater=tf.transformations.quaternion_from_euler(*angles)
        #上面计算出的夹爪坐标系的旋转部分总是出错，于是需要对计算出的坐标系进行调整
        rot_matrix_modifiy=tf.transformations.quaternion_matrix(rot_quater)
        #print("rot_matrix_modifiy")
        #print(rot_matrix_modifiy)
        #approach_modifiy=rot_matrix_modifiy[0:3,2].reshape(1,3)
        #binormal_modifiy=rot_matrix_modifiy[0:3,1].reshape(1,3)
        #axis_modifiy=rot_matrix_modifiy[0:3,0].reshape(1,3)
        #approach_standar=np.array([0,0,-1])
        #temp=approach_modifiy.dot(approach_standar)
        #if temp<0:
            #approach_modifiy=-1*approach_modifiy
        
        #rot_matrix_modifiy=np.vstack([axis_modifiy,binormal_modifiy,approach_modifiy,[0,0,0]]).reshape(4,3).T
        #rot_matrix_modifiy=np.vstack([rot_matrix_modifiy,scale])




        #下面的这个看着方向是对的，但是很不准
        #rot_quater_test=tf.transformations.quaternion_from_matrix(marker2gripper)
        #rot_quater_test=rot_quater_test/np.linalg.norm(rot_quater_test)
        #rot_test=base2grasplink8[0:3,0:3].reshape(3,3)
        #rot_quater_test=Quaternion(matrix=rot_test)


        #因为这个矩阵是构造出来的，而不是从tf中读出来的，就出发了一个bug感觉
        scale_, shear_, angles_test, trans_, persp_ = tf.transformations.decompose_matrix(base2grasplink8_test)
        rot_quater_test=tf.transformations.quaternion_from_euler(*angles_test)

        #rot_quater_test=tf.transformations.quaternion_from_matrix(marker2gripper)
        #rot_quater_test=rot_quater_test/np.linalg.norm(rot_quater_test)
        #rot_test=marker2gripper[0:3,0:3].reshape(3,3)
        #print(rot_test)
        #rot_quater_test=Quaternion(matrix=rot_test)






        #print(rot_quater)
        self.grasp_pose.orientation.x=rot_quater[0]  
        self.grasp_pose.orientation.y=rot_quater[1]
        self.grasp_pose.orientation.z=rot_quater[2]
        self.grasp_pose.orientation.w=rot_quater[3]

        #计算预抓取的位姿
        #self.pre_grasp_pose = self.grasp_pose
    

        #设定后撤距离
        retreat_dis=0.15
        #计算原夹爪中心
        #grasp_bottom_center=np.array([self.grasp_config.bottom.x,self.grasp_config.bottom.y,self.grasp_config.bottom.z])
        grasp_bottom_center=base2grasplink8[0:3,3].T

        #approach向量
        grasp_approach=base2grasplink8[0:3,2]
        #print(grasp_approach)
        #print(grasp_bottom_center)

        grasp_bottom_center+=0.09*grasp_approach
        #计算预抓取中心
        pre_grasp_bottom_center=grasp_bottom_center + retreat_dis*grasp_approach
        #print(pre_grasp_bottom_center)
        #修正过来
        self.grasp_pose.position.x= grasp_bottom_center[0]
        self.grasp_pose.position.y= grasp_bottom_center[1]
        self.grasp_pose.position.z= grasp_bottom_center[2]

        self.pre_grasp_pose.position.x=pre_grasp_bottom_center[0]
        self.pre_grasp_pose.position.y=pre_grasp_bottom_center[1]
        self.pre_grasp_pose.position.z=pre_grasp_bottom_center[2]

        self.pre_grasp_pose.orientation.x=rot_quater[0]  
        self.pre_grasp_pose.orientation.y=rot_quater[1]
        self.pre_grasp_pose.orientation.z=rot_quater[2]
        self.pre_grasp_pose.orientation.w=rot_quater[3]


        self.tf_broadcaster.sendTransform(
            pre_grasp_bottom_center,
            rot_quater,
            rospy.Time.now(),
            "pre_grasp_pose",
            "panda_link0")
        self.tf_broadcaster.sendTransform(
            grasp_bottom_center,
            rot_quater,
            rospy.Time.now(),
            "grasp_pose",
            "panda_link0")

        self.tf_broadcaster.sendTransform(
            #marker2gripper[0:3,3].T,
            pre_grasp_bottom_center,
            rot_quater_test,
            rospy.Time.now(),
            "test",
            "panda_link0")


        #print("#####grasp_bottom_center")
        #print(grasp_bottom_center)
        #print("#####pre_grasp_bottom_center")
        #print(pre_grasp_bottom_center)

        #print("#####self.grasp_pose.position1")
        #print(self.grasp_pose.position)
        #print("#####self.pre_grasp_pose.position1")
        #print(self.pre_grasp_pose.position)


    def scale_trajectory_speed(self,traj,spd=0.1):
        new_traj = RobotTrajectory()
        new_traj = traj

        n_joints = len(traj.joint_trajectory.joint_names)
        n_points = len(traj.joint_trajectory.points)

        #spd = 3.0

        points = list(traj.joint_trajectory.points)

        for i in range(n_points):
            point = JointTrajectoryPoint()
            point.time_from_start = traj.joint_trajectory.points[i].time_from_start / spd
            point.velocities = list(traj.joint_trajectory.points[i].velocities)
            point.accelerations = list(traj.joint_trajectory.points[i].accelerations)
            point.positions = traj.joint_trajectory.points[i].positions

            for j in range(n_joints):
                point.velocities[j] = point.velocities[j] * spd
                point.accelerations[j] = point.accelerations[j] * spd

            points[i] = point

        new_traj.joint_trajectory.points = points     
        return   new_traj


    def __init__(self):
        #关于baxter无法通过moveit获取当前姿态的错误    https://github.com/ros-planning/moveit/issues/1187
        #joint_state_topic = ['joint_states:=/robot/joint_states']
        #初始化moveit的 API接口
        moveit_commander.roscpp_initialize(sys.argv)


        #初始化ros节点，
        rospy.init_node('panda_grasp', anonymous=True)
        #构建一个tf发布器
        self.tf_broadcaster=tf.TransformBroadcaster()

        self.grasp_config=GraspConfig()

        #创建一个TF监听器
        self.tf_listener = tf.TransformListener()
        #一直等待接收到桌面标签和机器人base坐标系之间的变换（需要提前进行手眼标定）
        get_transform=False
        while not get_transform:
            try:
                #尝试查看机器人基座base与桌面标签之间的转换
                trans, rot = self.tf_listener.lookupTransform('/panda_link0', '/ar_marker_6', rospy.Time(0))
                euler = tf.transformations.euler_from_quaternion(rot)
                self.base2marker = tf.transformations.compose_matrix(translate = trans, angles = euler)
                #查看gripper到link8之间的变换
                trans, rot = self.tf_listener.lookupTransform( '/panda_EE', '/panda_link8',rospy.Time(0))
                euler = tf.transformations.euler_from_quaternion(rot)
                self.gripper2link8 = tf.transformations.compose_matrix(translate = trans, angles = euler)
                #查看base到panda_link8的变换，此时就是查询gripper的初始姿态
                trans, rot = self.tf_listener.lookupTransform( '/panda_link0', '/panda_link8',rospy.Time(0))
                euler = tf.transformations.euler_from_quaternion(rot)
                self.base2Initial_link8 = tf.transformations.compose_matrix(translate = trans, angles = euler)


                get_transform = True
                rospy.loginfo("got transform complete")
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("got transform failed")
                rospy.sleep(0.5)
                continue


        # 初始化场景对象
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(2)
        # 创建机械臂规划组对象
        self.selected_arm = moveit_commander.MoveGroupCommander('panda_arm')
        #创建机械手规划对象
        hand_group=moveit_commander.MoveGroupCommander('hand')

        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               DisplayTrajectory,
                                               queue_size=20)

        # 获取左臂末端执行器名称
        self.end_effector_link = self.selected_arm.get_end_effector_link()
        print("i am here")
        print(self.end_effector_link)
                        
        # 创建机械臂父坐标系名称字符
        reference_frame = 'panda_link0'

        #self.tf_listener = tf.TransformListener()
        
        # 设置父坐标系名称
        #self.selected_arm.set_pose_reference_frame(reference_frame)
        
        # 允许机械臂进行重规划
        #self.selected_arm.allow_replanning(True)
        
        # 允许机械臂末位姿的错误余量
        self.selected_arm.set_goal_position_tolerance(0.01)
        self.selected_arm.set_goal_orientation_tolerance(0.05)

        #不允许规划失败重规划,规划时间只允许5秒钟,否则很浪费时间
        self.selected_arm.allow_replanning(False)
        self.selected_arm.set_planning_time(5)

        #清除之前遗留的物体
        scene.remove_world_object('table') 
        #设置桌面高度
        table_ground = 0.6
        #桌面尺寸      x  y   z
        table_size = [0.6, 1.2, 0.01]

        # 将table加入场景当中
        table_pose = PoseStamped()
        table_pose.header.frame_id = 'panda_link0'
        table_pose.pose.position.x = 0.55
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = 0.025
        table_pose.pose.orientation.w = 1.0
        scene.add_box('table', table_pose, table_size)
        
        #设置初始姿态
        #Home_positions = [-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50]#移动到工作位置，使用正运动学
        # Set the arm's goal configuration to the be the joint positions
        #self.selected_arm.set_joint_value_target(Home_positions)           
        # Plan and execute the motion
        #self.selected_arm.go()
        #rospy.sleep(5)


        # 设置home姿态
        Home_positions = [0.04, -0.70, 0.18, -2.80,  0.19, 2.13, 0.92]#移动到工作位置，使用正运动学

        #self.selected_arm.remember_joint_values('resting', joint_positions)#存储当前状态为初始状态
        #self.start_state =self.selected_arm.get_current_pose()

        # Set the arm's goal configuration to the be the joint positions
        self.selected_arm.set_joint_value_target(Home_positions)
                 
        # Plan and execute the motion，运动到Home位置
        self.selected_arm.go()
        self.selected_arm.stop()
        


        joint_goal = hand_group.get_current_joint_values()
        joint_goal[0] = 0.04
        joint_goal[1] = 0.04
        hand_group.go(joint_goal, wait=True)
        hand_group.stop()

        ######################开始等待接收夹爪姿态#########################
        print("Waiting for gripper pose!")
        #等待gripper_pose这个话题的发布（目标抓取姿态，该姿态将会进行抓取）
        #rospy.wait_for_message('/detect_grasps/clustered_grasps', GraspConfigList) 
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
                #self.pre_grasp_pose
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
            #self.selected_arm.pick("test",self.grasp_config,plan_only = True)
            #traj=self.selected_arm.plan(self.pre_grasp_pose)
            #self.selected_arm.set_pose_target(self.pre_grasp_pose,end_effector_link="panda_EE")
            #traj=self.selected_arm.plan()

            #continue

            #success=self.selected_arm.execute(traj)

            #print(target_pose.pose)
            #设置规划
            #self.selected_arm.set_planning_time(5)
            success=self.selected_arm.go(self.pre_grasp_pose,wait=True)
            self.selected_arm.stop()
            self.selected_arm.clear_pose_targets()

            
            
            if not success:
                print('Failed to move to pre_grasp_pose!')
                continue
            
            print('Move to pre_grasp_pose succeed')
            #等待机械臂稳定
            rospy.sleep(1)
            #再设置当前姿态为起始姿态
            self.selected_arm.set_start_state_to_current_state()  
            #
            waypoints = []
            wpose=self.selected_arm.get_current_pose().pose
            #print("#####wpose.position")
            #print(wpose.position)
            #print("#####self.grasp_pose2")
            #print(self.grasp_pose.position)
            wpose.position.x=  self.grasp_pose.position.x
            wpose.position.y=  self.grasp_pose.position.y
            wpose.position.z=  self.grasp_pose.position.z

            waypoints.append(copy.deepcopy(wpose))
            #wpose = self.selected_arm.get_current_pose().pose
            #wpose.position.z -= scale * 0.1

            #规划从当前位姿，保持姿态，转移到目标夹爪姿态的路径
            (plan, fraction) = self.selected_arm.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step
                0.0)         # jump_threshold
             ##显示轨迹
            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = self.selected_arm.get_current_state()
            display_trajectory.trajectory.append(plan)
            # Publish
            display_trajectory_publisher.publish(display_trajectory)

            #执行,并等待这个轨迹执行成功
            new_plan=self.scale_trajectory_speed(plan,0.3)
            self.selected_arm.execute(new_plan,wait=True)
            #self.selected_arm.shift_pose_target(2,0.05,"panda_link8")
            #self.selected_arm.go()


            #执行抓取
            rospy.sleep(2)
            print("Grasping")
            joint_goal = hand_group.get_current_joint_values()
            joint_goal[0] = 0.015
            joint_goal[1] = 0.015
            #plan=hand_group.plan(joint_goal)
            #new_plan=self.scale_trajectory_speed(plan,0.3)
            hand_group.go(joint_goal,wait=True)
            hand_group.stop()

            ####################抓取完后撤####################
            waypoints = []
            wpose=self.selected_arm.get_current_pose().pose
            
            wpose.position.x=  self.pre_grasp_pose.position.x
            wpose.position.y=  self.pre_grasp_pose.position.y
            wpose.position.z=  self.pre_grasp_pose.position.z

            waypoints.append(copy.deepcopy(wpose))
            
            #规划从当前位姿，保持姿态，转移到目标夹爪姿态的路径
            (plan, fraction) = self.selected_arm.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step
                0.0)         # jump_threshold

            #执行,并等待后撤成功
            new_plan=self.scale_trajectory_speed(plan,0.6)
            self.selected_arm.execute(new_plan,wait=True)
            """
            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = self.selected_arm.get_current_state()
            display_trajectory.trajectory.append(plan)
            # Publish
            display_trajectory_publisher.publish(display_trajectory)
            """

            ######################暂时设置直接回到Home############################

            #self.selected_arm.remember_joint_values('resting', joint_positions)#存储当前状态为初始状态
            #self.start_state =self.selected_arm.get_current_pose(self.end_effector_link)
            
            # Set the arm's goal configuration to the be the joint positions
            self.selected_arm.set_joint_value_target(Home_positions)
                    
            # Plan and execute the motion，运动到Home位置
            self.selected_arm.go()
            self.selected_arm.stop()

            joint_goal = hand_group.get_current_joint_values()
            joint_goal[0] = 0.04
            joint_goal[1] = 0.04
            hand_group.go(joint_goal, wait=True)
            hand_group.stop()

            print("Grasp done")

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

    
    