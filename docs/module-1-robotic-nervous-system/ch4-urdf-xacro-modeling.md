-----
title: Ch4  URDF & Xacro for Humanoid Modeling
module: 1
chapter: 4
sidebar_label: Ch4: URDF & Xacro for Humanoid Modeling
description: Creating humanoid robot models using URDF and Xacro in ROS 2
tags: [ros2, urdf, xacro, modeling, humanoid, robot]
difficulty: intermediate
estimated_duration: 90
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# URDF & Xacro for Humanoid Modeling

## Learning Outcomes
 Understand the Unified Robot Description Format (URDF) for robot modeling
 Create complex humanoid robot models using URDF
 Utilize Xacro macros to simplify and parameterize robot models
 Implement joint constraints and physical properties for realistic simulation
 Validate robot models using ROS 2 tools
 Integrate robot models with ROS 2 simulation environments

## Theory

### URDF (Unified Robot Description Format)

URDF is an XML format used to describe robots in ROS. It defines the robot's physical structure including links (rigid bodies), joints (connections between links), and other properties like visual appearance and collision properties.

Key components of a URDF:
 **Links**: Rigid bodies that represent parts of the robot
 **Joints**: Connections between links that define how they move relative to each other
 **Visual**: How the robot appears in simulation and visualization tools
 **Collision**: How the robot interacts physically with the environment
 **Inertial**: Mass and inertia properties for physics simulation

### Xacro (XML Macros)

Xacro is an XML macro language that simplifies URDF creation by allowing:
 Parameterization of robot models
 Reusable components through macros
 Mathematical expressions
 File inclusion

Xacro files have the `.xacro` extension and are preprocessed to generate URDF.

### Humanoid Robot Topology

<MermaidDiagram chart={`
graph TD;
    A[Torso] > B[Head];
    A > C[Left Arm];
    A > D[Right Arm];
    A > E[Left Leg];
    A > F[Right Leg];
    
    C > G[Left Shoulder];
    G > H[Left Elbow];
    H > I[Left Hand];
    
    D > J[Right Shoulder];
    J > K[Right Elbow];
    K > L[Right Hand];
    
    E > M[Left Hip];
    M > N[Left Knee];
    N > O[Left Foot];
    
    F > P[Right Hip];
    P > Q[Right Knee];
    Q > R[Right Foot];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style B fill:#2196F3,stroke:#0D47A1,color:#fff;
    style C fill:#FF9800,stroke:#E65100,color:#fff;
    style D fill:#FF9800,stroke:#E65100,color:#fff;
    style E fill:#9C27B0,stroke:#4A148C,color:#fff;
    style F fill:#9C27B0,stroke:#4A148C,color:#fff;
`} />

Humanoid robots have a complex kinematic structure that mimics human anatomy, requiring careful modeling of multiple degrees of freedom in limbs.

## StepbyStep Labs

### Lab 1: Creating a Simple URDF Model

1. **Create a robot description package**:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create buildtype ament_cmake robot_description
   cd robot_description
   ```

2. **Create a URDF file** (`urdf/simple_humanoid.urdf`):
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_humanoid">
     <! Base Link >
     <link name="base_link">
       <visual>
         <geometry>
           <capsule length="0.2" radius="0.1"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <capsule length="0.2" radius="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="10.0"/>
         <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
       </inertial>
     </link>
     
     <! Head >
     <link name="head">
       <visual>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
         <material name="white">
           <color rgba="1 1 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="2.0"/>
         <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.008"/>
       </inertial>
     </link>
     
     <! Joint connecting head to torso >
     <joint name="neck_joint" type="revolute">
       <parent link="base_link"/>
       <child link="head"/>
       <origin xyz="0 0 0.2" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="1.0" upper="1.0" effort="100" velocity="1"/>
     </joint>
     
     <! Left Arm >
     <link name="left_upper_arm">
       <visual>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
         <material name="gray">
           <color rgba="0.5 0.5 0.5 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.0075" ixy="0.0" ixz="0.0" iyy="0.0075" iyz="0.0" izz="0.00125"/>
       </inertial>
     </link>
     
     <joint name="left_shoulder_joint" type="revolute">
       <parent link="base_link"/>
       <child link="left_upper_arm"/>
       <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="1.57" upper="1.57" effort="100" velocity="1"/>
     </joint>
   </robot>
   ```

3. **Install the URDF file** by updating `CMakeLists.txt`:
   ```cmake
   cmake_minimum_required(VERSION 3.8)
   project(robot_description)
   
   find_package(ament_cmake REQUIRED)
   
   # Install URDF files
   install(DIRECTORY urdf
     DESTINATION share/${PROJECT_NAME})
   
   ament_package()
   ```

### Lab 2: Converting to Xacro

1. **Create a Xacro file** (`urdf/humanoid.xacro`):
   ```xml
   <?xml version="1.0"?>
   <robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">
     <! Define properties >
     <xacro:property name="M_PI" value="3.1415926535897931"/>
     <xacro:property name="base_mass" value="10.0"/>
     <xacro:property name="head_mass" value="2.0"/>
     <xacro:property name="arm_mass" value="1.0"/>
     <xacro:property name="leg_mass" value="3.0"/>
     
     <! Base Link >
     <link name="base_link">
       <visual>
         <geometry>
           <capsule length="0.2" radius="0.1"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <capsule length="0.2" radius="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="${base_mass}"/>
         <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
       </inertial>
     </link>
     
     <! Head Macro >
     <xacro:macro name="head" params="parent xyz rpy">
       <joint name="neck_joint" type="revolute">
         <parent link="${parent}"/>
         <child link="head"/>
         <origin xyz="${xyz}" rpy="${rpy}"/>
         <axis xyz="0 1 0"/>
         <limit lower="1.0" upper="1.0" effort="100" velocity="1"/>
       </joint>
       
       <link name="head">
         <visual>
           <geometry>
             <sphere radius="0.1"/>
           </geometry>
           <material name="white">
             <color rgba="1 1 1 1"/>
           </material>
         </visual>
         <collision>
           <geometry>
             <sphere radius="0.1"/>
           </geometry>
         </collision>
         <inertial>
           <mass value="${head_mass}"/>
           <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.008"/>
         </inertial>
       </link>
     </xacro:macro>
     
     <! Arm Macro >
     <xacro:macro name="arm" params="side parent xyz rpy">
       <joint name="${side}_shoulder_joint" type="revolute">
         <parent link="${parent}"/>
         <child link="${side}_upper_arm"/>
         <origin xyz="${xyz}" rpy="${rpy}"/>
         <axis xyz="0 1 0"/>
         <limit lower="1.57" upper="1.57" effort="100" velocity="1"/>
       </joint>
       
       <link name="${side}_upper_arm">
         <visual>
           <geometry>
             <cylinder length="0.3" radius="0.05"/>
           </geometry>
           <material name="gray">
             <color rgba="0.5 0.5 0.5 1"/>
           </material>
         </visual>
         <collision>
           <geometry>
             <cylinder length="0.3" radius="0.05"/>
           </geometry>
         </collision>
         <inertial>
           <mass value="${arm_mass}"/>
           <inertia ixx="0.0075" ixy="0.0" ixz="0.0" iyy="0.0075" iyz="0.0" izz="0.00125"/>
         </inertial>
       </link>
       
       <joint name="${side}_elbow_joint" type="revolute">
         <parent link="${side}_upper_arm"/>
         <child link="${side}_lower_arm"/>
         <origin xyz="0 0 0.3" rpy="0 0 0"/>
         <axis xyz="1 0 0"/>
         <limit lower="1.57" upper="1.57" effort="100" velocity="1"/>
       </joint>
       
       <link name="${side}_lower_arm">
         <visual>
           <geometry>
             <cylinder length="0.25" radius="0.04"/>
           </geometry>
           <material name="gray">
             <color rgba="0.5 0.5 0.5 1"/>
           </material>
         </visual>
         <collision>
           <geometry>
             <cylinder length="0.25" radius="0.04"/>
           </geometry>
         </collision>
         <inertial>
           <mass value="${arm_mass * 0.7}"/>
           <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
         </inertial>
       </link>
     </xacro:macro>
     
     <! Leg Macro >
     <xacro:macro name="leg" params="side parent xyz rpy">
       <joint name="${side}_hip_joint" type="revolute">
         <parent link="${parent}"/>
         <child link="${side}_upper_leg"/>
         <origin xyz="${xyz}" rpy="${rpy}"/>
         <axis xyz="0 1 0"/>
         <limit lower="1.0" upper="1.0" effort="200" velocity="1"/>
       </joint>
       
       <link name="${side}_upper_leg">
         <visual>
           <geometry>
             <cylinder length="0.4" radius="0.06"/>
           </geometry>
           <material name="brown">
             <color rgba="0.6 0.4 0.2 1"/>
           </material>
         </visual>
         <collision>
           <geometry>
             <cylinder length="0.4" radius="0.06"/>
           </geometry>
         </collision>
         <inertial>
           <mass value="${leg_mass}"/>
           <inertia ixx="0.016" ixy="0.0" ixz="0.0" iyy="0.016" iyz="0.0" izz="0.0018"/>
         </inertial>
       </link>
       
       <joint name="${side}_knee_joint" type="revolute">
         <parent link="${side}_upper_leg"/>
         <child link="${side}_lower_leg"/>
         <origin xyz="0 0 0.4" rpy="0 0 0"/>
         <axis xyz="1 0 0"/>
         <limit lower="0.0" upper="1.57" effort="200" velocity="1"/>
       </joint>
       
       <link name="${side}_lower_leg">
         <visual>
           <geometry>
             <cylinder length="0.35" radius="0.05"/>
           </geometry>
           <material name="brown">
             <color rgba="0.6 0.4 0.2 1"/>
           </material>
         </visual>
         <collision>
           <geometry>
             <cylinder length="0.35" radius="0.05"/>
           </geometry>
         </collision>
         <inertial>
           <mass value="${leg_mass * 0.8}"/>
           <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.0015"/>
         </inertial>
       </link>
       
       <joint name="${side}_ankle_joint" type="revolute">
         <parent link="${side}_lower_leg"/>
         <child link="${side}_foot"/>
         <origin xyz="0 0 0.35" rpy="0 0 0"/>
         <axis xyz="0 1 0"/>
         <limit lower="0.5" upper="0.5" effort="100" velocity="1"/>
       </joint>
       
       <link name="${side}_foot">
         <visual>
           <geometry>
             <box size="0.2 0.1 0.05"/>
           </geometry>
           <material name="black">
             <color rgba="0 0 0 1"/>
           </material>
         </visual>
         <collision>
           <geometry>
             <box size="0.2 0.1 0.05"/>
           </geometry>
         </collision>
         <inertial>
           <mass value="${leg_mass * 0.3}"/>
           <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0017" iyz="0.0" izz="0.0013"/>
         </inertial>
       </link>
     </xacro:macro>
     
     <! Use the macros to build the robot >
     <xacro:head parent="base_link" xyz="0 0 0.2" rpy="0 0 0"/>
     <xacro:arm side="left" parent="base_link" xyz="0.15 0 0.1" rpy="0 0 0"/>
     <xacro:arm side="right" parent="base_link" xyz="0.15 0 0.1" rpy="0 0 0"/>
     <xacro:leg side="left" parent="base_link" xyz="0.07 0 0.2" rpy="0 0 0"/>
     <xacro:leg side="right" parent="base_link" xyz="0.07 0 0.2" rpy="0 0 0"/>
   </robot>
   ```

### Lab 3: Validating the Robot Model

1. **Check the URDF** with xacro:
   ```bash
   # Convert xacro to URDF
   ros2 run xacro xacro inorder urdf/humanoid.xacro > temp.urdf
   
   # Check for errors
   check_urdf temp.urdf
   ```

2. **Visualize the robot** using RViz:
   ```bash
   # Launch RViz with robot model
   ros2 run rviz2 rviz2 d `ros2 pkg prefix robot_description`/share/robot_description/rviz/humanoid.rviz
   ```

3. **Create a launch file** (`launch/display.launch.py`):
   ```python
   import os
   from ament_index_python.packages import get_package_share_directory
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   
   def generate_launch_description():
       # Find the package share directory
       pkg_share = get_package_share_directory('robot_description')
       
       # Create launch configuration variables
       use_sim_time = LaunchConfiguration('use_sim_time')
       
       # Declare launch arguments
       declare_use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true')
       
       # Create robot state publisher node
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           name='robot_state_publisher',
           output='screen',
           parameters=[{
               'use_sim_time': use_sim_time,
               'robot_description': open(
                   os.path.join(pkg_share, 'urdf/humanoid.xacro')
               ).read()
           }])
       
       # Create joint state publisher GUI
       joint_state_publisher_gui = Node(
           package='joint_state_publisher_gui',
           executable='joint_state_publisher_gui',
           name='joint_state_publisher_gui',
           output='screen')
       
       # Create RViz node
       rviz = Node(
           package='rviz2',
           executable='rviz2',
           name='rviz2',
           output='screen',
           parameters=[{'use_sim_time': use_sim_time}],
           arguments=['d', os.path.join(pkg_share, 'rviz/humanoid.rviz')])
       
       # Create nodes and launch description
       ld = LaunchDescription()
       
       # Add launch arguments
       ld.add_action(declare_use_sim_time)
       
       # Add nodes
       ld.add_action(robot_state_publisher)
       ld.add_action(joint_state_publisher_gui)
       ld.add_action(rviz)
       
       return ld
   ```

## Runnable Code Example

Here's a complete Xacro file for a more sophisticated humanoid model with gaze sensors:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_humanoid">
  <! Define properties >
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="base_mass" value="10.0"/>
  <xacro:property name="head_mass" value="2.0"/>
  <xacro:property name="arm_mass" value="1.0"/>
  <xacro:property name="leg_mass" value="3.0"/>

  <! Include gazebo plugins >
  <xacro:include filename="$(find gazebo_plugins)/urdf/camera.gazebo.xacro"/>
  
  <! Materials >
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  <material name="gray">
    <color rgba="0.5 0.5 0.5 1"/>
  </material>
  <material name="brown">
    <color rgba="0.6 0.4 0.2 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>

  <! Base Link >
  <link name="base_link">
    <visual>
      <geometry>
        <capsule length="0.2" radius="0.1"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <capsule length="0.2" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${base_mass}"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <! Gazebo plugin for robot >
  <gazebo>
    <plugin name="ground_truth" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>ground_truth/state</topicName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>map</frameName>
    </plugin>
  </gazebo>

  <! Head Macro with sensor >
  <xacro:macro name="head_with_sensor" params="parent xyz rpy">
    <joint name="neck_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="head"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="1.0" upper="1.0" effort="100" velocity="1"/>
    </joint>

    <link name="head">
      <visual>
        <geometry>
          <sphere radius="0.1"/>
        </geometry>
        <material name="white"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.1"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${head_mass}"/>
        <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.008"/>
      </inertial>
    </link>

    <! Camera sensor in head >
    <joint name="camera_joint" type="fixed">
      <parent link="head"/>
      <child link="camera_link"/>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
    </joint>

    <link name="camera_link">
      <visual>
        <geometry>
          <box size="0.02 0.04 0.02"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.02 0.04 0.02"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="1e6" ixy="0" ixz="0" iyy="1e6" iyz="0" izz="1e6"/>
      </inertial>
    </link>

    <! Gazebo camera sensor >
    <gazebo reference="camera_link">
      <sensor type="camera" name="camera1">
        <update_rate>30.0</update_rate>
        <camera name="head_camera">
          <horizontal_fov>1.3962634</horizontal_fov>
          <image>
            <width>800</width>
            <height>600</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.02</near>
            <far>300</far>
          </clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <frame_name>camera_link</frame_name>
          <min_depth>0.1</min_depth>
          <max_depth>100.0</max_depth>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:macro>

  <! Arm Macro >
  <xacro:macro name="arm" params="side parent xyz rpy">
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <material name="gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${arm_mass}"/>
        <inertia ixx="0.0075" ixy="0.0" ixz="0.0" iyy="0.0075" iyz="0.0" izz="0.00125"/>
      </inertial>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_lower_arm">
      <visual>
        <geometry>
          <cylinder length="0.25" radius="0.04"/>
        </geometry>
        <material name="gray"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.25" radius="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${arm_mass * 0.7}"/>
        <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <! Leg Macro >
  <xacro:macro name="leg" params="side parent xyz rpy">
    <joint name="${side}_hip_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_leg"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="1.57" upper="1.57" effort="200" velocity="1"/>
    </joint>

    <link name="${side}_upper_leg">
      <visual>
        <geometry>
          <cylinder length="0.4" radius="0.06"/>
        </geometry>
        <material name="brown"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.4" radius="0.06"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${leg_mass}"/>
        <inertia ixx="0.016" ixy="0.0" ixz="0.0" iyy="0.016" iyz="0.0" izz="0.0018"/>
      </inertial>
    </link>

    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_upper_leg"/>
      <child link="${side}_lower_leg"/>
      <origin xyz="0 0 0.4" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="0.0" upper="1.57" effort="200" velocity="1"/>
    </joint>

    <link name="${side}_lower_leg">
      <visual>
        <geometry>
          <cylinder length="0.35" radius="0.05"/>
        </geometry>
        <material name="brown"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.35" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${leg_mass * 0.8}"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.0015"/>
      </inertial>
    </link>

    <joint name="${side}_ankle_joint" type="revolute">
      <parent link="${side}_lower_leg"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 0.35" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0.5" upper="0.5" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_foot">
      <visual>
        <geometry>
          <box size="0.2 0.1 0.05"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.2 0.1 0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${leg_mass * 0.3}"/>
        <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0017" iyz="0.0" izz="0.0013"/>
      </inertial>
    </link>
  </xacro:macro>

  <! Use the macros to build the robot >
  <xacro:head_with_sensor parent="base_link" xyz="0 0 0.2" rpy="0 0 0"/>
  <xacro:arm side="left" parent="base_link" xyz="0.15 0 0.1" rpy="0 0 0"/>
  <xacro:arm side="right" parent="base_link" xyz="0.15 0 0.1" rpy="0 0 0"/>
  <xacro:leg side="left" parent="base_link" xyz="0.07 0 0.2" rpy="0 0 0"/>
  <xacro:leg side="right" parent="base_link" xyz="0.07 0 0.2" rpy="0 0 0"/>
  
  <! Gazebo environment settings >
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>
</robot>
```

### Sample launch file to visualize the robot (`launch/humanoid_display.launch.py`):
```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Find the package share directory
    pkg_share = get_package_share_directory('robot_description')
    
    # Create launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')
    
    # Create robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(
                os.path.join(pkg_share, 'urdf/humanoid.xacro')
            ).read()
        }])
    
    # Create joint state publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen')
    
    # Create RViz node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}])
    
    # Create nodes and launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time)
    
    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher_gui)
    ld.add_action(rviz)
    
    return ld
```

## Miniproject

Create a complete humanoid robot model with the following requirements:

1. Model should include at least 20 joints (head, 2 arms with 3 joints each, 2 legs with 3 joints each)
2. Include at least 2 sensors (camera in head, IMU in torso)
3. Use Xacro macros to avoid repetition in the model
4. Parameterize the model so different sizes can be created
5. Include realistic inertial properties for all links
6. Add Gazebo plugins for physics simulation
7. Create a launch file to visualize the model in RViz

Validate your model by checking for errors and visualizing it in RViz before submission.

## Summary

This chapter covered the creation of humanoid robot models using URDF and Xacro in ROS 2:

 URDF is the standard XML format for describing robot models
 Xacro enhances URDF with macros, parameters, and mathematical expressions
 Humanoid robots require complex kinematic structures with multiple degrees of freedom
 Proper inertial properties are essential for realistic simulation
 Sensors and plugins can be included to enhance the robot model functionality

Creating accurate robot models is fundamental to successful robotics simulation and development, allowing developers to test algorithms before deployment on real hardware.