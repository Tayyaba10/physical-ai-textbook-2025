-----
title: Ch7  Gazebo Ignition Setup & World Building
module: 2
chapter: 7
sidebar_label: Ch7: Gazebo Ignition Setup & World Building
description: Setting up Gazebo Ignition for robotics simulation and creating custom worlds
tags: [gazebo, ignition, simulation, worldbuilding, sdf, robotics]
difficulty: intermediate
estimated_duration: 90
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Gazebo Ignition Setup & World Building

## Learning Outcomes
 Install and configure Gazebo Ignition for robotics simulation
 Understand the SDF (Simulation Description Format) for world definition
 Create custom worlds with static and dynamic objects
 Configure sensors and physics properties for simulation
 Integrate Gazebo with ROS 2 for robot simulation
 Build complex simulation environments for specific robotics applications
 Optimize simulation performance for different use cases

## Theory

### Gazebo Ignition Architecture

Gazebo Ignition (now known as Ignition Gazebo) is a nextgeneration robotics simulation framework that provides a modular architecture for creating realistic simulation environments. It replaced the older Gazebo Classic with improved performance, modularity, and plugin architecture.

<MermaidDiagram chart={`
graph TD;
    A[Gazebo Ignition] > B[Simulation Core];
    A > C[Rendering Engine];
    A > D[Physics Engine];
    A > E[Sensor System];
    A > F[GUI Framework];
    
    B > G[Entity Component System];
    C > H[OGRE 2.x];
    D > I[DART/ODE/Bullet];
    E > J[Sensor Plugins];
    F > K[QML/Qt GUI];
    
    L[ROS 2 Integration] > M[ROS 2 Bridge];
    M > N[Topic Mapping];
    M > O[Service Mapping];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style M fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

### SDF (Simulation Description Format)

SDF is the XMLbased format used to describe simulation environments in Gazebo. It defines:
 World structure and properties
 Models and their relationships
 Physics engine settings
 Lighting and environment properties
 Sensors and plugins

Key SDF elements:
 `<world>`: The root element for a simulation world
 `<model>`: Represents a physical object in the world
 `<link>`: A rigid body part of a model
 `<joint>`: Connection between links
 `<sensor>`: Sensor attached to a link
 `<plugin>`: Custom code to extend simulation functionality

### World Building Principles

Creating effective simulation worlds requires understanding:
 Spatial relationships and scale
 Object mass and inertial properties
 Collision and visual geometries
 Sensor placement and environment interaction
 Physics properties that affect robot behavior

## StepbyStep Labs

### Lab 1: Installing and Setting up Gazebo Ignition

1. **Install Gazebo Ignition** (Garden or newer):
   ```bash
   # For Ubuntu 22.04
   sudo apt update
   sudo apt install y wget
   wget https://packages.osrfoundation.org/gazebo.gpg O /tmp/gazebo.gpg
   sudo mkdir p /etc/apt/keyrings
   sudo cp /tmp/gazebo.gpg /etc/apt/keyrings/gazeboarchivekeyring.gpg
   echo "deb [arch=$(dpkg printarchitecture) signedby=/etc/apt/keyrings/gazeboarchivekeyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntustable $(source /etc/osrelease && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/gazebostable.list > /dev/null
   sudo apt update
   sudo apt install ignitiongarden
   ```

2. **Verify Installation**:
   ```bash
   ign gazebo version
   ```

3. **Set up Gazebo resources**:
   ```bash
   # Create a directory for custom worlds
   mkdir p ~/gazebo_worlds
   export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:~/gazebo_worlds
   ```

### Lab 2: Creating a Basic World File

1. **Create a simple world file** (`simple_room.sdf`):
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="simple_room">
       <physics type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1.0</real_time_factor>
         <real_time_update_rate>1000.0</real_time_update_rate>
       </physics>

       <! Lighting >
       <light name="sun" type="directional">
         <cast_shadows>true</cast_shadows>
         <pose>0 0 10 0 0 0</pose>
         <diffuse>0.8 0.8 0.8 1</diffuse>
         <specular>0.2 0.2 0.2 1</specular>
         <attenuation>
           <range>1000</range>
           <constant>0.9</constant>
           <linear>0.01</linear>
           <quadratic>0.001</quadratic>
         </attenuation>
         <direction>0.3 0.3 1</direction>
       </light>

       <! Ground plane >
       <model name="ground_plane">
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
             <material>
               <diffuse>0.7 0.7 0.7 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! Walls >
       <model name="wall_1">
         <pose>5 0 2.5 0 0 0</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
             </box>
             </geometry>
             <material>
               <diffuse>0.8 0.2 0.2 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <model name="wall_2">
         <pose>5 0 2.5 0 0 0</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
             </box>
             </geometry>
             <material>
               <diffuse>0.8 0.2 0.2 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <model name="wall_3">
         <pose>0 5 2.5 0 0 1.5707</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
             </box>
             </geometry>
             <material>
               <diffuse>0.2 0.8 0.2 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <model name="wall_4">
         <pose>0 5 2.5 0 0 1.5707</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.1 10 5</size>
             </box>
             </geometry>
             <material>
               <diffuse>0.2 0.8 0.2 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! Add a simple robot >
       <model name="simple_robot">
         <pose>0 0 0.2 0 0 0</pose>
         <link name="chassis">
           <inertial>
             <mass>1.0</mass>
             <inertia>
               <ixx>0.01</ixx>
               <iyy>0.01</iyy>
               <izz>0.02</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.5 0.3 0.2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.5 0.3 0.2</size>
               </box>
             </geometry>
             <material>
               <diffuse>0.1 0.1 0.8 1</diffuse>
               <specular>0.5 0.5 0.5 1</specular>
             </material>
           </visual>
         </link>
       </model>
     </world>
   </sdf>
   ```

2. **Launch the world**:
   ```bash
   ign gazebo simple_room.sdf
   ```

### Lab 3: Creating a Custom World with Sensors

1. **Create a more complex world file** (`office_world.sdf`) with sensors:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="office_world">
       <physics type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1.0</real_time_factor>
         <real_time_update_rate>1000.0</real_time_update_rate>
       </physics>

       <! Lighting >
       <light name="sun" type="directional">
         <cast_shadows>true</cast_shadows>
         <pose>0 0 10 0 0 0</pose>
         <diffuse>0.8 0.8 0.8 1</diffuse>
         <specular>0.2 0.2 0.2 1</specular>
         <direction>0.3 0.3 1</direction>
       </light>

       <! Ground plane with texture >
       <model name="floor">
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>20 20</size>
               </plane>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>20 20</size>
               </plane>
             </geometry>
             <material>
               <diffuse>0.8 0.8 0.8 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! Furniture >
       <model name="desk">
         <pose>2 2 0.4 0 0 0</pose>
         <static>true</static>
         <link name="base">
           <collision name="collision">
             <geometry>
               <box>
                 <size>1.5 0.8 0.8</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>1.5 0.8 0.8</size>
               </box>
             </geometry>
             <material>
               <diffuse>0.6 0.3 0.1 1</diffuse>
               <specular>0.2 0.2 0.2 1</specular>
             </material>
           </visual>
         </link>
         <link name="top">
           <pose>0 0 0.4 0 0 0</pose>
           <collision name="collision">
             <geometry>
               <box>
                 <size>1.5 0.8 0.05</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>1.5 0.8 0.05</size>
               </box>
             </geometry>
             <material>
               <diffuse>0.7 0.5 0.2 1</diffuse>
               <specular>0.3 0.3 0.3 1</specular>
             </material>
           </visual>
         </link>
         <joint name="top_joint" type="fixed">
           <parent>base</parent>
           <child>top</child>
         </joint>
       </model>

       <! Bookshelf >
       <model name="bookshelf">
         <pose>3 0 0.6 0 0 0</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.3 1.2 1.2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.3 1.2 1.2</size>
               </box>
             </geometry>
             <material>
               <diffuse>0.4 0.2 0.0 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! Add a robot with sensors >
       <model name="sensor_robot">
         <pose>2 3 0.3 0 0 0</pose>
         <link name="chassis">
           <inertial>
             <mass>2.0</mass>
             <inertia>
               <ixx>0.1</ixx>
               <iyy>0.1</iyy>
               <izz>0.2</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <cylinder>
                 <radius>0.2</radius>
                 <length>0.3</length>
               </cylinder>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <cylinder>
                 <radius>0.2</radius>
                 <length>0.3</length>
               </cylinder>
             </geometry>
             <material>
               <diffuse>0.0 0.6 0.8 1</diffuse>
               <specular>0.5 0.5 0.5 1</specular>
             </material>
           </visual>
         </link>

         <! Laser scanner >
         <link name="laser_link">
           <pose>0.15 0 0.1 0 0 0</pose>
           <inertial>
             <mass>0.1</mass>
             <inertia>
               <ixx>0.001</ixx>
               <iyy>0.001</iyy>
               <izz>0.001</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.05 0.05 0.05</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.05 0.05 0.05</size>
               </box>
             </geometry>
             <material>
               <diffuse>0.8 0.1 0.1 1</diffuse>
               <specular>0.8 0.8 0.8 1</specular>
             </material>
           </visual>
         </link>

         <joint name="laser_joint" type="fixed">
           <parent>chassis</parent>
           <child>laser_link</child>
           <pose>0.15 0 0.1 0 0 0</pose>
         </joint>

         <! Attach laser sensor >
         <sensor name="laser" type="ray">
           <pose>0 0 0 0 0 0</pose>
           <ray>
             <scan>
               <horizontal>
                 <samples>360</samples>
                 <resolution>1</resolution>
                 <min_angle>3.14159</min_angle>
                 <max_angle>3.14159</max_angle>
               </horizontal>
             </scan>
             <range>
               <min>0.1</min>
               <max>10.0</max>
               <resolution>0.01</resolution>
             </range>
           </ray>
           <always_on>true</always_on>
           <update_rate>10</update_rate>
         </sensor>
       </model>
     </world>
   </sdf>
   ```

2. **Launch the office world**:
   ```bash
   ign gazebo office_world.sdf
   ```

### Lab 4: Integrating with ROS 2

1. **Install Ignition ROS 2 bridge**:
   ```bash
   sudo apt install roshumblerosignbridge
   sudo apt install roshumbleigngazebo
   ```

2. **Create a launch file** (`launch/gazebo_with_robot.launch.py`):
   ```python
   import os
   from ament_index_python.packages import get_package_share_directory
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   
   def generate_launch_description():
       # Get Gazebo launch file
       gazebo_ros_share = get_package_share_directory('ros_gz_sim')
       
       # Declare launch arguments
       world_file = DeclareLaunchArgument(
           'world',
           default_value=os.path.join(
               get_package_share_directory('robot_description'),
               'worlds',
               'office_world.sdf'
           ),
           description='World file to load in Gazebo'
       )
       
       # Include Gazebo launch
       gazebo_launch = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(gazebo_ros_share, 'launch', 'gz_sim.launch.py')
           ),
           launch_arguments={
               'gz_args': ['r ', LaunchConfiguration('world')]
           }.items()
       )
       
       # Bridge for ROS 2 communication
       bridge = Node(
           package='ros_gz_bridge',
           executable='parameter_bridge',
           arguments=[
               '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
               '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
               '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
               '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V'
           ],
           remapping=[
               ('/scan', '/laser_scan'),
               ('/tf', '/tf')
           ],
           output='screen'
       )
       
       # Create launch description
       ld = LaunchDescription()
       ld.add_action(world_file)
       ld.add_action(gazebo_launch)
       ld.add_action(bridge)
       
       return ld
   ```

## Runnable Code Example

Here's a complete example with a more sophisticated world that includes a structured city environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="city_world">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <! Sky >
    <scene>
      <sky>
        <clouds>
          <speed>1</speed>
          <direction>0.8 0.1 0</direction>
        </clouds>
      </sky>
    </scene>

    <! Lighting >
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>0.3 0.3 1</direction>
    </light>

    <! Ground plane with road markings >
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <diffuse>0.3 0.3 0.3 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
          <! Road markings >
          <plugin filename="gzsimvisualsystem" name="gz::sim::systems::Visual">
            <visual>
              <geometry type="box">
                <box size="20 0.2 0.01" />
              </geometry>
              <pose>0 0 0.01 0 0 0</pose>
              <material>
                <diffuse>1 1 1 1</diffuse>
              </material>
            </visual>
          </plugin>
        </visual>
      </link>
    </model>

    <! Buildings >
    <model name="building_1">
      <pose>5 5 5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>6 6 10</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>6 6 10</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.7 0.7 0.8 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name="building_2">
      <pose>8 3 7 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>8 4 14</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>8 4 14</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.6 0.6 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <! Trees >
    <model name="tree_1">
      <pose>8 8 1.5 0 0 0</pose>
      <static>true</static>
      <link name="trunk">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>3</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>3</length>
            </cylinder>
          </geometry>
          <material>
            <diffuse>0.4 0.2 0.1 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
      <link name="leaves">
        <pose>0 0 2 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>1.5</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>1.5</radius>
            </sphere>
          </geometry>
          <material>
            <diffuse>0.1 0.6 0.2 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
      <joint name="leaves_joint" type="fixed">
        <parent>trunk</parent>
        <child>leaves</child>
      </joint>
    </model>

    <! Robotic platform with sensors >
    <model name="autonomous_robot">
      <pose>0 0 0.3 0 0 0</pose>
      <link name="base_link">
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <iyy>1.0</iyy>
            <izz>1.8</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.8 0.6 0.3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.8 0.6 0.3</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.1 0.1 0.9 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>

      <! Differential drive wheels >
      <link name="left_wheel">
        <pose>0.2 0.4 0.15 0 0 0</pose>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.01</ixx>
            <iyy>0.01</iyy>
            <izz>0.02</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <diffuse>0.2 0.2 0.2 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>

      <link name="right_wheel">
        <pose>0.2 0.4 0.15 0 0 0</pose>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.01</ixx>
            <iyy>0.01</iyy>
            <izz>0.02</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.15</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <diffuse>0.2 0.2 0.2 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>

      <joint name="left_wheel_joint" type="continuous">
        <parent>base_link</parent>
        <child>left_wheel</child>
        <axis>
          <xyz>0 1 0</xyz>
        </axis>
      </joint>

      <joint name="right_wheel_joint" type="continuous">
        <parent>base_link</parent>
        <child>right_wheel</child>
        <axis>
          <xyz>0 1 0</xyz>
        </axis>
      </joint>

      <! Sensors >
      <! RGBD Camera >
      <link name="camera_link">
        <pose>0.3 0 0.2 0 0 0</pose>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.001</ixx>
            <iyy>0.001</iyy>
            <izz>0.001</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.08 0.05</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.08 0.05</size>
            </box>
          </geometry>
          <material>
            <diffuse>0.8 0.1 0.1 1</diffuse>
          </material>
        </visual>
      </link>

      <joint name="camera_joint" type="fixed">
        <parent>base_link</parent>
        <child>camera_link</child>
      </joint>

      <sensor name="rgbd_camera" type="rgbd_camera">
        <pose>0 0 0 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov> <! 60 degrees >
          <image>
            <width>640</width>
            <height>480</height>
          </image>
          <clip>
            <near>0.1</near>
            <far>10.0</far>
          </clip>
        </camera>
        <always_on>true</always_on>
        <update_rate>15</update_rate>
      </sensor>

      <! IMU Sensor >
      <sensor name="imu_sensor" type="imu">
        <pose>0 0 0 0 0 0</pose>
        <always_on>true</always_on>
        <update_rate>100</update_rate>
      </sensor>
    </model>
  </world>
</sdf>
```

### Python script to interact with the simulation (`gazebo_robot_controller.py`):

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

class GazeboRobotController(Node):
    def __init__(self):
        super().__init__('gazebo_robot_controller')
        
        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Create subscriber for laser scan
        self.scan_sub = self.create_subscription(
            LaserScan, '/laser_scan', self.scan_callback, 10)
        
        # Create subscriber for odometry
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Robot state
        self.scan_data = None
        self.odom_data = None
        self.linear_vel = 0.5
        self.angular_vel = 0.5
        
        self.get_logger().info('Gazebo Robot Controller initialized')
    
    def scan_callback(self, msg):
        self.scan_data = msg
    
    def odom_callback(self, msg):
        self.odom_data = msg
    
    def control_loop(self):
        cmd = Twist()
        
        if self.scan_data is not None:
            # Get frontfacing ranges (±30 degrees)
            front_ranges = self.scan_data.ranges[
                len(self.scan_data.ranges)//2  30 : 
                len(self.scan_data.ranges)//2 + 30]
            
            # Filter out invalid values
            front_ranges = [r for r in front_ranges if not (math.isinf(r) or math.isnan(r))]
            
            if front_ranges:
                min_distance = min(front_ranges)
                
                # Obstacle avoidance
                if min_distance < 1.0:
                    cmd.linear.x = 0.0
                    cmd.angular.z = self.angular_vel
                    self.get_logger().info(f'Obstacle detected at {min_distance:.2f}m, turning...')
                else:
                    cmd.linear.x = self.linear_vel
                    cmd.angular.z = 0.0
                    self.get_logger().info(f'Moving forward, obstacle distance: {min_distance:.2f}m')
            else:
                # If no valid readings, stop
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
        else:
            # If no scan data yet, don't move
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    controller = GazeboRobotController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Miniproject

Create a complete warehouse simulation environment with the following features:

1. A warehouse layout with shelves, corridors, and workstations
2. A mobile robot with appropriate sensors (lidar, RGB camera)
3. Moving obstacles (simulated people or other robots)
4. A navigation task where the robot must reach different waypoints
5. A launch file that starts the entire simulation with ROS 2 bridge
6. A controller node that enables the robot to navigate autonomously

Your simulation should include:
 Custom SDF world file for the warehouse
 Robot model with sensors
 At least 3 moving obstacles
 Navigation controller to move between waypoints
 Collision avoidance
 Performance metrics logging

## Summary

This chapter covered the setup and use of Gazebo Ignition for robotics simulation:

 **Installation and Configuration**: Setting up Gazebo Ignition and understanding its architecture
 **SDF World Building**: Creating simulation environments using the Simulation Description Format
 **Sensor Integration**: Adding sensors to robots in simulation
 **ROS 2 Integration**: Connecting Gazebo with ROS 2 for realistic robotics simulation
 **Environment Design**: Creating complex simulation worlds with static and dynamic objects

Gazebo Ignition provides a powerful platform for developing, testing, and validating robotics applications in a safe, controlled environment before deploying to real hardware.