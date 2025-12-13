-----
title: Ch8  Simulating LiDAR, Depth Cameras & IMUs
module: 2
chapter: 8
sidebar_label: Ch8: Simulating LiDAR, Depth Cameras & IMUs
description: Setting up and using sensor simulation in Gazebo for robotics applications
tags: [lidar, camera, imus, sensors, simulation, perception, robotics]
difficulty: intermediate
estimated_duration: 90
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Simulating LiDAR, Depth Cameras & IMUs

## Learning Outcomes
 Understand the principles behind LiDAR, depth camera, and IMU sensors
 Configure and implement these sensors in Gazebo simulation
 Integrate sensor data with ROS 2 for robotics applications
 Process and interpret sensor data for perception tasks
 Calibrate sensor models to match realworld performance
 Implement sensor fusion techniques for enhanced perception
 Troubleshoot common issues in sensor simulation

## Theory

### LiDAR Sensors in Simulation

LiDAR (Light Detection and Ranging) sensors emit laser pulses and measure the time it takes for the light to return after reflecting off objects. In simulation, LiDAR sensors work by casting rays into the environment and detecting collisions with objects.

<MermaidDiagram chart={`
graph TD;
    A[LiDAR Sensor] > B[Ray Casting];
    B > C[Range Measurement];
    C > D[Point Cloud];
    D > E[Object Detection];
    E > F[Mapping];
    
    G[LiDAR Parameters] > H[Sample Count];
    G > I[Field of View];
    G > J[Range Min/Max];
    G > K[Update Rate];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style D fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

### Depth Cameras in Simulation

Depth cameras, such as RGBD sensors, provide both color images and depth information. In simulation, these are typically implemented using stereo vision principles or structured light methods. The depth information is calculated by measuring the distance to objects in the scene.

### IMU Sensors in Simulation

Inertial Measurement Units (IMUs) measure linear acceleration and angular velocity. In simulation, IMUs are typically modeled using:
 Noise models to simulate realworld sensor inaccuracies
 Bias terms that drift over time
 Sampling rate to match real hardware

## StepbyStep Labs

### Lab 1: Configuring LiDAR Sensors in Gazebo

1. **Create a robot model with LiDAR** (`lidar_robot.sdf`):
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <model name="lidar_robot">
       <link name="chassis">
         <pose>0 0 0.1 0 0 0</pose>
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

       <! LiDAR sensor >
       <link name="lidar_link">
         <pose>0.1 0 0.2 0 0 0</pose>
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
             <cylinder>
               <radius>0.05</radius>
               <length>0.05</length>
             </cylinder>
           </geometry>
         </collision>
         <visual name="visual">
           <geometry>
             <cylinder>
               <radius>0.05</radius>
               <length>0.05</length>
             </cylinder>
           </geometry>
           <material>
             <diffuse>0.8 0.1 0.1 1</diffuse>
             <specular>0.8 0.8 0.8 1</specular>
           </material>
         </visual>
       </link>

       <joint name="lidar_joint" type="fixed">
         <parent>chassis</parent>
         <child>lidar_link</child>
         <pose>0.1 0 0.2 0 0 0</pose>
       </joint>

       <! 2D LiDAR sensor >
       <sensor name="lidar_2d" type="ray">
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
         <visualize>true</visualize>
       </sensor>
     </model>
   </sdf>
   ```

2. **Create a world with obstacles for LiDAR testing** (`lidar_test_world.sdf`):
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="lidar_test">
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

       <! Ground plane >
       <model name="ground_plane">
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
               <diffuse>0.7 0.7 0.7 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! Obstacles for LiDAR testing >
       <model name="wall_1">
         <pose>5 0 1 0 0 0</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.1 10 2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.1 10 2</size>
             </box>
             </geometry>
             <material>
               <diffuse>0.8 0.2 0.2 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <model name="obstacle_1">
         <pose>2 2 0.5 0 0 0</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <cylinder>
                 <radius>0.3</radius>
                 <length>1</length>
               </cylinder>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <cylinder>
                 <radius>0.3</radius>
                 <length>1</length>
               </cylinder>
             </geometry>
             <material>
               <diffuse>0.2 0.8 0.2 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <model name="obstacle_2">
         <pose>3 1 0.3 0 0 0</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.4 0.4 0.6</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.4 0.4 0.6</size>
             </box>
             </geometry>
             <material>
               <diffuse>0.8 0.8 0.2 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! Include the robot >
       <include>
         <uri>model://lidar_robot</uri>
         <pose>0 0 0 0 0 0</pose>
       </include>
     </world>
   </sdf>
   ```

### Lab 2: Configuring Depth Cameras in Gazebo

1. **Create a robot model with RGBD camera** (`rgbd_camera_robot.sdf`):
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <model name="rgbd_camera_robot">
       <link name="chassis">
         <pose>0 0 0.1 0 0 0</pose>
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
             <diffuse>0.1 0.8 0.1 1</diffuse>
             <specular>0.5 0.5 0.5 1</specular>
           </material>
         </visual>
       </link>

       <! RGBD camera >
       <link name="camera_link">
         <pose>0.15 0 0.2 0 0 0</pose>
         <inertial>
           <mass>0.05</mass>
           <inertia>
             <ixx>0.0001</ixx>
             <iyy>0.0001</iyy>
             <izz>0.0001</izz>
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
             <diffuse>0.1 0.1 0.1 1</diffuse>
             <specular>0.8 0.8 0.8 1</specular>
           </material>
         </visual>
       </link>

       <joint name="camera_joint" type="fixed">
         <parent>chassis</parent>
         <child>camera_link</child>
         <pose>0.15 0 0.2 0 0 0</pose>
       </joint>

       <! RGBD camera sensor >
       <sensor name="rgbd_camera" type="rgbd_camera">
         <pose>0 0 0 0 0 0</pose>
         <camera>
           <horizontal_fov>1.047</horizontal_fov> <! 60 degrees >
           <image>
             <width>640</width>
             <height>480</height>
             <format>R8G8B8</format>
           </image>
           <clip>
             <near>0.1</near>
             <far>10.0</far>
           </clip>
         </camera>
         <always_on>true</always_on>
         <update_rate>15</update_rate>
         <visualize>true</visualize>
       </sensor>
     </model>
   </sdf>
   ```

### Lab 3: Configuring IMU Sensors in Gazebo

1. **Create a robot model with IMU** (`imu_robot.sdf`):
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <model name="imu_robot">
       <link name="chassis">
         <pose>0 0 0.1 0 0 0</pose>
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
             <diffuse>0.8 0.1 0.8 1</diffuse>
             <specular>0.5 0.5 0.5 1</specular>
           </material>
         </visual>
       </link>

       <! IMU sensor >
       <sensor name="imu_sensor" type="imu">
         <pose>0 0 0 0 0 0</pose>
         <topic>imu</topic>
         <always_on>true</always_on>
         <update_rate>100</update_rate>
         <imu>
           <angular_velocity>
             <x>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>2e4</stddev>
               </noise>
             </x>
             <y>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>2e4</stddev>
               </noise>
             </y>
             <z>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>2e4</stddev>
               </noise>
             </z>
           </angular_velocity>
           <linear_acceleration>
             <x>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>1.7e2</stddev>
               </noise>
             </x>
             <y>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>1.7e2</stddev>
               </noise>
             </y>
             <z>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>1.7e2</stddev>
               </noise>
             </z>
           </linear_acceleration>
         </imu>
       </sensor>
     </model>
   </sdf>
   ```

2. **Create a world with IMU testing environment** (`imu_test_world.sdf`):
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="imu_test">
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

       <! Ground plane >
       <model name="ground_plane">
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
               <diffuse>0.7 0.7 0.7 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! A ramp to test IMU on slopes >
       <model name="ramp">
         <pose>3 0 0.5 0 0 0</pose>
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <mesh>
                 <scale>1 1 1</scale>
                 <uri>file://meshes/ramp.dae</uri>
               </mesh>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <mesh>
                 <scale>1 1 1</scale>
                 <uri>file://meshes/ramp.dae</uri>
               </mesh>
             </geometry>
             <material>
               <diffuse>0.5 0.3 0.0 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <! Include the robot >
       <include>
         <uri>model://imu_robot</uri>
         <pose>0 0 0.1 0 0 0</pose>
       </include>
     </world>
   </sdf>
   ```

### Lab 4: Integrating Sensors with ROS 2

1. **Create a launch file for sensor simulation** (`launch/sensor_simulation.launch.py`):
   ```python
   import os
   from ament_index_python.packages import get_package_share_directory
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   
   def generate_launch_description():
       # Get Gazebo ROS launch directory
       gazebo_ros_share = get_package_share_directory('ros_gz_sim')
       
       # Declare launch arguments
       world = DeclareLaunchArgument(
           'world',
           default_value=os.path.join(
               get_package_share_directory('sensor_integration'),
               'worlds',
               'multi_sensor_test.sdf'
           ),
           description='World file to load in Gazebo'
       )
       
       # Launch Gazebo
       gazebo = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(gazebo_ros_share, 'launch', 'gz_sim.launch.py')
           ),
           launch_arguments={
               'gz_args': ['r ', LaunchConfiguration('world')]
           }.items()
       )
       
       # ROSGazebo bridge for sensors
       bridge = Node(
           package='ros_gz_bridge',
           executable='parameter_bridge',
           parameters=[{
               'config_file': os.path.join(
                   get_package_share_directory('sensor_integration'),
                   'config',
                   'sensors_bridge.yaml'
               )
           }],
           output='screen'
       )
       
       # Create launch description
       ld = LaunchDescription()
       ld.add_action(world)
       ld.add_action(gazebo)
       ld.add_action(bridge)
       
       return ld
   ```

2. **Create bridge configuration** (`config/sensors_bridge.yaml`):
   ```yaml
   # Bridge configuration for sensors
    ros_topic_name: "/scan"
     gz_topic_name: "/lidar_2d/scan"
     ros_type_name: "sensor_msgs/msg/LaserScan"
     gz_type_name: "gz.msgs.LaserScan"
   
    ros_topic_name: "/rgb_camera/image_raw"
     gz_topic_name: "/rgbd_camera/image"
     ros_type_name: "sensor_msgs/msg/Image"
     gz_type_name: "gz.msgs.Image"
   
    ros_topic_name: "/rgb_camera/camera_info"
     gz_topic_name: "/rgbd_camera/camera_info"
     ros_type_name: "sensor_msgs/msg/CameraInfo"
     gz_type_name: "gz.msgs.CameraInfo"
   
    ros_topic_name: "/depth_camera/image_raw"
     gz_topic_name: "/rgbd_camera/depth_image"
     ros_type_name: "sensor_msgs/msg/Image"
     gz_type_name: "gz.msgs.Image"
   
    ros_topic_name: "/imu"
     gz_topic_name: "/imu_sensor/imu"
     ros_type_name: "sensor_msgs/msg/Imu"
     gz_type_name: "gz.msgs.IMU"
   ```

## Runnable Code Example

Here's a complete example of sensor processing in ROS 2:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf_transformations import euler_from_quaternion
import math

class MultiSensorRobot(Node):
    def __init__(self):
        super().__init__('multi_sensor_robot')
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/rgb_camera/image_raw', self.image_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Robot state
        self.scan_data = None
        self.image_data = None
        self.imu_data = None
        self.linear_vel = 0.3
        self.angular_vel = 0.5
        
        self.get_logger().info('Multisensor robot node started')
    
    def lidar_callback(self, msg):
        self.scan_data = msg
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.image_data = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
    
    def imu_callback(self, msg):
        self.imu_data = msg
    
    def detect_objects_in_image(self, img):
        """Simple object detection using color thresholding"""
        if img is None:
            return []
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for red color
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)
                objects.append((x, y, w, h, area))
        
        return objects
    
    def get_orientation_from_imu(self):
        """Extract orientation from IMU data"""
        if self.imu_data is None:
            return 0.0
        
        # Extract orientation quaternion
        orientation = self.imu_data.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        
        # Convert to Euler angles
        euler = euler_from_quaternion(quaternion)
        
        return euler[2]  # Return yaw/heading
    
    def control_loop(self):
        cmd = Twist()
        
        # Process LiDAR data for obstacle avoidance
        if self.scan_data is not None:
            # Get frontfacing ranges (±30 degrees)
            front_ranges = self.scan_data.ranges[
                len(self.scan_data.ranges)//2  30 : 
                len(self.scan_data.ranges)//2 + 30]
            
            # Filter out invalid values
            front_ranges = [r for r in front_ranges if not (math.isinf(r) or math.isnan(r))]
            
            if front_ranges:
                min_distance = min(front_ranges)
                
                # Check if we need to avoid obstacles
                if min_distance < 0.8:
                    cmd.linear.x = 0.0
                    cmd.angular.z = self.angular_vel
                    self.get_logger().info(f'Obstacle detected at {min_distance:.2f}m, turning...')
                else:
                    # Process camera data for navigation
                    if self.image_data is not None:
                        objects = self.detect_objects_in_image(self.image_data)
                        if objects:
                            # If objects detected, turn away from the largest one
                            largest_obj = max(objects, key=lambda x: x[4])  # Sort by area
                            (x, y, w, h, area) = largest_obj
                            
                            # Determine turn direction based on object position
                            center_x = x + w/2
                            image_center = self.image_data.shape[1] / 2
                            
                            if center_x < image_center:
                                cmd.angular.z = 0.3  # Turn right
                            else:
                                cmd.angular.z = 0.3   # Turn left
                            
                            cmd.linear.x = 0.0  # Stop moving forward when detecting objects
                            self.get_logger().info(f'Detected object, turning to avoid')
                        else:
                            # No objects detected, move forward
                            cmd.linear.x = self.linear_vel
                            cmd.angular.z = 0.0
                    else:
                        # No camera data, move forward if no obstacles
                        cmd.linear.x = self.linear_vel
                        cmd.angular.z = 0.0
            else:
                # If no valid readings, stop
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
        else:
            # If no scan data yet, don't move
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)
        
        # Log IMU data
        if self.imu_data is not None:
            orientation = self.get_orientation_from_imu()
            self.get_logger().info(f'Robot orientation: {orientation:.2f} rad')


def main(args=None):
    rclpy.init(args=args)
    robot = MultiSensorRobot()
    
    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        pass
    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Processing LiDAR data for mapping (`lidar_processing.py`):

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point
import numpy as np
import math

class LidarMapper(Node):
    def __init__(self):
        super().__init__('lidar_mapper')
        
        # Parameters
        self.declare_parameter('map_resolution', 0.1)  # meters per cell
        self.declare_parameter('map_width', 20.0)      # meters
        self.declare_parameter('map_height', 20.0)     # meters
        
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        
        # Calculate map dimensions in cells
        self.map_width_cells = int(self.map_width / self.map_resolution)
        self.map_height_cells = int(self.map_height / self.map_resolution)
        
        # Initialize occupancy grid
        self.occupancy_map = np.full((self.map_height_cells, self.map_width_cells), 1, dtype=np.int8)  # 1 = unknown
        
        # Subscribers and publishers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map', 10)
        
        # Timer to publish map
        self.map_timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info('Lidar Mapper initialized')
    
    def scan_callback(self, msg):
        """Process incoming LiDAR scan and update occupancy grid"""
        if len(msg.ranges) == 0:
            return
        
        # Robot position in map coordinates (assuming robot is at center)
        robot_x = self.map_width_cells // 2
        robot_y = self.map_height_cells // 2
        
        # Process each range measurement
        for i, range_val in enumerate(msg.ranges):
            if math.isinf(range_val) or math.isnan(range_val) or range_val > msg.range_max:
                continue
            
            # Calculate angle of this measurement
            angle = msg.angle_min + i * msg.angle_increment
            
            # Calculate end point of this ray
            end_x = robot_x + int((range_val * math.cos(angle)) / self.map_resolution)
            end_y = robot_y + int((range_val * math.sin(angle)) / self.map_resolution)
            
            # Perform ray tracing to mark free space
            self.trace_ray(robot_x, robot_y, end_x, end_y)
            
            # Mark endpoint as occupied (if it's within sensor range)
            if range_val < msg.range_max * 0.9:  # Only mark as occupied if it's a real obstacle
                if 0 <= end_x < self.map_width_cells and 0 <= end_y < self.map_height_cells:
                    self.occupancy_map[end_y, end_x] = 100  # Occupied (100%)
    
    def trace_ray(self, start_x, start_y, end_x, end_y):
        """Ray tracing algorithm to mark free space"""
        # Bresenham's line algorithm
        dx = abs(end_x  start_x)
        dy = abs(end_y  start_y)
        sx = 1 if start_x < end_x else 1
        sy = 1 if start_y < end_y else 1
        err = dx  dy
        
        x, y = start_x, start_y
        
        while True:
            # Check bounds
            if not (0 <= x < self.map_width_cells and 0 <= y < self.map_height_cells):
                break
                
            # Mark as free space, but only if it's not already marked as occupied
            if self.occupancy_map[y, x] != 100:
                self.occupancy_map[y, x] = 0  # Free space (0%)
                
            if x == end_x and y == end_y:
                break
                
            e2 = 2 * err
            if e2 > dy:
                err = dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def publish_map(self):
        """Publish the occupancy grid map"""
        msg = OccupancyGrid()
        
        # Set header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        # Set map info
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width_cells
        msg.info.height = self.map_height_cells
        
        # Set origin (assuming map center is at (0,0,0))
        msg.info.origin.position.x = self.map_width / 2.0
        msg.info.origin.position.y = self.map_height / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Flatten the occupancy map for the message
        msg.data = self.occupancy_map.flatten().tolist()
        
        self.map_pub.publish(msg)
        
        self.get_logger().info(f'Published map: {self.map_width_cells}x{self.map_height_cells} cells')


def main(args=None):
    rclpy.init(args=args)
    mapper = LidarMapper()
    
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        pass
    finally:
        mapper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Miniproject

Create a complete sensor fusion system that:

1. Simulates a robot with LiDAR, RGBD camera, and IMU sensors
2. Processes data from all sensors simultaneously
3. Implements sensor fusion for improved perception
4. Creates a map of the environment using LiDAR data
5. Localizes the robot in the map using IMU data
6. Uses camera data to identify and track objects
7. Implements obstacle avoidance using fused sensor data
8. Provides visualization of the fused sensor information

Your system should include:
 Complete Gazebo world with features for sensor testing
 Robot model with all three sensor types
 ROS 2 nodes for sensor processing and fusion
 Launch file to start the complete system
 Performance metrics and logging

## Summary

This chapter covered the simulation, integration, and processing of key robotic sensors:

 **LiDAR Simulation**: Configuring 2D and 3D laser scanners in Gazebo with proper parameters
 **Depth Camera Simulation**: Setting up RGBD cameras with realistic image and depth data
 **IMU Simulation**: Creating inertial measurement units with noise models matching real sensors
 **ROS 2 Integration**: Connecting simulated sensors to the ROS 2 ecosystem
 **Sensor Processing**: Techniques for processing and interpreting sensor data
 **Sensor Fusion**: Combining data from multiple sensors for enhanced perception

These sensors form the foundation of robot perception systems, and their simulation is crucial for developing and testing robotics algorithms in safe, controlled environments before deployment on real hardware.