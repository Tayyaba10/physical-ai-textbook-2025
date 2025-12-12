---
title: Ch5 - Building & Launching ROS 2 Packages
module: 1
chapter: 5
sidebar_label: Ch5: Building & Launching ROS 2 Packages
description: Managing ROS 2 packages, building systems, and launch configurations
tags: [ros2, packages, building, launch, cmake, ament]
difficulty: intermediate
estimated_duration: 60
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Building & Launching ROS 2 Packages

## Learning Outcomes
- Understand the structure and organization of ROS 2 packages
- Create and manage packages using colcon build system
- Configure CMakeLists.txt and package.xml for different build types
- Create launch files to start multiple nodes with parameters
- Use launch arguments and conditional launching
- Integrate third-party libraries into ROS 2 packages
- Deploy packages to different environments

## Theory

### Package Structure in ROS 2

ROS 2 packages follow a standardized structure that enables consistent building and deployment across different systems. The package is the basic unit of code organization in ROS 2.

<MermaidDiagram chart={`
graph TD;
    A[ROS 2 Package] --> B[package.xml];
    A --> C[CMakeLists.txt];
    A --> D[src/];
    A --> E[include/];
    A --> F[launch/];
    A --> G[config/];
    A --> H[rviz/];
    A --> I[urdf/];
    A --> J[params/];
    
    D --> K[Node Source Files];
    E --> L[Header Files];
    F --> M[Launch Files];
    G --> N[Configuration Files];
    I --> O[URDF/Xacro Files];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style B fill:#2196F3,stroke:#0D47A1,color:#fff;
    style C fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

### Build Systems in ROS 2

ROS 2 uses the `colcon` build system, which is a command-line tool that provides a unified interface for building different types of packages (CMake, Python, etc.) in a workspace. The primary build systems are:
- `ament_cmake` for C++ packages
- `ament_python` for Python packages
- `ament_cmake_python` for mixed C++/Python packages

### Launch System

The ROS 2 launch system uses Python files instead of XML (as in ROS 1) to provide more flexibility and powerful features such as:
- Conditional launching
- Launch arguments
- Node composition
- Event handling

## Step-by-Step Labs

### Lab 1: Creating a Complex Package with Multiple Components

1. **Create a comprehensive package** with both C++ and Python nodes:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_cmake --dependencies rclcpp rclpy std_msgs geometry_msgs sensor_msgs example_interfaces complex_robot_controller
   cd complex_robot_controller
   ```

2. **Create directory structure**:
   ```bash
   mkdir -p src include launch config
   ```

3. **Update package.xml** to include all dependencies:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>complex_robot_controller</name>
     <version>0.0.0</version>
     <description>Complex robot controller with multiple components</description>
     <maintainer email="user@todo.todo">Your Name</maintainer>
     <license>Apache License 2.0</license>

     <buildtool_depend>ament_cmake</buildtool_depend>

     <depend>rclcpp</depend>
     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>geometry_msgs</depend>
     <depend>sensor_msgs</depend>
     <depend>example_interfaces</depend>
     <depend>launch</depend>
     <depend>launch_ros</depend>

     <test_depend>ament_lint_auto</test_depend>
     <test_depend>ament_lint_common</test_depend>

     <export>
       <build_type>ament_cmake</build_type>
     </export>
   </package>
   ```

4. **Create C++ header file** (`include/complex_robot_controller/motion_controller.hpp`):
   ```cpp
   #ifndef MOTION_CONTROLLER_HPP_
   #define MOTION_CONTROLLER_HPP_

   #include "rclcpp/rclcpp.hpp"
   #include "geometry_msgs/msg/twist.hpp"
   #include "std_msgs/msg/float64.hpp"
   #include "sensor_msgs/msg/laser_scan.hpp"

   namespace complex_robot_controller
   {
     class MotionController : public rclcpp::Node
     {
     public:
       MotionController();
       
     private:
       void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
       void safety_timer_callback();
       
       rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
       rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
       rclcpp::TimerBase::SharedPtr safety_timer_;
       
       double linear_vel_;
       double angular_vel_;
       bool obstacle_detected_;
       double safety_distance_;
     };
   }

   #endif  // MOTION_CONTROLLER_HPP_
   ```

5. **Create C++ implementation** (`src/motion_controller.cpp`):
   ```cpp
   #include "complex_robot_controller/motion_controller.hpp"
   #include <cmath>

   namespace complex_robot_controller
   {
     MotionController::MotionController()
     : Node("motion_controller"),
       linear_vel_(0.5),
       angular_vel_(0.5),
       obstacle_detected_(false),
       safety_distance_(0.8)
     {
       cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
       scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
         "scan", 10, 
         std::bind(&MotionController::scan_callback, this, std::placeholders::_1));
       
       safety_timer_ = this->create_wall_timer(
         std::chrono::milliseconds(100),  // 10 Hz
         std::bind(&MotionController::safety_timer_callback, this));
     }

     void MotionController::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
     {
       // Find minimum distance in front of the robot
       float min_distance = std::numeric_limits<float>::max();
       
       // Consider only the front 90 degrees
       int start_idx = msg->ranges.size() / 2 - 45;
       int end_idx = msg->ranges.size() / 2 + 45;
       
       for (int i = start_idx; i < end_idx && i < msg->ranges.size(); ++i) {
         if (msg->ranges[i] < min_distance && !std::isinf(msg->ranges[i]) && !std::isnan(msg->ranges[i])) {
           min_distance = msg->ranges[i];
         }
       }
       
       obstacle_detected_ = (min_distance < safety_distance_);
     }

     void MotionController::safety_timer_callback()
     {
       auto cmd_msg = geometry_msgs::msg::Twist();
       
       if (obstacle_detected_) {
         // Stop and rotate away from obstacle
         cmd_msg.linear.x = 0.0;
         cmd_msg.angular.z = angular_vel_;
         RCLCPP_WARN(this->get_logger(), "Obstacle detected! Rotating away.");
       } else {
         // Move forward
         cmd_msg.linear.x = linear_vel_;
         cmd_msg.angular.z = 0.0;
       }
       
       cmd_vel_pub_->publish(cmd_msg);
     }
   }

   int main(int argc, char * argv[])
   {
     rclcpp::init(argc, argv);
     rclcpp::spin(std::make_shared<complex_robot_controller::MotionController>());
     rclcpp::shutdown();
     return 0;
   }
   ```

### Lab 2: Configuring CMakeLists.txt for Complex Package

1. **Update CMakeLists.txt**:
   ```cmake
   cmake_minimum_required(VERSION 3.8)
   project(complex_robot_controller)

   if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     add_compile_options(-Wall -Wextra -Wpedantic)
   endif()

   # Find dependencies
   find_package(ament_cmake REQUIRED)
   find_package(rclcpp REQUIRED)
   find_package(rclpy REQUIRED)
   find_package(std_msgs REQUIRED)
   find_package(geometry_msgs REQUIRED)
   find_package(sensor_msgs REQUIRED)
   find_package(example_interfaces REQUIRED)

   # Include header directories
   include_directories(include)

   # Create executable
   add_executable(motion_controller src/motion_controller.cpp)
   ament_target_dependencies(motion_controller 
     rclcpp 
     std_msgs 
     geometry_msgs 
     sensor_msgs)

   # Install executables
   install(TARGETS
     motion_controller
     DESTINATION lib/${PROJECT_NAME})

   # Install launch files
   install(DIRECTORY
     launch
     config
     DESTINATION share/${PROJECT_NAME}/
   )

   ament_package()
   ```

### Lab 3: Creating Launch Files

1. **Create a launch file** (`launch/robot_system.launch.py`):
   ```python
   import os
   from ament_index_python.packages import get_package_share_directory
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node, ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true')
           
       robot_namespace = DeclareLaunchArgument(
           'robot_namespace',
           default_value='robot1',
           description='Robot namespace for multi-robot systems')
           
       linear_velocity = DeclareLaunchArgument(
           'linear_velocity',
           default_value='0.5',
           description='Linear velocity of the robot')

       # Create nodes
       motion_controller_node = Node(
           package='complex_robot_controller',
           executable='motion_controller',
           name='motion_controller',
           parameters=[
               {'use_sim_time': LaunchConfiguration('use_sim_time')},
               {'linear_velocity': LaunchConfiguration('linear_velocity')}
           ],
           remappings=[
               ('/cmd_vel', [LaunchConfiguration('robot_namespace'), '/cmd_vel']),
               ('/scan', [LaunchConfiguration('robot_namespace'), '/scan'])
           ],
           output='screen'
       )

       # Create a diagnostics node
       diagnostics_node = Node(
           package='complex_robot_controller',
           executable='diagnostics_node',  # You would create this separately
           name='diagnostics',
           parameters=[
               {'use_sim_time': LaunchConfiguration('use_sim_time')}
           ],
           output='screen'
       )

       # Create launch description
       ld = LaunchDescription()
       
       # Add launch arguments
       ld.add_action(use_sim_time)
       ld.add_action(robot_namespace)
       ld.add_action(linear_velocity)
       
       # Add nodes
       ld.add_action(motion_controller_node)
       # ld.add_action(diagnostics_node)  # Uncomment once diagnostics_node exists
       
       return ld
   ```

2. **Create a launch file for multi-robot setup** (`launch/multi_robot.launch.py`):
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, TextSubstitution
   from launch_ros.actions import Node, PushRosNamespace


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true')
           
       num_robots = DeclareLaunchArgument(
           'num_robots',
           default_value='2',
           description='Number of robots to launch')

       # Create a group for each robot
       robot_groups = []
       for i in range(int(LaunchConfiguration('num_robots').perform({}))):
           robot_namespace = f'robot_{i}'
           
           # Create group action for each robot
           robot_group = GroupAction(
               actions=[
                   PushRosNamespace(robot_namespace),
                   IncludeLaunchDescription(
                       PythonLaunchDescriptionSource(
                           [os.path.join(
                               os.path.dirname(__file__),
                               'robot_system.launch.py'
                           )]
                       ),
                       launch_arguments={
                           'use_sim_time': LaunchConfiguration('use_sim_time'),
                           'robot_namespace': robot_namespace,
                           'linear_velocity': TextSubstitution(text=str(0.3 + i * 0.1))
                       }.items()
                   )
               ]
           )
           
           robot_groups.append(robot_group)

       # Create launch description
       ld = LaunchDescription()
       
       # Add launch arguments
       ld.add_action(use_sim_time)
       ld.add_action(num_robots)
       
       # Add robot groups
       for group in robot_groups:
           ld.add_action(group)
       
       return ld
   ```

### Lab 4: Building and Running the Package

1. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select complex_robot_controller --symlink-install
   ```

2. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

3. **Run the package**:
   ```bash
   ros2 run complex_robot_controller motion_controller
   ```

4. **Launch using the launch file**:
   ```bash
   ros2 launch complex_robot_controller robot_system.launch.py
   ```

## Runnable Code Example

Here's a complete example of a parameterized robot controller with configuration files:

**Configuration file** (`config/robot_params.yaml`):
```yaml
/**:
  ros__parameters:
    use_sim_time: false
    linear_velocity: 0.5
    angular_velocity: 0.3
    safety_distance: 0.8
    controller_freq: 10.0
    max_linear_speed: 1.0
    max_angular_speed: 1.5
    safety_threshold_multiplier: 1.2
```

**Complete Python node** (`complex_robot_controller/behavior_controller.py`):
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math
import time


class BehaviorController(Node):
    def __init__(self):
        super().__init__('behavior_controller')
        
        # Declare parameters with defaults
        self.declare_parameter('linear_velocity', 0.5)
        self.declare_parameter('angular_velocity', 0.3)
        self.declare_parameter('safety_distance', 0.8)
        self.declare_parameter('controller_freq', 10.0)
        self.declare_parameter('behavior_mode', 'explore')  # explore, follow, avoid
        
        # Get parameters
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.angular_velocity = self.get_parameter('angular_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.controller_freq = self.get_parameter('controller_freq').value
        self.behavior_mode = self.get_parameter('behavior_mode').value
        
        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)
        
        # Internal state
        self.obstacle_detected = False
        self.scan_data = None
        
        # Create timer
        self.timer = self.create_timer(1.0/self.controller_freq, self.control_loop)
        
        self.get_logger().info(f'Behavior Controller initialized with mode: {self.behavior_mode}')
        
    def scan_callback(self, msg):
        self.scan_data = msg
        # Process scan data to detect obstacles
        if len(msg.ranges) > 0:
            # Get front-facing ranges (±30 degrees)
            front_ranges = msg.ranges[len(msg.ranges)//2 - 15 : len(msg.ranges)//2 + 15]
            front_ranges = [r for r in front_ranges if not (math.isinf(r) or math.isnan(r))]
            
            if front_ranges:
                min_distance = min(front_ranges)
                self.obstacle_detected = min_distance < self.safety_distance
            else:
                self.obstacle_detected = False
    
    def control_loop(self):
        cmd = Twist()
        
        if self.scan_data is None:
            self.get_logger().warn('No scan data received yet')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif self.behavior_mode == 'explore':
            if self.obstacle_detected:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_velocity
                status_msg = String()
                status_msg.data = "Avoiding obstacle"
                self.status_pub.publish(status_msg)
            else:
                cmd.linear.x = self.linear_velocity
                cmd.angular.z = 0.0
                status_msg = String()
                status_msg.data = "Exploring"
                self.status_pub.publish(status_msg)
        elif self.behavior_mode == 'follow':
            # Simple wall-following behavior
            if self.scan_data.ranges:
                right_dist = self.scan_data.ranges[0] if not math.isinf(self.scan_data.ranges[0]) else float('inf')
                front_dist = self.scan_data.ranges[len(self.scan_data.ranges)//2] if not math.isinf(self.scan_data.ranges[len(self.scan_data.ranges)//2]) else float('inf')
                
                if front_dist < self.safety_distance:
                    cmd.angular.z = self.angular_velocity  # Turn left
                elif right_dist > 0.5:  # Too far from wall
                    cmd.angular.z = -self.angular_velocity  # Turn right
                elif right_dist < 0.2:  # Too close to wall
                    cmd.angular.z = self.angular_velocity  # Turn left
                else:
                    cmd.linear.x = self.linear_velocity  # Follow wall at distance
        elif self.behavior_mode == 'avoid':
            # More aggressive obstacle avoidance
            if self.obstacle_detected:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_velocity * 2.0  # Faster turning
            else:
                cmd.linear.x = self.linear_velocity
                cmd.angular.z = 0.0
        else:
            self.get_logger().warn(f'Unknown behavior mode: {self.behavior_mode}')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    behavior_controller = BehaviorController()
    
    try:
        rclpy.spin(behavior_controller)
    except KeyboardInterrupt:
        pass
    finally:
        behavior_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Launch file with parameters** (`launch/behavior.launch.py`):
```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')
    
    linear_velocity = DeclareLaunchArgument(
        'linear_velocity',
        default_value='0.5',
        description='Linear velocity of the robot')
    
    behavior_mode = DeclareLaunchArgument(
        'behavior_mode',
        default_value='explore',
        description='Behavior mode: explore, follow, or avoid')
    
    # Get config file path
    config = os.path.join(
        get_package_share_directory('complex_robot_controller'),
        'config',
        'robot_params.yaml'
    )
    
    # Create behavior controller node
    behavior_controller = Node(
        package='complex_robot_controller',
        executable='behavior_controller',  # This needs to be added as an entry point
        name='behavior_controller',
        parameters=[
            config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'linear_velocity': LaunchConfiguration('linear_velocity')},
            {'behavior_mode': LaunchConfiguration('behavior_mode')}
        ],
        remappings=[
            ('/cmd_vel', '/cmd_vel'),
            ('/scan', '/scan')
        ],
        output='screen'
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(use_sim_time)
    ld.add_action(linear_velocity)
    ld.add_action(behavior_mode)
    
    # Add nodes
    ld.add_action(behavior_controller)
    
    return ld
```

**Update setup.py** to include the Python executable:
```python
from setuptools import setup

package_name = 'complex_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch and config files
        ('share/' + package_name + '/launch', 
         ['launch/robot_system.launch.py', 'launch/multi_robot.launch.py', 'launch/behavior.launch.py']),
        ('share/' + package_name + '/config', 
         ['config/robot_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Complex robot controller with multiple components',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_controller = complex_robot_controller.motion_controller:main',
            'behavior_controller = complex_robot_controller.behavior_controller:main',
        ],
    },
)
```

## Mini-project

Create a complete navigation stack package with the following components:

1. A path planner node that takes goal poses and computes a path
2. A local controller that follows the path while avoiding obstacles
3. A parameter configuration system that allows tuning of navigation parameters
4. A launch file that starts the entire navigation stack with appropriate parameters
5. A configuration file in YAML format for all navigation parameters
6. Proper error handling and logging throughout the system

Create a README.md file that explains how to:
- Build the package
- Launch the navigation stack
- Configure different navigation parameters
- Test the system with simulated sensor data

## Summary

This chapter covered the complete workflow for building and launching ROS 2 packages:

- Package structure organization with proper CMakeLists.txt and package.xml configuration
- The colcon build system for compiling different types of packages
- Launch files using Python for complex system startup
- Parameter management using YAML configuration files
- Multi-robot system setup with namespaces
- Best practices for package organization and deployment

The launch system in ROS 2 provides powerful features for managing complex robotic systems, allowing for conditional launching, argument passing, and node composition. Properly configured packages with appropriate parameters and launch files are essential for developing robust and maintainable robotics applications.