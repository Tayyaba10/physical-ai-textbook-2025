---
title: Ch14 - Bipedal Locomotion & Balance Control
module: 3
chapter: 14
sidebar_label: Ch14: Bipedal Locomotion & Balance Control
description: Implementing bipedal locomotion and balance control using Isaac Sim and advanced control systems
tags: [bipedal, locomotion, balance, control, humanoid, robotics, isaac-sim, MPC]
difficulty: advanced
estimated_duration: 150
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Bipedal Locomotion & Balance Control

## Learning Outcomes
- Understand the principles of bipedal locomotion and balance control
- Implement advanced control strategies for humanoid robots
- Create dynamic walking gaits using model predictive control (MPC)
- Simulate bipedal robots in Isaac Sim with realistic physics
- Design and tune balance controllers for perturbation recovery
- Implement footstep planning and trajectory generation
- Evaluate stability and performance of bipedal locomotion systems

## Theory

### Bipedal Locomotion Fundamentals

Bipedal locomotion presents unique challenges compared to wheeled or quadrupedal locomotion. The system must maintain balance with only two points of contact with the ground while in motion.

<MermaidDiagram chart={`
graph TD;
    A[Bipedal Locomotion] --> B[Balance Control];
    A --> C[Locomotion Gait];
    A --> D[Stability Analysis];
    
    B --> E[Center of Mass Control];
    B --> F[Zero Moment Point];
    B --> G[Capture Point];
    
    C --> H[Walk Cycle];
    C --> I[Step Planning];
    C --> J[Trajectory Generation];
    
    D --> K[Stability Margins];
    D --> L[Robustness Analysis];
    D --> M[Recovery Strategies];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style B fill:#2196F3,stroke:#0D47A1,color:#fff;
    style C fill:#FF9800,stroke:#E65100,color:#fff;
`} />

### Balance Control Approaches

**Zero Moment Point (ZMP)**: A critical concept in bipedal robotics representing the point on the ground where the net moment of the ground reaction force is zero. Maintaining ZMP within the support polygon is essential for stability.

**Capture Point (CP)**: The point where a biped can come to rest given its current velocity. The capture point is used to design stable walking patterns.

**Linear Inverted Pendulum Model (LIPM)**: Simplifies the robot to a point mass with constant height, allowing for analytical solutions to balance control.

### Gait Generation and Control

Bipedal gaits involve complex coordination of joints and careful foot placement. Key phases include:

- **Single Support**: One foot in contact with ground
- **Double Support**: Both feet in contact during step transitions
- **Swing Phase**: Non-stance leg swinging forward

### Model Predictive Control (MPC) for Walking

MPC is particularly well-suited for bipedal walking due to its ability to handle constraints and predict future states. The controller optimizes a cost function over a prediction horizon while respecting system dynamics and constraints.

## Step-by-Step Labs

### Lab 1: Setting up a Bipedal Robot Model in Isaac Sim

1. **Create a simplified bipedal robot URDF** (`bipedal_robot.urdf`):
   ```xml
   <?xml version="1.0" ?>
   <robot name="simple_biped">
     <material name="blue">
       <color rgba="0.0 0.0 1.0 1.0"/>
     </material>
     <material name="red">
       <color rgba="1.0 0.0 0.0 1.0"/>
     </material>
     <material name="white">
       <color rgba="1.0 1.0 1.0 1.0"/>
     </material>

     <!-- Torso -->
     <link name="torso">
       <inertial>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <mass value="10.0"/>
         <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.3"/>
       </inertial>
       <visual>
         <origin xyz="0 0 0.5" rpy="0 0 0"/>
         <geometry>
           <box size="0.3 0.3 1.0"/>
         </geometry>
         <material name="white"/>
       </visual>
       <collision>
         <origin xyz="0 0 0.5" rpy="0 0 0"/>
         <geometry>
           <box size="0.3 0.3 1.0"/>
         </geometry>
       </collision>
     </link>

     <!-- Head -->
     <link name="head">
       <inertial>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <mass value="2.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
       <visual>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <geometry>
           <sphere radius="0.15"/>
         </geometry>
         <material name="red"/>
       </visual>
       <collision>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <geometry>
           <sphere radius="0.15"/>
         </geometry>
       </collision>
     </link>

     <joint name="torso_head" type="fixed">
       <parent link="torso"/>
       <child link="head"/>
       <origin xyz="0 0 1.0" rpy="0 0 0"/>
     </joint>

     <!-- Left Hip -->
     <link name="left_hip">
       <inertial>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <mass value="2.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.02"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_hip_joint" type="revolute">
       <parent link="torso"/>
       <child link="left_hip"/>
       <origin xyz="0 0.15 0" rpy="0 0 0"/>
       <axis xyz="1 0 0"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
     </joint>

     <!-- Left Upper Leg -->
     <link name="left_upper_leg">
       <inertial>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <mass value="3.0"/>
         <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.03"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.06" length="0.4"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.06" length="0.4"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_knee_joint" type="revolute">
       <parent link="left_hip"/>
       <child link="left_upper_leg"/>
       <origin xyz="0 0 -0.3" rpy="0 0 0"/>
       <axis xyz="1 0 0"/>
       <limit lower="0" upper="2.35" effort="100" velocity="1.0"/>
     </joint>

     <!-- Left Lower Leg -->
     <link name="left_lower_leg">
       <inertial>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <mass value="2.5"/>
         <inertia ixx="0.15" ixy="0.0" ixz="0.0" iyy="0.15" iyz="0.0" izz="0.02"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.4"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.4"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_ankle_joint" type="revolute">
       <parent link="left_upper_leg"/>
       <child link="left_lower_leg"/>
       <origin xyz="0 0 -0.4" rpy="0 0 0"/>
       <axis xyz="1 0 0"/>
       <limit lower="-0.78" upper="0.78" effort="50" velocity="1.0"/>
     </joint>

     <!-- Left Foot -->
     <link name="left_foot">
       <inertial>
         <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
       </inertial>
       <visual>
         <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
         <material name="red"/>
       </visual>
       <collision>
         <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_foot_joint" type="fixed">
       <parent link="left_lower_leg"/>
       <child link="left_foot"/>
       <origin xyz="0 0 -0.4" rpy="0 0 0"/>
     </joint>

     <!-- Right Leg (mirror of left) -->
     <link name="right_hip">
       <inertial>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <mass value="2.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.02"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_hip_joint" type="revolute">
       <parent link="torso"/>
       <child link="right_hip"/>
       <origin xyz="0 -0.15 0" rpy="0 0 0"/>
       <axis xyz="1 0 0"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
     </joint>

     <link name="right_upper_leg">
       <inertial>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <mass value="3.0"/>
         <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.03"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.06" length="0.4"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.06" length="0.4"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_knee_joint" type="revolute">
       <parent link="right_hip"/>
       <child link="right_upper_leg"/>
       <origin xyz="0 0 -0.3" rpy="0 0 0"/>
       <axis xyz="1 0 0"/>
       <limit lower="0" upper="2.35" effort="100" velocity="1.0"/>
     </joint>

     <link name="right_lower_leg">
       <inertial>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <mass value="2.5"/>
         <inertia ixx="0.15" ixy="0.0" ixz="0.0" iyy="0.15" iyz="0.0" izz="0.02"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.4"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.2" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.05" length="0.4"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_ankle_joint" type="revolute">
       <parent link="right_upper_leg"/>
       <child link="right_lower_leg"/>
       <origin xyz="0 0 -0.4" rpy="0 0 0"/>
       <axis xyz="1 0 0"/>
       <limit lower="-0.78" upper="0.78" effort="50" velocity="1.0"/>
     </joint>

     <link name="right_foot">
       <inertial>
         <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
       </inertial>
       <visual>
         <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
         <material name="red"/>
       </visual>
       <collision>
         <origin xyz="0.05 0 -0.025" rpy="0 0 0"/>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_foot_joint" type="fixed">
       <parent link="right_lower_leg"/>
       <child link="right_foot"/>
       <origin xyz="0 0 -0.4" rpy="0 0 0"/>
     </joint>

     <!-- Arms for additional balance control -->
     <link name="left_upper_arm">
       <inertial>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <mass value="1.5"/>
         <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_shoulder_joint" type="revolute">
       <parent link="torso"/>
       <child link="left_upper_arm"/>
       <origin xyz="0.1 0.15 0.7" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
     </joint>

     <link name="right_upper_arm">
       <inertial>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <mass value="1.5"/>
         <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
       </inertial>
       <visual>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
         <material name="blue"/>
       </visual>
       <collision>
         <origin xyz="0 0 -0.15" rpy="0 0 0"/>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_shoulder_joint" type="revolute">
       <parent link="torso"/>
       <child link="right_upper_arm"/>
       <origin xyz="0.1 -0.15 0.7" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
     </joint>
   </robot>
   ```

2. **Load the robot into Isaac Sim**:
   ```python
   # biped_loader.py
   import omni
   from pxr import Gf, UsdGeom
   from omni.isaac.core import World
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.core.utils.prims import get_prim_at_path
   from omni.isaac.core.utils.viewports import set_camera_view
   import numpy as np

   def load_biped_robot():
       """Load the biped robot into Isaac Sim"""
       # Add the robot model to the stage
       robot_path = "/World/BipedRobot"
       # In practice, you would load the URDF using Isaac Sim's URDF import functionality
       # For this example, we'll create a simplified representation
       
       # Create the main body
       from omni.isaac.core.utils.prims import create_prim
       create_prim(
           "/World/BipedRobot/Torso",
           "Cuboid",
           position=(0, 0, 1.0),
           attributes={"size": 0.3}
       )
       
       # Set up physics
       from omni.physx.scripts import physicsUtils
       stage = omni.usd.get_context().get_stage()
       default_prim = stage.GetDefaultPrim()
       
       # Add ground plane
       create_prim(
           "/World/GroundPlane",
           "Plane",
           position=(0, 0, 0),
           attributes={"size": 10.0}
       )
       
       # Set camera view
       set_camera_view(eye=(5, 5, 3), target=(0, 0, 1))
       
       print("Biped robot loaded into Isaac Sim")

   # Run when in Isaac Sim environment
   if __name__ == "__main__":
       load_biped_robot()
   ```

### Lab 2: Implementing Inverted Pendulum Balance Controller

1. **Create a Linear Inverted Pendulum Model (LIPM) controller** (`lipm_controller.py`):
   ```python
   #!/usr/bin/env python3

   import numpy as np
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState, Imu
   from geometry_msgs.msg import Point, Vector3
   from std_msgs.msg import Float64MultiArray
   import math

   class LIPMController(Node):
       def __init__(self):
           super().__init__('lipm_controller')
           
           # LIPM parameters
           self.height = 0.84  # Center of mass height (meters)
           self.gravity = 9.81  # Gravity (m/s²)
           self.omega = math.sqrt(self.gravity / self.height)  # Natural frequency
           
           # Robot state
           self.com_position = np.array([0.0, 0.0, self.height])  # Center of mass position
           self.com_velocity = np.array([0.0, 0.0, 0.0])          # Center of mass velocity
           self.support_foot = "left"  # Current support foot
           self.step_frequency = 0.5  # Steps per second
           
           # Walking parameters
           self.step_length = 0.3    # Forward step length (m)
           self.step_width = 0.2     # Lateral step width (m)
           self.nominal_com_height = self.height
           
           # PID controllers for balance
           self.kp_com_x = 8.0
           self.kd_com_x = 2.0
           self.kp_com_y = 8.0
           self.kd_com_y = 2.0
           
           # Subscribers
           self.joint_state_sub = self.create_subscription(
               JointState, '/joint_states', self.joint_state_callback, 10)
           self.imu_sub = self.create_subscription(
               Imu, '/imu', self.imu_callback, 10)
           
           # Publishers
           self.com_pub = self.create_publisher(
               Point, '/center_of_mass', 10)
           self.zmp_pub = self.create_publisher(
               Point, '/zmp', 10)
           self.footstep_pub = self.create_publisher(
               Float64MultiArray, '/desired_footstep', 10)
           self.com_ref_pub = self.create_publisher(
               Point, '/com_reference', 10)
           
           # Timer for control loop
           self.control_timer = self.create_timer(0.01, self.balance_control_step)  # 100 Hz
           
           self.get_logger().info('LIPM Balance Controller initialized')
       
       def joint_state_callback(self, msg):
           """Update robot state from joint positions and velocities"""
           # In a real implementation, this would compute CoM from joint angles
           # For this example, we'll just track a moving CoM reference
           pass
       
       def imu_callback(self, msg):
           """Update CoM estimate from IMU data"""
           # Integrate acceleration to estimate velocity and position
           # This is a simplified approach - in reality you'd use sensor fusion
           linear_acc = msg.linear_acceleration
           angular_vel = msg.angular_velocity
           
           # Update CoM velocity (would require more complex integration in practice)
           dt = 0.01  # Assuming 100Hz control loop
           self.com_velocity[0] += linear_acc.x * dt
           self.com_velocity[1] += linear_acc.y * dt
           self.com_velocity[2] += (linear_acc.z - self.gravity) * dt
           
           # Update CoM position
           self.com_position += self.com_velocity * dt
           
           # Also update height from IMU if necessary
           self.com_position[2] = max(self.com_position[2], self.nominal_com_height)
       
       def compute_zmp(self):
           """Compute Zero Moment Point from CoM state"""
           # ZMP_x = com_x - (h/g) * com_acc_x
           # ZMP_y = com_y - (h/g) * com_acc_y
           
           # For this simplified version, we'll use the current CoM position
           # adjusted by the inverted pendulum dynamics
           zmp = Point()
           zmp.x = self.com_position[0] - (self.height / self.gravity) * (self.com_velocity[0] * self.omega)
           zmp.y = self.com_position[1] - (self.height / self.gravity) * (self.com_velocity[1] * self.omega)
           zmp.z = 0.0  # ZMP is on ground plane
           
           return zmp
       
       def compute_capture_point(self):
           """Compute Capture Point for balance recovery"""
           # Capture Point = com_pos + com_vel/omega
           cp_x = self.com_position[0] + self.com_velocity[0] / self.omega
           cp_y = self.com_position[1] + self.com_velocity[1] / self.omega
           
           return np.array([cp_x, cp_y])
       
       def balance_control_step(self):
           """Main balance control step"""
           # Compute ZMP
           zmp = self.compute_zmp()
           self.zmp_pub.publish(zmp)
           
           # Publish CoM
           com_msg = Point()
           com_msg.x = float(self.com_position[0])
           com_msg.y = float(self.com_position[1])
           com_msg.z = float(self.com_position[2])
           self.com_pub.publish(com_msg)
           
           # Compute reference CoM based on desired walking pattern
           self.compute_com_reference()
           
           # Generate footsteps based on capture point
           self.generate_footsteps()
           
           # Log current state
           self.get_logger().debug(
               f'CoM: ({self.com_position[0]:.3f}, {self.com_position[1]:.3f}, {self.com_position[2]:.3f}), '
               f'ZMP: ({zmp.x:.3f}, {zmp.y:.3f})'
           )
       
       def compute_com_reference(self):
           """Compute desired CoM position for walking"""
           # For walking, we want to oscillate CoM laterally and move forward
           current_time = self.get_clock().now().nanoseconds / 1e9
           
           # Lateral oscillation for walking
           lateral_oscillation = 0.05 * math.sin(2 * math.pi * self.step_frequency * current_time)
           
           # Forward progression
           forward_progress = self.step_length * self.step_frequency * current_time
           
           # Update reference
           self.com_ref = np.array([
               forward_progress,  # Forward movement
               lateral_oscillation if self.support_foot == "left" else -lateral_oscillation,  # Lateral movement
               self.nominal_com_height
           ])
           
           # Publish reference
           ref_msg = Point()
           ref_msg.x = float(self.com_ref[0])
           ref_msg.y = float(self.com_ref[1])
           ref_msg.z = float(self.com_ref[2])
           self.com_ref_pub.publish(ref_msg)
       
       def generate_footsteps(self):
           """Generate desired footsteps based on capture point"""
           capture_point = self.compute_capture_point()
           
           # Simple strategy: place foot near capture point with some offset
           # to ensure ZMP remains in support polygon
           if self.support_foot == "left":
               # Right foot should go near capture point
               target_pos = capture_point + np.array([0.1, -self.step_width/2])
               self.support_foot = "right"
           else:
               # Left foot should go near capture point
               target_pos = capture_point + np.array([0.1, self.step_width/2])
               self.support_foot = "left"
           
           # Publish desired footstep
           footstep_msg = Float64MultiArray()
           footstep_msg.data = [float(target_pos[0]), float(target_pos[1]), 0.0]
           self.footstep_pub.publish(footstep_msg)
           
           self.get_logger().info(f'Next footstep: ({target_pos[0]:.3f}, {target_pos[1]:.3f}) on {self.support_foot} foot')

   def main(args=None):
       rclpy.init(args=args)
       controller = LIPMController()
       
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

### Lab 3: Model Predictive Control for Walking

1. **Create an MPC controller for bipedal walking** (`mpc_walking_controller.py`):
   ```python
   #!/usr/bin/env python3

   import numpy as np
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState, Imu
   from geometry_msgs.msg import Point, Twist, PoseStamped
   from std_msgs.msg import Float64MultiArray, String
   import math
   import cvxpy as cp  # Convex optimization library for MPC

   class MPCWalkingController(Node):
       def __init__(self):
           super().__init__('mpc_walking_controller')
           
           # MPC parameters
           self.prediction_horizon = 20      # Number of steps to predict
           self.control_horizon = 5          # Number of steps to optimize
           self.dt = 0.1                     # Time step (seconds)
           self.height = 0.84                # CoM height (m)
           self.gravity = 9.81               # Gravity (m/s²)
           self.omega = math.sqrt(self.gravity / self.height)
           
           # Walking parameters
           self.step_length = 0.3            # Step length (m)
           self.step_width = 0.2             # Step width (m)
           self.step_duration = 1.0          # Time per step (s)
           
           # Current state [x, y, xdot, ydot]
           self.state = np.array([0.0, 0.0, 0.0, 0.0])
           
           # MPC weights
           self.Q = np.diag([10.0, 10.0, 1.0, 1.0])  # State cost matrix
           self.R = 0.1  # Control cost weight
           self.Q_final = np.diag([50.0, 50.0, 5.0, 5.0])  # Terminal cost
           
           # Footstep plan
           self.footstep_plan = []
           self.current_step_idx = 0
           self.support_foot = "left"
           
           # Subscribers
           self.joint_state_sub = self.create_subscription(
               JointState, '/joint_states', self.joint_state_callback, 10)
           self.imu_sub = self.create_subscription(
               Imu, '/imu', self.imu_callback, 10)
           self.footstep_sub = self.create_subscription(
               Float64MultiArray, '/desired_footsteps', self.footstep_callback, 10)
           
           # Publishers
           self.zmp_pub = self.create_publisher(
               Point, '/zmp_reference', 10)
           self.com_traj_pub = self.create_publisher(
               Float64MultiArray, '/com_trajectory', 10)
           self.mpc_status_pub = self.create_publisher(
               String, '/mpc_status', 10)
           self.com_vel_pub = self.create_publisher(
               Twist, '/com_velocity', 10)
           
           # Timer for control loop
           self.control_timer = self.create_timer(0.05, self.mpc_control_step)  # 20 Hz
           
           self.get_logger().info('MPC Walking Controller initialized')
       
       def joint_state_callback(self, msg):
           """Update robot state from joint positions and velocities"""
           # In a real implementation, this would compute CoM from joint angles
           # This is a simplified approach
           pass
       
       def imu_callback(self, msg):
           """Update state estimate from IMU"""
           # This would implement more sophisticated state estimation in practice
           pass
       
       def footstep_callback(self, msg):
           """Update footstep plan from planner"""
           if len(msg.data) >= 2:
               # Add new footstep to plan
               new_step = (msg.data[0], msg.data[1], self.get_clock().now().seconds_nanoseconds()[0] + self.step_duration)
               self.footstep_plan.append(new_step)
               
               # Keep only future footsteps
               current_time = self.get_clock().now().seconds_nanoseconds()[0]
               self.footstep_plan = [step for step in self.footstep_plan if step[2] > current_time]
               
               self.get_logger().info(f'Updated footstep plan. {len(self.footstep_plan)} steps in plan.')
       
       def setup_mpc_problem(self, x0, reference_trajectory):
           """Set up the MPC optimization problem"""
           N = self.prediction_horizon
           
           # State variables [x, y, xdot, ydot] for each time step
           X = cp.Variable((4, N+1))
           # Control variables [zmp_x, zmp_y] for each time step
           U = cp.Variable((2, N))
           
           # System dynamics matrix for LIPM
           A_cont = np.array([
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [self.omega**2, 0, 0, 0],
               [0, self.omega**2, 0, 0]
           ])
           
           # Discretize system
           I = np.eye(4)
           A = I + self.dt * A_cont
           B = self.dt * np.array([
               [0, 0],
               [0, 0],
               [self.omega**2, 0],
               [0, self.omega**2]
           ])
           
           # Cost function
           cost = 0
           
           # State and control penalties
           for k in range(N):
               # State cost (tracking reference)
               state_error = X[:, k] - reference_trajectory[k]
               cost += cp.quad_form(state_error, self.Q)
               
               # Control effort penalty
               cost += self.R * cp.sum_squares(U[:, k])
           
           # Terminal cost
           final_error = X[:, N] - reference_trajectory[N-1]  # Use last reference
           cost += cp.quad_form(final_error, self.Q_final)
           
           # Constraints
           constraints = []
           
           # Initial state
           constraints.append(X[:, 0] == x0)
           
           # Dynamics constraints
           for k in range(N):
               constraints.append(X[:, k+1] == A @ X[:, k] + B @ U[:, k])
           
           # ZMP constraints (must be within foot for stability)
           for k in range(N):
               # For simplicity, assume feet remain fixed for this prediction
               # In reality, you'd have moving support polygons
               constraints.append(cp.norm(U[:, k], 'inf') <= 0.3)  # ZMP within reasonable bounds
           
           # Create and solve problem
           problem = cp.Problem(cp.Minimize(cost), constraints)
           
           return problem, X, U
       
       def solve_mpc(self):
           """Solve the MPC optimization problem"""
           # Create reference trajectory
           reference_trajectory = self.generate_reference_trajectory()
           
           # Set up optimization problem
           problem, X, U = self.setup_mpc_problem(self.state, reference_trajectory)
           
           try:
               # Solve the problem
               problem.solve(solver=cp.CLARABEL, verbose=False)
               
               if problem.status not in ["infeasible", "unbounded"]:
                   # Extract optimal control
                   optimal_controls = U.value
                   optimal_states = X.value
                   
                   # Return the first control input
                   if optimal_controls is not None and optimal_controls.shape[1] > 0:
                       first_control = optimal_controls[:, 0]
                       return first_control, optimal_states
                   else:
                       self.get_logger().warn('MPC solution invalid')
                       return np.array([0.0, 0.0]), self.state.reshape(4,1)
               else:
                   self.get_logger().warn(f'MPC problem status: {problem.status}')
                   # Return zero control if optimization fails
                   return np.array([0.0, 0.0]), self.state.reshape(4,1)
           except Exception as e:
               self.get_logger().error(f'MPC optimization failed: {str(e)}')
               return np.array([0.0, 0.0]), self.state.reshape(4,1)
       
       def generate_reference_trajectory(self):
           """Generate a reference trajectory for the MPC"""
           N = self.prediction_horizon
           ref_traj = np.zeros((4, N+1))
           
           # For walking, we want to follow a path while maintaining balance
           current_time = self.get_clock().now().seconds_nanoseconds()[0]
           
           for k in range(N+1):
               t = current_time + k * self.dt
               
               # Forward progression
               x_ref = self.step_length * t / self.step_duration
               
               # Lateral oscillation (for walking gait)
               if self.support_foot == "left":
                   y_ref = 0.1 * math.sin(2 * math.pi * t / (2 * self.step_duration))
               else:
                   y_ref = -0.1 * math.sin(2 * math.pi * t / (2 * self.step_duration))
               
               # For now, assume zero velocity in reference (this is simplified)
               ref_traj[:, k] = [x_ref, y_ref, 0, 0]
           
           return ref_traj
       
       def mpc_control_step(self):
           """Main MPC control step"""
           # Solve MPC problem
           zmp_cmd, state_traj = self.solve_mpc()
           
           # Publish ZMP command
           zmp_msg = Point()
           zmp_msg.x = float(zmp_cmd[0])
           zmp_msg.y = float(zmp_cmd[1])
           zmp_msg.z = 0.0  # ZMP is on ground plane
           self.zmp_pub.publish(zmp_msg)
           
           # Publish CoM trajectory
           traj_msg = Float64MultiArray()
           traj_msg.data = state_traj.flatten().tolist()
           self.com_traj_pub.publish(traj_msg)
           
           # Publish CoM velocity
           vel_msg = Twist()
           vel_msg.linear.x = float(self.state[2])  # x velocity
           vel_msg.linear.y = float(self.state[3])  # y velocity
           self.com_vel_pub.publish(vel_msg)
           
           # Update status
           status_msg = String()
           status_msg.data = f"MPC Active - State: [{self.state[0]:.3f}, {self.state[1]:.3f}, {self.state[2]:.3f}, {self.state[3]:.3f}]"
           self.mpc_status_pub.publish(status_msg)
           
           # Simple integration to update state (in reality, this would come from sensors)
           A_cont = np.array([
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [self.omega**2, 0, 0, 0],
               [0, self.omega**2, 0, 0]
           ])
           
           A = np.eye(4) + self.dt * A_cont
           B = self.dt * np.array([
               [0, 0],
               [0, 0],
               [self.omega**2, 0],
               [0, self.omega**2]
           ])
           
           # Update state: x_{k+1} = A*x_k + B*u_k
           self.state = A @ self.state + B @ zmp_cmd
           
           self.get_logger().debug(f'MPC Step - ZMP Cmd: ({zmp_cmd[0]:.3f}, {zmp_cmd[1]:.3f})')

   def main(args=None):
       rclpy.init(args=args)
       controller = MPCWalkingController()
       
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

## Runnable Code Example

Here's a complete bipedal walking simulation with balance control:

```python
#!/usr/bin/env python3
# complete_bipedal_controller.py

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class CompleteBipedalController(Node):
    def __init__(self):
        super().__init__('complete_bipedal_controller')
        
        # Robot configuration
        self.hip_offset = 0.15  # Lateral hip offset (m)
        self.leg_length = 0.8   # Total leg length (m)
        self.robot_height = 0.84  # Stance height (m)
        
        # Walking parameters
        self.step_length = 0.3
        self.step_height = 0.05
        self.step_duration = 1.0
        self.nominal_width = 0.2
        
        # State variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.left_foot_pos = np.array([0.0, self.hip_offset, 0.0])
        self.right_foot_pos = np.array([0.0, -self.hip_offset, 0.0])
        self.com_pos = np.array([0.0, 0.0, self.robot_height])
        
        # Walking state machine
        self.phase = "stance"  # "stance", "left_swing", "right_swing", "double_support"
        self.swing_leg = "right"
        self.step_phase = 0.0  # 0.0 to 1.0
        self.last_step_time = self.get_clock().now().nanoseconds / 1e9
        
        # Timing
        self.dt = 0.01  # 100 Hz control loop
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        # Publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10)
        self.com_pub = self.create_publisher(
            Point, '/center_of_mass', 10)
        self.state_pub = self.create_publisher(
            String, '/walking_state', 10)
        self.foot_pos_pub = self.create_publisher(
            Float64MultiArray, '/foot_positions', 10)
        
        # Timer for control loop
        self.control_timer = self.create_timer(self.dt, self.control_step)
        
        self.get_logger().info('Complete Bipedal Controller initialized')
    
    def joint_state_callback(self, msg):
        """Update robot state from joint feedback"""
        # In a real implementation, this would integrate joint positions
        # to update robot pose estimates
        pass
    
    def compute_inverse_kinematics(self, foot_pos, leg_side):
        """Compute joint angles for a given foot position"""
        # Simplified inverse kinematics for planar leg
        # foot_pos is [x, y, z] relative to hip
        
        # Calculate hip-to-foot vector
        dx = foot_pos[0]
        dy = foot_pos[1] - (self.hip_offset if leg_side == "left" else -self.hip_offset)
        dz = foot_pos[2] - self.robot_height  # Hip height is at torso level
        
        # Calculate leg length
        leg_length_sq = dx*dx + dy*dy + dz*dz
        leg_length = math.sqrt(leg_length_sq)
        
        # Check if position is reachable
        if leg_length > 2 * self.leg_length:
            self.get_logger().warn(f'Foot position {foot_pos} not reachable')
            return [0.0, 0.0, 0.0]  # Return neutral position
        
        # Hip pitch (sagittal plane)
        hip_pitch = math.atan2(-dz, math.sqrt(dx*dx + dy*dy))
        
        # Hip roll (coronal plane)
        hip_roll = math.atan2(dy, math.sqrt(dx*dx + dz*dz))
        
        # Knee angle (to straighten leg)
        # Using law of cosines to find knee angle
        a = self.leg_length
        b = self.leg_length
        c = leg_length
        
        if a > 0 and b > 0 and c > 0:
            knee_angle = math.pi - math.acos((a*a + b*b - c*c) / (2*a*b))
        else:
            knee_angle = 0.0
        
        return [hip_roll, hip_pitch, knee_angle]
    
    def generate_walking_pattern(self):
        """Generate walking pattern for the current phase"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        time_in_step = current_time - self.last_step_time
        self.step_phase = time_in_step / self.step_duration
        
        if self.step_phase >= 1.0:
            # Step complete, switch legs
            if self.swing_leg == "right":
                self.swing_leg = "left"
                self.left_foot_pos[0] = self.robot_x + self.step_length
                self.left_foot_pos[1] = -self.hip_offset if self.phase == "stance" else self.hip_offset
            else:
                self.swing_leg = "right"
                self.right_foot_pos[0] = self.robot_x + self.step_length
                self.right_foot_pos[1] = self.hip_offset if self.phase == "stance" else -self.hip_offset
            
            self.last_step_time = current_time
            self.step_phase = 0.0
            self.robot_x += self.step_length  # Update robot position
        
        # Calculate swing foot trajectory
        if self.swing_leg == "right":
            # Right foot is swinging forward
            swing_progress = min(self.step_phase * 2, 1.0)  # Swing phase is first half of step
            target_x = self.robot_x + swing_progress * self.step_length
            target_y = -self.hip_offset  # Start and end at nominal width
            
            # Vertical trajectory (cycloid or similar)
            if swing_progress < 0.5:
                # Rise
                swing_vertical = swing_progress * 2 * self.step_height
            else:
                # Fall
                fall_progress = (swing_progress - 0.5) * 2
                swing_vertical = self.step_height * (1 - fall_progress)
            
            self.right_foot_pos = np.array([target_x, target_y, swing_vertical])
        else:
            # Left foot is swinging
            swing_progress = min(self.step_phase * 2, 1.0)
            target_x = self.robot_x + swing_progress * self.step_length
            target_y = self.hip_offset
            
            if swing_progress < 0.5:
                swing_vertical = swing_progress * 2 * self.step_height
            else:
                fall_progress = (swing_progress - 0.5) * 2
                swing_vertical = self.step_height * (1 - fall_progress)
            
            self.left_foot_pos = np.array([target_x, target_y, swing_vertical])
        
        # Update CoM position for balance (simplified)
        # Keep CoM between feet
        avg_foot_x = (self.left_foot_pos[0] + self.right_foot_pos[0]) / 2
        avg_foot_y = (self.left_foot_pos[1] + self.right_foot_pos[1]) / 2
        
        # Add some lateral offset toward stance foot
        stance_foot = "left" if self.swing_leg == "right" else "right"
        if stance_foot == "left":
            com_offset_y = -0.02  # Slightly toward left foot
        else:
            com_offset_y = 0.02   # Slightly toward right foot
        
        self.com_pos[0] = avg_foot_x
        self.com_pos[1] = avg_foot_y + com_offset_y
        self.com_pos[2] = self.robot_height
    
    def publish_joint_commands(self):
        """Publish joint trajectory commands"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'right_hip_joint', 'right_knee_joint',
            'left_ankle_joint', 'right_ankle_joint', 'left_shoulder_joint', 'right_shoulder_joint'
        ]
        
        point = JointTrajectoryPoint()
        
        # Compute joint angles using inverse kinematics
        left_angles = self.compute_inverse_kinematics(self.left_foot_pos, "left")
        right_angles = self.compute_inverse_kinematics(self.right_foot_pos, "right")
        
        # For this simplified version, we'll use just hip and knee angles
        # More complex implementation would include ankle, etc.
        point.positions = [
            left_angles[0],           # left_hip_roll
            left_angles[1],           # left_hip_pitch
            right_angles[0],          # right_hip_roll
            right_angles[1],          # right_hip_pitch
            left_angles[2],           # left_knee
            right_angles[2],          # right_knee
            0.0,                      # left_shoulder (for balance)
            0.0                       # right_shoulder (for balance)
        ]
        
        # Set zero velocities and accelerations
        point.velocities = [0.0] * len(point.positions)
        point.accelerations = [0.0] * len(point.positions)
        
        # Set time from start
        point.time_from_start = Duration(sec=0, nanosec=int(self.dt * 1e9))
        
        trajectory_msg.points.append(point)
        self.joint_trajectory_pub.publish(trajectory_msg)
    
    def broadcast_transforms(self):
        """Broadcast TF transforms for visualization"""
        current_time = self.get_clock().now()
        
        # Robot base frame
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.com_pos[0])
        t.transform.translation.y = float(self.com_pos[1])
        t.transform.translation.z = float(self.com_pos[2])
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        
        # Left foot frame
        t.child_frame_id = 'left_foot'
        t.transform.translation.x = float(self.left_foot_pos[0])
        t.transform.translation.y = float(self.left_foot_pos[1])
        t.transform.translation.z = float(self.left_foot_pos[2])
        self.tf_broadcaster.sendTransform(t)
        
        # Right foot frame
        t.child_frame_id = 'right_foot'
        t.transform.translation.x = float(self.right_foot_pos[0])
        t.transform.translation.y = float(self.right_foot_pos[1])
        t.transform.translation.z = float(self.right_foot_pos[2])
        self.tf_broadcaster.sendTransform(t)
    
    def control_step(self):
        """Main control step"""
        # Generate walking pattern
        self.generate_walking_pattern()
        
        # Publish commands
        self.publish_joint_commands()
        
        # Broadcast transforms
        self.broadcast_transforms()
        
        # Publish state information
        com_msg = Point()
        com_msg.x = float(self.com_pos[0])
        com_msg.y = float(self.com_pos[1])
        com_msg.z = float(self.com_pos[2])
        self.com_pub.publish(com_msg)
        
        # Publish foot positions
        foot_pos_msg = Float64MultiArray()
        foot_pos_msg.data = [
            float(self.left_foot_pos[0]), float(self.left_foot_pos[1]), float(self.left_foot_pos[2]),
            float(self.right_foot_pos[0]), float(self.right_foot_pos[1]), float(self.right_foot_pos[2])
        ]
        self.foot_pos_pub.publish(foot_pos_msg)
        
        # Publish state
        state_msg = String()
        state_msg.data = f"Phase: {self.phase}, Swing: {self.swing_leg}, Step: {self.step_phase:.2f}"
        self.state_pub.publish(state_msg)
        
        # Log progress
        self.get_logger().debug(
            f'Walking: CoM=({self.com_pos[0]:.3f}, {self.com_pos[1]:.3f}, {self.com_pos[2]:.3f}), '
            f'Left_foot=({self.left_foot_pos[0]:.3f}, {self.left_foot_pos[1]:.3f}), '
            f'Right_foot=({self.right_foot_pos[0]:.3f}, {self.right_foot_pos[1]:.3f})'
        )

class BalanceRecoveryController(Node):
    def __init__(self):
        super().__init__('balance_recovery_controller')
        
        # Subscribe to CoM and foot position data
        self.com_sub = self.create_subscription(
            Point, '/center_of_mass', self.com_callback, 10)
        self.foot_pos_sub = self.create_subscription(
            Float64MultiArray, '/foot_positions', self.foot_pos_callback, 10)
        
        # Publisher for recovery commands
        self.recovery_cmd_pub = self.create_publisher(
            Twist, '/recovery_command', 10)
        
        # State variables
        self.com_pos = np.array([0.0, 0.0, 0.84])
        self.left_foot = np.array([0.0, 0.15, 0.0])
        self.right_foot = np.array([0.0, -0.15, 0.0])
        self.recovery_active = False
        
        # Recovery parameters
        self.stability_threshold = 0.15  # meters from foot center
        self.recovery_gain = 2.0
        
        # Timer for recovery check
        self.recovery_timer = self.create_timer(0.05, self.check_stability)  # 20 Hz
        
        self.get_logger().info('Balance Recovery Controller initialized')
    
    def com_callback(self, msg):
        self.com_pos = np.array([msg.x, msg.y, msg.z])
    
    def foot_pos_callback(self, msg):
        if len(msg.data) >= 6:
            self.left_foot = np.array(msg.data[0:3])
            self.right_foot = np.array(msg.data[3:6])
    
    def check_stability(self):
        """Check if robot is in danger of falling and initiate recovery"""
        # Calculate ZMP (simplified as CoM position projected to ground)
        zmp_x = self.com_pos[0]
        zmp_y = self.com_pos[1]
        
        # Determine support polygon (convex hull of feet)
        if self.left_foot[2] < 0.01 and self.right_foot[2] < 0.01:
            # Both feet on ground - double support
            min_y = min(self.left_foot[1], self.right_foot[1])
            max_y = max(self.left_foot[1], self.right_foot[1])
            center_y = (min_y + max_y) / 2
            support_width = max_y - min_y
        elif self.left_foot[2] < 0.01:
            # Left foot support
            center_y = self.left_foot[1]
            support_width = 0.1  # Approximate foot size
        elif self.right_foot[2] < 0.01:
            # Right foot support
            center_y = self.right_foot[1]
            support_width = 0.1  # Approximate foot size
        else:
            # No support - critical condition
            self.get_logger().error('NO FOOT SUPPORT - EMERGENCY!')
            return
        
        # Check if ZMP is outside support region
        margin = support_width / 2 - self.stability_threshold
        if abs(zmp_y - center_y) > margin:
            # Robot is in danger of falling laterally
            self.initiate_lateral_recovery(zmp_y, center_y)
        elif abs(zmp_x - (self.left_foot[0] + self.right_foot[0]) / 2) > 0.2:
            # Robot is in danger of falling forward/backward
            self.initiate_angular_recovery(zmp_x)
        else:
            # Stable, reduce recovery if active
            if self.recovery_active:
                self.recovery_active = False
                self.get_logger().info('Stability restored')
    
    def initiate_lateral_recovery(self, zmp_y, support_center_y):
        """Initiate lateral balance recovery"""
        self.recovery_active = True
        
        # Generate recovery command to shift CoM back to safe area
        recovery_cmd = Twist()
        recovery_cmd.linear.y = -self.recovery_gain * (zmp_y - support_center_y)
        recovery_cmd.angular.z = -0.3 * (zmp_y - support_center_y)  # Counter-rotate
        
        self.recovery_cmd_pub.publish(recovery_cmd)
        
        self.get_logger().warn(
            f'Lateral recovery activated: ZMP Y={zmp_y:.3f}, Center Y={support_center_y:.3f}, '
            f'Correction Y cmd={recovery_cmd.linear.y:.3f}'
        )
    
    def initiate_angular_recovery(self, zmp_x):
        """Initiate forward/backward balance recovery"""
        self.recovery_active = True
        
        # Generate recovery command to pitch the robot
        recovery_cmd = Twist()
        recovery_cmd.linear.x = 0.0  # Forward/backward motion
        recovery_cmd.angular.y = -0.2 * (zmp_x - (self.left_foot[0] + self.right_foot[0]) / 2)
        
        self.recovery_cmd_pub.publish(recovery_cmd)
        
        self.get_logger().warn(
            f'Angular recovery activated: ZMP X={zmp_x:.3f}, '
            f'Correction pitch cmd={recovery_cmd.angular.y:.3f}'
        )

def main(args=None):
    rclpy.init(args=args)
    
    # Create both controllers
    walking_controller = CompleteBipedalController()
    balance_controller = BalanceRecoveryController()
    
    try:
        # Create executor and add both nodes
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(walking_controller)
        executor.add_node(balance_controller)
        
        # Spin both nodes
        executor.spin()
        
    except KeyboardInterrupt:
        pass
    finally:
        walking_controller.destroy_node()
        balance_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Mini-project

Create a complete bipedal locomotion system that:

1. Implements both LIPM and MPC balance controllers
2. Simulates a 3D bipedal robot in Isaac Sim with realistic physics
3. Generates stable walking gaits with proper footstep planning
4. Implements balance recovery strategies for perturbations
5. Evaluates stability margins and performance metrics
6. Includes visualization of ZMP, CoM, and footstep plans
7. Tests the system with various perturbations and uneven terrain

Your project should include:
- Complete robot model with proper dynamics
- LIPM and MPC controllers implementation
- Footstep planning algorithm
- Balance recovery system
- Performance evaluation metrics
- Isaac Sim integration for realistic physics simulation

## Summary

This chapter covered bipedal locomotion and balance control:

- **LIPM Control**: Linear Inverted Pendulum Model for balance control
- **MPC Walking**: Model Predictive Control for dynamic walking patterns
- **Gait Generation**: Creating stable walking patterns with proper foot placement
- **Balance Recovery**: Strategies to maintain stability during perturbations
- **Simulation**: Using Isaac Sim for realistic bipedal robot simulation
- **Control Integration**: Combining multiple control strategies for robust locomotion

Bipedal locomotion requires sophisticated control algorithms that account for the underactuated nature of bipedal systems while maintaining stability through careful foot placement and CoM control.