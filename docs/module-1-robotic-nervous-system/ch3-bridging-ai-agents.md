-----
title: Ch3  Bridging Python AI Agents with rclpy
module: 1
chapter: 3
sidebar_label: Ch3: Bridging Python AI Agents with rclpy
description: Integrating AI algorithms into ROS 2 using Python
tags: [ros2, python, rclpy, ai, machine learning, integration]
difficulty: intermediate
estimated_duration: 75
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Bridging Python AI Agents with rclpy

## Learning Outcomes
 Integrate Pythonbased AI algorithms into ROS 2 systems
 Understand the rclpy library and its API for ROS 2 communication
 Implement data exchange between AI algorithms and ROS 2 nodes
 Design nodes that process sensor data with AI algorithms
 Create nodes that execute AIdriven actions
 Implement proper error handling in AIROS integration

## Theory

### Python in ROS 2 Ecosystem

Python is one of the primary languages supported in ROS 2, making it an ideal choice for integrating AI algorithms into robotic systems. Python's rich ecosystem of libraries for machine learning, computer vision, and data processing makes it particularly valuable for robotics applications.

The `rclpy` library provides the Python client library for ROS 2. It implements the standard ROS 2 concepts (nodes, publishers, subscribers, services, actions) within Python.

### rclpy Architecture

<MermaidDiagram chart={`
graph TD;
    A[rclpy] > B[rclpy.core];
    A > C[rclpy.node];
    A > D[rclpy.publisher];
    A > E[rclpy.subscriber];
    A > F[rclpy.service];
    A > G[rclpy.client];
    A > H[rclpy.action];
    
    B > I[rclpy.typesupport];
    B > J[rclpy.utilities];
    
    C > K[Node];
    D > L[Publisher];
    E > M[Subscriber];
    
    style A fill:#2196F3,stroke:#0D47A1,color:#fff;
    style K fill:#4CAF50,stroke:#388E3C,color:#fff;
`} />

The `rclpy` library provides a Python interface to the ROS 2 client library (`rcl`), which handles communication with the DDS middleware. This allows Python code to participate in ROS 2 communication patterns seamlessly.

### AIROS Integration Patterns

Common patterns for integrating AI algorithms with ROS 2 include:

1. **Sensor Processing Nodes**: Nodes that receive sensor data, process it with AI algorithms, and publish results
2. **DecisionMaking Nodes**: Nodes that subscribe to multiple data streams, make AIdriven decisions, and publish commands
3. **Action Execution Nodes**: Nodes that execute complex AIdriven tasks with feedback mechanisms

## StepbyStep Labs

### Lab 1: Creating Your First AIROS Node

1. **Create a new package** for AI integration examples:
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create buildtype ament_python py_ai_integration
   cd py_ai_integration
   ```

2. **Create the main Python file** (`py_ai_integration/sensor_ai_node.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   import numpy as np
   from std_msgs.msg import Float32MultiArray
   from sensor_msgs.msg import LaserScan
   
   class SensorAINode(Node):
       def __init__(self):
           super().__init__('sensor_ai_node')
           
           # Create subscriber for laser scan data
           self.subscription = self.create_subscription(
               LaserScan,
               'scan',
               self.laser_callback,
               10)
           
           # Create publisher for processed data
           self.publisher = self.create_publisher(
               Float32MultiArray,
               'processed_scan',
               10)
           
           self.get_logger().info('AI Node initialized')
       
       def laser_callback(self, msg):
           # Process the laser scan data using AI algorithms
           ranges = np.array(msg.ranges)
           
           # Remove invalid measurements (inf or nan)
           ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
           ranges = np.nan_to_num(ranges, nan=msg.range_max)
           
           # Detect obstacles using simple thresholding
           obstacle_indices = np.where(ranges < 1.0)[0]  # Objects within 1m
           
           # Calculate some statistics about obstacles
           if len(obstacle_indices) > 0:
               obstacle_angles = [msg.angle_min + i * msg.angle_increment 
                                 for i in obstacle_indices]
               avg_distance = np.mean(ranges[obstacle_indices])
           else:
               obstacle_angles = []
               avg_distance = msg.range_max
           
           # Prepare AI output
           ai_output = Float32MultiArray()
           ai_output.data = [len(obstacle_indices), avg_distance, 
                             len(obstacle_angles), msg.range_max]
           
           # Publish results
           self.publisher.publish(ai_output)
           self.get_logger().info(f'Detected {len(obstacle_indices)} obstacles, avg distance: {avg_distance:.2f}m')


   def main(args=None):
       rclpy.init(args=args)
       sensor_ai_node = SensorAINode()
       
       try:
           rclpy.spin(sensor_ai_node)
       except KeyboardInterrupt:
           pass
       finally:
           sensor_ai_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Update setup.py** to include executables:
   ```python
   from setuptools import setup
   
   package_name = 'py_ai_integration'
   
   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='AI integration examples for ROS 2',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'sensor_ai_node = py_ai_integration.sensor_ai_node:main',
           ],
       },
   )
   ```

### Lab 2: Implementing a Machine Learning Node

1. **Create a new node** for MLbased classification (`py_ai_integration/object_classifier.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   import numpy as np
   from std_msgs.msg import String
   from sensor_msgs.msg import PointCloud2
   from sensor_msgs_py import point_cloud2
   from sklearn.cluster import DBSCAN
   import pickle
   
   class ObjectClassifierNode(Node):
       def __init__(self):
           super().__init__('object_classifier_node')
           
           # Create subscriber for point cloud data
           self.subscription = self.create_subscription(
               PointCloud2,
               'pointcloud_input',
               self.pointcloud_callback,
               10)
           
           # Create publisher for classification results
           self.publisher = self.create_publisher(
               String,
               'object_classification',
               10)
           
           # Initialize ML model (in a real application, you'd load a pretrained model)
           self.get_logger().info('Object Classifier Node initialized')
       
       def pointcloud_callback(self, msg):
           # Convert PointCloud2 message to numpy array
           points = np.array(list(point_cloud2.read_points(msg, 
                                         field_names=("x", "y", "z"),
                                         skip_nans=True)))
           
           if len(points) == 0:
               self.get_logger().info('No points in cloud')
               return
           
           # Use DBSCAN clustering to group points
           clustering = DBSCAN(eps=0.3, min_samples=10)
           cluster_labels = clustering.fit_predict(points)
           
           # Count number of clusters (potential objects)
           n_clusters = len(set(cluster_labels))  (1 if 1 in cluster_labels else 0)
           
           # Calculate basic statistics about clusters
           if n_clusters > 0:
               cluster_sizes = []
               for i in range(n_clusters):
                   cluster_size = len(cluster_labels[cluster_labels == i])
                   cluster_sizes.append(cluster_size)
               
               avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
           else:
               avg_cluster_size = 0
           
           # Prepare classification result
           result_msg = String()
           result_msg.data = f'Objects detected: {n_clusters}, Avg cluster size: {avg_cluster_size:.2f}'
           
           # Publish results
           self.publisher.publish(result_msg)
           self.get_logger().info(f'Classification result: {result_msg.data}')


   def main(args=None):
       rclpy.init(args=args)
       classifier_node = ObjectClassifierNode()
       
       try:
           rclpy.spin(classifier_node)
       except KeyboardInterrupt:
           pass
       finally:
           classifier_node.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

### Lab 3: Creating an AI Decision Node

1. **Create a decisionmaking node** (`py_ai_integration/decision_maker.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   import numpy as np
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import LaserScan
   
   class DecisionMakerNode(Node):
       def __init__(self):
           super().__init__('decision_maker_node')
           
           # Subscribers
           self.laser_sub = self.create_subscription(
               LaserScan,
               'scan',
               self.laser_callback,
               10)
           
           # Publishers
           self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
           self.alert_pub = self.create_publisher(Bool, 'safety_alert', 10)
           
           # Internal state
           self.obstacle_detected = False
           self.safety_distance = 0.8  # meters
           self.forward_speed = 0.5    # m/s
           
           # Create a timer for decisionmaking loop
           self.timer = self.create_timer(0.1, self.decision_loop)  # 10 Hz
           
           self.get_logger().info('Decision Maker Node initialized')
       
       def laser_callback(self, msg):
           # Process laser scan to detect obstacles
           ranges = np.array(msg.ranges)
           ranges = np.where(np.isinf(ranges), msg.range_max, ranges)
           ranges = np.nan_to_num(ranges, nan=msg.range_max)
           
           # Find the closest obstacle
           min_distance = np.min(ranges)
           self.obstacle_detected = min_distance < self.safety_distance
       
       def decision_loop(self):
           # AIbased decision making
           cmd = Twist()
           
           if self.obstacle_detected:
               # Obstacle detected  stop and turn
               cmd.linear.x = 0.0
               cmd.angular.z = 0.5  # Turn right
               
               # Publish safety alert
               alert_msg = Bool()
               alert_msg.data = True
               self.alert_pub.publish(alert_msg)
               
               self.get_logger().info('Obstacle detected! Stopping and turning...')
           else:
               # No obstacle  move forward
               cmd.linear.x = self.forward_speed
               cmd.angular.z = 0.0
               
               # Publish safety clear
               alert_msg = Bool()
               alert_msg.data = False
               self.alert_pub.publish(alert_msg)
               
               self.get_logger().info('Moving forward...')
           
           # Publish velocity command
           self.cmd_pub.publish(cmd)


   def main(args=None):
       rclpy.init(args=args)
       decision_maker = DecisionMakerNode()
       
       try:
           rclpy.spin(decision_maker)
       except KeyboardInterrupt:
           pass
       finally:
           decision_maker.destroy_node()
           rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's a complete example combining AI and ROS 2 for a simple object recognition task:

```python
# object_recognition_node.py
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class ObjectRecognitionNode(Node):
    def __init__(self):
        super().__init__('object_recognition_node')
        
        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            'camera_image',
            self.image_callback,
            10)
        
        # Create publisher for recognition results
        self.result_pub = self.create_publisher(
            String,
            'object_recognition_result',
            10)
        
        # Initialize OpenCV bridge
        self.bridge = CvBridge()
        
        # Simple color detection parameters
        self.lower_red = np.array([0, 50, 50])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.get_logger().info('Object Recognition Node initialized')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {str(e)}')
            return
        
        # Convert BGR to HSV for color detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 + mask2
        
        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (9, 9), 2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count red objects (areas larger than threshold)
        red_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small areas
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                red_objects.append({
                    'area': area,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
        
        # Publish results
        result_msg = String()
        if len(red_objects) > 0:
            result_msg.data = f"Detected {len(red_objects)} red objects. Largest area: {max([obj['area'] for obj in red_objects]):.2f}"
        else:
            result_msg.data = "No red objects detected"
        
        self.result_pub.publish(result_msg)
        self.get_logger().info(f'AI result: {result_msg.data}')


def main(args=None):
    rclpy.init(args=args)
    object_recognition_node = ObjectRecognitionNode()
    
    try:
        rclpy.spin(object_recognition_node)
    except KeyboardInterrupt:
        pass
    finally:
        object_recognition_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Additional dependencies for the AI node (setup.cfg):
```
[develop]
scriptdir=$base/lib/py_ai_integration
[install_scripts]
install_dir=$base/lib/py_ai_integration
```

## Miniproject

Create a complete AIROS integration project that:

1. Creates a simulated sensor node that publishes mock sensor data (e.g., temperature, humidity, light levels)
2. Implements an AI node that uses a machine learning algorithm to predict environmental conditions based on sensor data
3. Creates a decisionmaking node that takes the AI predictions and sends appropriate commands to simulated actuators
4. Includes proper error handling and logging
5. Uses rclpy best practices like proper shutdown and resource cleanup

Test your system by running all nodes together and verifying the data flows correctly from sensor to AI to decision maker.

## Summary

This chapter introduced the integration of Pythonbased AI algorithms with ROS 2 systems using the rclpy library. Key concepts include:

 The rclpy library serves as the Python client library for ROS 2
 Common patterns for AIROS integration include sensor processing, decisionmaking, and action execution
 Proper error handling and resource management are essential for robust AIROS systems
 The Python ecosystem provides rich support for AI algorithms that can be easily integrated into ROS 2

Successfully bridging AI algorithms with ROS 2 enables the development of intelligent robotic systems that can perceive, reason, and act in complex environments.