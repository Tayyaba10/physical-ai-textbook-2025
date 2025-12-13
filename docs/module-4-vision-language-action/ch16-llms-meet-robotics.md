-----
title: Ch16  LLMs Meet Robotics
module: 4
chapter: 16
sidebar_label: Ch16: LLMs Meet Robotics
description: Integrating Large Language Models with robotics for natural language interaction and command execution
tags: [llm, robotics, naturallanguage, gpt, visionlanguageaction, embodiedai, transformers]
difficulty: intermediate
estimated_duration: 90
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# LLMs Meet Robotics

## Learning Outcomes
 Understand the integration of Large Language Models with robotic systems
 Implement natural language processing for robot command interpretation
 Design visionlanguageaction pipelines for embodied AI
 Create multimodal interfaces combining language, vision, and action
 Evaluate the performance of LLMpowered robotic systems
 Develop techniques for grounding language in physical environments
 Implement safety and error handling for LLMcontrolled robots

## Theory

### LLMs in Robotics Context

Large Language Models (LLMs) bring natural language understanding to robotics systems. When combined with robotics, LLMs can:

<MermaidDiagram chart={`
graph TD;
    A[Large Language Model] > B[Language Understanding];
    A > C[Task Planning];
    A > D[Command Interpretation];
    
    E[Robot System] > F[Sensors];
    E > G[Actuators];
    E > H[Navigation];
    
    I[VisionLanguageAction] > J[Perception];
    I > K[Action Selection];
    I > L[Planning];
    
    B > I;
    C > I;
    D > I;
    F > J;
    G > K;
    H > L;
    
    M[Human] > N[Natural Language Command];
    N > A;
    I > O[Robotic Action];
    O > E;
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style E fill:#2196F3,stroke:#0D47A1,color:#fff;
    style I fill:#FF9800,stroke:#E65100,color:#fff;
`} />

### Key Components of LLMRobotics Systems

**Natural Language Understanding**: Processing human commands into actionable robot instructions.

**Task Planning**: Breaking down highlevel commands into sequences of primitive actions.

**Perception Integration**: Combining vision data with language understanding.

**Action Grounding**: Connecting abstract language concepts to specific robot actions.

### VisionLanguageAction (VLA) Models

VLA models represent a new paradigm where language understanding, visual perception, and action selection are combined in a single model:

 **Embodied Learning**: Learning from robot interactions with the environment
 **Multimodal Fusion**: Combining language, vision, and proprioceptive data
 **Policy Learning**: Learning to map multimodal inputs to robot actions

### Grounding Language in Physical Reality

One of the biggest challenges in LLMrobotics integration is **grounding**  connecting abstract language concepts to physical entities and actions in the robot's environment.

## StepbyStep Labs

### Lab 1: Setting up LLM Integration with Robotics

1. **Install required dependencies**:
   ```bash
   pip install openai anthropic transformers torch torchvision torchaudio
   pip install gymnasium ros2 rospy geometry_msgs sensor_msgs
   pip install langchain langchainopenai llamaindex
   ```

2. **Create a basic LLMrobot interface class** (`llm_robot_interface.py`):
   ```python
   #!/usr/bin/env python3

   import asyncio
   import openai
   import json
   import rospy
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import LaserScan, Image
   from cv_bridge import CvBridge
   import numpy as np
   import cv2
   from typing import Dict, List, Optional
   from dataclasses import dataclass

   @dataclass
   class RobotCommand:
       action: str
       parameters: Dict[str, float]
       confidence: float

   @dataclass
   class PerceptualData:
       laser_scan: Optional[List[float]] = None
       camera_image: Optional[np.ndarray] = None
       robot_pose: Optional[Dict[str, float]] = None

   class LLMRobotInterface:
       def __init__(self, api_key: str, model_name: str = "gpt3.5turbo"):
           # Initialize LLM
           openai.api_key = api_key
           self.model_name = model_name
           
           # Initialize ROS
           rospy.init_node('llm_robot_interface', anonymous=True)
           
           # Initialize CV bridge
           self.cv_bridge = CvBridge()
           
           # Robot state
           self.perceptual_data = PerceptualData()
           self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
           
           # Publishers and Subscribers
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           
           # Subscribe to robot sensors
           rospy.Subscriber('/scan', LaserScan, self.laser_callback)
           rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
           
           # Robot control parameters
           self.linear_speed = 0.5
           self.angular_speed = 0.5
           
           print("LLM Robot Interface initialized")
       
       def laser_callback(self, msg):
           """Update laser scan data"""
           self.perceptual_data.laser_scan = list(msg.ranges)
           # Keep only finite values within range
           self.perceptual_data.laser_scan = [
               r for r in self.perceptual_data.laser_scan 
               if not (np.isinf(r) or np.isnan(r)) and 0.1 < r < 10.0
           ]
       
       def image_callback(self, msg):
           """Update camera image data"""
           try:
               cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
               self.perceptual_data.camera_image = cv_image
           except Exception as e:
               print(f"Error converting image: {e}")
       
       def update_robot_pose(self, x, y, theta):
           """Update robot pose estimate"""
           self.perceptual_data.robot_pose = {'x': x, 'y': y, 'theta': theta}
       
       def interpret_command(self, user_command: str) > RobotCommand:
           """Use LLM to interpret natural language command"""
           # Format the prompt with current robot state
           prompt = f"""
           You are a robot command interpreter. Given the user's natural language command,
           convert it to a specific robot action with parameters.
           
           Current robot state:
            Pose: x={self.perceptual_data.robot_pose['x']:.2f}, y={self.perceptual_data.robot_pose['y']:.2f}, theta={self.perceptual_data.robot_pose['theta']:.2f}
            Laser scan: {f"Min distance {min(self.perceptual_data.laser_scan):.2f}m" if self.perceptual_data.laser_scan and len(self.perceptual_data.laser_scan) > 0 else 'No scan data'}
           
           User command: "{user_command}"
           
           Respond with ONLY a JSON object in this exact format:
           {{
             "action": "forward|backward|left|right|stop|approach_object|avoid_object|turn_to_face",
             "parameters": {{
               "linear_speed": float,  # 1.0 to 1.0
               "angular_speed": float, # 1.0 to 1.0
               "distance": float,      # meters, for movements
               "angle": float          # radians, for turns
             }},
             "confidence": float       # 0.0 to 1.0
           }}
           
           Be specific and only respond with the JSON object.
           """
           
           try:
               response = openai.ChatCompletion.create(
                   model=self.model_name,
                   messages=[{"role": "user", "content": prompt}],
                   temperature=0.1,  # Low temperature for consistency
                   max_tokens=200
               )
               
               # Extract response
               response_text = response.choices[0].message.content.strip()
               
               # Clean up response to extract only JSON
               if response_text.startswith('```'):
                   response_text = response_text.split('\n', 1)[1]
                   response_text = response_text.rstrip('`')
               
               # Parse JSON
               json_data = json.loads(response_text)
               
               # Create RobotCommand object
               command = RobotCommand(
                   action=json_data['action'],
                   parameters=json_data['parameters'],
                   confidence=json_data['confidence']
               )
               
               return command
               
           except Exception as e:
               print(f"Error interpreting command: {e}")
               # Return default command
               return RobotCommand(
                   action="stop",
                   parameters={"linear_speed": 0.0, "angular_speed": 0.0},
                   confidence=0.5
               )
       
       def execute_command(self, command: RobotCommand) > bool:
           """Execute the interpreted command on the robot"""
           msg = Twist()
           
           if command.confidence < 0.3:
               rospy.logwarn(f"Command confidence is low ({command.confidence}), refusing to execute")
               return False
           
           if command.action == "forward":
               msg.linear.x = self.linear_speed * command.parameters.get('linear_speed', 1.0)
               msg.angular.z = 0.0
           elif command.action == "backward":
               msg.linear.x = self.linear_speed * command.parameters.get('linear_speed', 1.0)
               msg.angular.z = 0.0
           elif command.action == "left":
               msg.linear.x = 0.0
               msg.angular.z = self.angular_speed * command.parameters.get('angular_speed', 1.0)
           elif command.action == "right":
               msg.linear.x = 0.0
               msg.angular.z = self.angular_speed * command.parameters.get('angular_speed', 1.0)
           elif command.action == "stop":
               msg.linear.x = 0.0
               msg.angular.z = 0.0
           elif command.action == "turn_to_face":
               # Calculate turn to face a direction
               angle = command.parameters.get('angle', 0.0)
               msg.angular.z = self.angular_speed * np.sign(angle)
           elif command.action == "approach_object":
               # Move forward with obstacle avoidance
               min_distance = min(self.perceptual_data.laser_scan) if self.perceptual_data.laser_scan else float('inf')
               if min_distance > 0.5:  # Safe distance
                   msg.linear.x = self.linear_speed * 0.5
               else:
                   msg.linear.x = 0.0
           else:
               rospy.logwarn(f"Unknown action: {command.action}")
               return False
           
           self.cmd_vel_pub.publish(msg)
           rospy.loginfo(f"Executed command: {command.action} with params {command.parameters}")
           return True
       
       def process_user_command(self, user_command: str) > bool:
           """Process a complete user command: interpret and execute"""
           rospy.loginfo(f"Processing user command: '{user_command}'")
           
           # Interpret command with LLM
           command = self.interpret_command(user_command)
           
           rospy.loginfo(f"Interpreted command: {command.action}, confidence: {command.confidence:.2f}")
           
           # Execute command
           success = self.execute_command(command)
           
           return success
       
       def run(self):
           """Run the main control loop"""
           rate = rospy.Rate(10)  # 10 Hz
           
           while not rospy.is_shutdown():
               # Process any pending tasks
               # In a real implementation, this might receive commands from a queue or other source
               rate.sleep()
   ```

### Lab 2: Creating a VisionLanguage Interface

1. **Create visionlanguage processing module** (`vision_language_processor.py`):
   ```python
   #!/usr/bin/env python3

   import cv2
   import numpy as np
   import torch
   import openai
   from PIL import Image
   import base64
   import io
   from typing import List, Dict, Tuple
   import json

   class VisionLanguageProcessor:
       def __init__(self, openai_api_key: str):
           openai.api_key = openai_api_key
           self.openai_client = openai.OpenAI(api_key=openai_api_key)
       
       def encode_image(self, image: np.ndarray) > str:
           """Encode a numpy image to base64 string"""
           # Convert BGR to RGB if needed
           if len(image.shape) == 3 and image.shape[2] == 3:
               image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           else:
               image_rgb = image
           
           # Convert to PIL
           pil_image = Image.fromarray(image_rgb)
           
           # Save to bytes
           buffer = io.BytesIO()
           pil_image.save(buffer, format="JPEG")
           img_bytes = buffer.getvalue()
           
           # Encode to base64
           base64_str = base64.b64encode(img_bytes).decode('utf8')
           return base64_str
       
       def describe_scene(self, image: np.ndarray) > str:
           """Generate a textual description of the scene"""
           base64_image = self.encode_image(image)
           
           prompt = "Describe this robot's view of the environment in detail. Focus on objects, their positions relative to the robot, potential obstacles, navigable paths, and anything relevant for robot navigation and manipulation."
           
           try:
               response = self.openai_client.chat.completions.create(
                   model="gpt4visionpreview",
                   messages=[
                       {
                           "role": "user",
                           "content": [
                               {"type": "text", "text": prompt},
                               {
                                   "type": "image_url",
                                   "image_url": {
                                       "url": f"data:image/jpeg;base64,{base64_image}"
                                   }
                               }
                           ]
                       }
                   ],
                   max_tokens=300
               )
               
               return response.choices[0].message.content
               
           except Exception as e:
               print(f"Error describing scene: {e}")
               return "Unable to describe the scene"
       
       def identify_objects(self, image: np.ndarray) > List[Dict]:
           """Identify objects in the image and their locations"""
           base64_image = self.encode_image(image)
           
           prompt = """
           Identify and locate the following types of objects in the image:
            Doors, windows
            Tables, chairs, desks
            People
            Obstacles that might block robot movement
            Any potentially interesting objects for robot interaction
           
           For each object, provide:
            type: object category
            position_in_image: approximate position (left, center, right, top, bottom)
            estimated_distance: distance estimate if possible
            relevance_for_navigation: how relevant this is for robot navigation
           
           Respond with only a JSON array of objects.
           """
           
           try:
               response = self.openai_client.chat.completions.create(
                   model="gpt4visionpreview",
                   messages=[
                       {
                           "role": "user",
                           "content": [
                               {"type": "text", "text": prompt},
                               {
                                   "type": "image_url",
                                   "image_url": {
                                       "url": f"data:image/jpeg;base64,{base64_image}"
                                   }
                               }
                           ]
                       }
                   ],
                   max_tokens=500
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON if wrapped in markdown
               if "```json" in response_text:
                   json_start = response_text.find("```json") + 7
                   json_end = response_text.find("```", json_start)
                   response_text = response_text[json_start:json_end].strip()
               elif "```" in response_text:
                   first_json = response_text.find("{")
                   last_json = response_text.rfind("}") + 1
                   response_text = response_text[first_json:last_json].strip()
               
               objects = json.loads(response_text)
               return objects
               
           except Exception as e:
               print(f"Error identifying objects: {e}")
               return []
       
       def answer_visual_question(self, image: np.ndarray, question: str) > str:
           """Answer a specific question about the visual scene"""
           base64_image = self.encode_image(image)
           
           prompt = f"Answer the following question about the image: {question}"
           
           try:
               response = self.openai_client.chat.completions.create(
                   model="gpt4visionpreview",
                   messages=[
                       {
                           "role": "user",
                           "content": [
                               {"type": "text", "text": prompt},
                               {
                                   "type": "image_url",
                                   "image_url": {
                                       "url": f"data:image/jpeg;base64,{base64_image}"
                                   }
                               }
                           ]
                       }
                   ],
                   max_tokens=200
               )
               
               return response.choices[0].message.content
               
           except Exception as e:
               print(f"Error answering question: {e}")
               return "Unable to answer the question"
       
       def detect_navigation_hazards(self, image: np.ndarray) > Dict:
           """Detect potential hazards for navigation"""
           base64_image = self.encode_image(image)
           
           prompt = """
           Analyze the image for potential navigation hazards for a robot. 
           Consider:
            Dropoffs or steep edges
            Narrow passages
            Moving obstacles
            Unstable surfaces
            Areas that might trap the robot
           
           Rate the overall navigation safety and list specific hazards.
           Respond with a JSON object containing:
            safety_rating: integer 15 (1=unsafe, 5=very safe)
            hazards: array of hazard descriptions
            safe_paths: description of any clearly safe paths
           """
           
           try:
               response = self.openai_client.chat.completions.create(
                   model="gpt4visionpreview",
                   messages=[
                       {
                           "role": "user",
                           "content": [
                               {"type": "text", "text": prompt},
                               {
                                   "type": "image_url",
                                   "image_url": {
                                       "url": f"data:image/jpeg;base64,{base64_image}"
                                   }
                               }
                           ]
                       }
                   ],
                   max_tokens=300
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON
               json_start = response_text.find("{")
               json_end = response_text.rfind("}") + 1
               response_text = response_text[json_start:json_end].strip()
               
               hazards = json.loads(response_text)
               return hazards
               
           except Exception as e:
               print(f"Error detecting hazards: {e}")
               return {
                   "safety_rating": 3,
                   "hazards": ["Unable to analyze image"],
                   "safe_paths": "Unknown"
               }
   ```

### Lab 3: Implementing LanguageGrounded Navigation

1. **Create a languagegrounded navigation system** (`language_navigable_planner.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   from geometry_msgs.msg import PoseStamped
   from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
   import actionlib
   import tf
   import numpy as np
   from typing import List, Dict, Optional
   from llm_robot_interface import LLMRobotInterface
   from vision_language_processor import VisionLanguageProcessor

   class LanguageGroundedNavigator:
       def __init__(self, llm_interface: LLMRobotInterface, vision_processor: VisionLanguageProcessor):
           self.llm_interface = llm_interface
           self.vision_processor = vision_processor
           
           # Initialize ROS components
           self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
           self.move_base.wait_for_server(rospy.Duration(60))  # Wait up to 60 seconds
           
           self.tf_listener = tf.TransformListener()
           
           # Known locations in the environment (would be learned over time)
           self.known_locations = {
               "kitchen": {"x": 5.0, "y": 3.0, "theta": 0.0},
               "living_room": {"x": 2.0, "y": 1.0, "theta": 0.0},
               "bedroom": {"x": 7.0, "y": 1.0, "theta": 0.0},
               "entrance": {"x": 0.0, "y": 0.0, "theta": 0.0}
           }
           
           print("LanguageGrounded Navigator initialized")
       
       def parse_navigation_command(self, command: str) > Dict:
           """Parse natural language navigation command using LLM"""
           current_pos = self.llm_interface.perceptual_data.robot_pose
           
           prompt = f"""
           You are a navigation command parser for a mobile robot. Interpret the user's command
           to determine where the robot should go.
           
           Current position: x={current_pos['x']:.1f}, y={current_pos['y']:.1f}
           Known locations: {list(self.known_locations.keys())}
           
           Navigation command: "{command}"
           
           Respond with a JSON object containing:
           {{
             "target_location": "name of target location if known, otherwise 'unknown'",
             "relative_direction": "forward, backward, left, right, or 'none' if specific location",
             "distance": float value if relative, else null,
             "description": "brief description of the destination",
             "landmarks": ["list", "of", "identifying", "landmarks"]
           }}
           
           Respond with ONLY the JSON object.
           """
           
           try:
               response = openai.ChatCompletion.create(
                   model="gpt3.5turbo",  # or your preferred model
                   messages=[{"role": "user", "content": prompt}],
                   temperature=0.1,
                   max_tokens=200
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON
               json_start = response_text.find("{")
               json_end = response_text.rfind("}") + 1
               response_text = response_text[json_start:json_end].strip()
               
               return json.loads(response_text)
               
           except Exception as e:
               print(f"Error parsing navigation command: {e}")
               return {
                   "target_location": "unknown",
                   "relative_direction": "none",
                   "distance": None,
                   "description": "Error parsing command",
                   "landmarks": []
               }
       
       def get_location_coordinates(self, location_name: str) > Optional[Dict]:
           """Get coordinates for a known location"""
           if location_name in self.known_locations:
               return self.known_locations[location_name]
           else:
               # In a complete system, this would use localization or mapping to find the location
               # For now, return None
               return None
       
       def navigate_to_waypoint(self, x: float, y: float, theta: float = 0.0) > bool:
           """Navigate to specific coordinates"""
           goal = MoveBaseGoal()
           goal.target_pose.header.frame_id = "map"
           goal.target_pose.header.stamp = rospy.Time.now()
           
           goal.target_pose.pose.position.x = x
           goal.target_pose.pose.position.y = y
           goal.target_pose.pose.position.z = 0.0
           
           # Convert theta to quaternion
           from tf.transformations import quaternion_from_euler
           q = quaternion_from_euler(0, 0, theta)
           goal.target_pose.pose.orientation.x = q[0]
           goal.target_pose.pose.orientation.y = q[1]
           goal.target_pose.pose.orientation.z = q[2]
           goal.target_pose.pose.orientation.w = q[3]
           
           rospy.loginfo(f"Navigating to x={x}, y={y}, theta={theta}")
           
           # Send goal
           self.move_base.send_goal(goal)
           
           # Wait for result
           finished_within_time = self.move_base.wait_for_result(rospy.Duration(120))  # 2 minutes timeout
           
           if not finished_within_time:
               self.move_base.cancel_goal()
               rospy.logerr("Timed out achieving goal")
               return False
           
           state = self.move_base.get_state()
           result = self.move_base.get_result()
           
           if state == 3:  # SUCCEEDED
               rospy.loginfo("Goal reached successfully")
               return True
           else:
               rospy.logerr(f"Navigation failed with state: {state}")
               return False
       
       def execute_navigation_command(self, command: str) > bool:
           """Execute a natural language navigation command"""
           rospy.loginfo(f"Parsing navigation command: '{command}'")
           
           # Parse the command
           parsed_command = self.parse_navigation_command(command)
           rospy.loginfo(f"Parsed command: {parsed_command}")
           
           # Determine target
           target_coords = None
           
           if parsed_command["target_location"] != "unknown":
               # Try to get coordinates for known location
               target_coords = self.get_location_coordinates(parsed_command["target_location"])
               
               if target_coords is None:
                   rospy.logwarn(f"Unknown location: {parsed_command['target_location']}")
                   # Try to learn the location based on landmarks
                   target_coords = self.learn_new_location(parsed_command["landmarks"])
           
           elif parsed_command["relative_direction"] != "none":
               # Calculate relative movement
               current_pos = self.llm_interface.perceptual_data.robot_pose
               distance = parsed_command.get("distance", 1.0)
               
               if parsed_command["relative_direction"] == "forward":
                   new_x = current_pos['x'] + distance * np.cos(current_pos['theta'])
                   new_y = current_pos['y'] + distance * np.sin(current_pos['theta'])
                   target_coords = {"x": new_x, "y": new_y, "theta": current_pos['theta']}
               elif parsed_command["relative_direction"] == "backward":
                   new_x = current_pos['x']  distance * np.cos(current_pos['theta'])
                   new_y = current_pos['y']  distance * np.sin(current_pos['theta'])
                   target_coords = {"x": new_x, "y": new_y, "theta": current_pos['theta']}
               elif parsed_command["relative_direction"] == "left":
                   new_theta = current_pos['theta'] + np.pi/2
                   new_x = current_pos['x'] + distance * np.cos(new_theta)
                   new_y = current_pos['y'] + distance * np.sin(new_theta)
                   target_coords = {"x": new_x, "y": new_y, "theta": new_theta}
               elif parsed_command["relative_direction"] == "right":
                   new_theta = current_pos['theta']  np.pi/2
                   new_x = current_pos['x'] + distance * np.cos(new_theta)
                   new_y = current_pos['y'] + distance * np.sin(new_theta)
                   target_coords = {"x": new_x, "y": new_y, "theta": new_theta}
           
           if target_coords:
               rospy.loginfo(f"Navigating to: {target_coords}")
               return self.navigate_to_waypoint(
                   target_coords["x"], 
                   target_coords["y"], 
                   target_coords["theta"]
               )
           else:
               rospy.logerr("Could not determine navigation target")
               return False
       
       def learn_new_location(self, landmarks: List[str]) > Optional[Dict]:
           """Attempt to learn a new location based on landmarks"""
           # In a real system, this would use visual SLAM to identify the location
           # For now, just return current position as a placeholder
           current_pos = self.llm_interface.perceptual_data.robot_pose
           rospy.loginfo(f"Learning new location near: {landmarks}")
           return current_pos

   # Example usage function
   def example_navigation():
       """Example of using the languagegrounded navigator"""
       # Initialize components (API keys would need to be provided)
       # llm_interface = LLMRobotInterface(api_key="youropenaiapikey")
       # vision_processor = VisionLanguageProcessor(openai_api_key="youropenaiapikey")
       # navigator = LanguageGroundedNavigator(llm_interface, vision_processor)
       
       # Example commands that could be processed
       commands = [
           "Go to the kitchen",
           "Move forward 2 meters",
           "Navigate to the living room",
           "Go near the red chair"
       ]
       
       for cmd in commands:
           print(f"Command: {cmd}")
           # result = navigator.execute_navigation_command(cmd)
           print("Navigation command processed (in simulation)\n")
   ```

## Runnable Code Example

Here's a complete example that brings together all the components:

```python
#!/usr/bin/env python3
# complete_llm_robot_integration.py

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import cv2
import numpy as np
import openai
import json
from cv_bridge import CvBridge
from typing import Dict, List

class CompleteLLMRobotSystem:
    """Complete system integrating LLMs with robotics for natural interaction"""
    
    def __init__(self, openai_api_key: str):
        # Initialize OpenAI
        openai.api_key = openai_api_key
        
        # Initialize ROS
        rospy.init_node('complete_llm_robot_system', anonymous=True)
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Robot state
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.laser_scan = []
        self.camera_image = None
        
        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Robot parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.3
        
        print("Complete LLM Robot System initialized")
    
    def laser_callback(self, msg):
        """Update laser scan data"""
        self.laser_scan = [r for r in msg.ranges if not (np.isinf(r) or np.isnan(r))]
    
    def image_callback(self, msg):
        """Update camera image data"""
        try:
            self.camera_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(f"Error converting image: {e}")
    
    def analyze_visual_scene(self) > str:
        """Analyze the current visual scene"""
        if self.camera_image is None:
            return "No camera image available"
        
        # Convert image for analysis (simplified  in real implementation would use VLM)
        height, width = self.camera_image.shape[:2]
        
        # Simple analysis based on image data
        analysis = []
        
        # Analyze color distribution (simplified)
        color_distances = np.mean(self.camera_image, axis=(0, 1))
        dominant_color = "unknown"
        if color_distances[2] > color_distances[1] and color_distances[2] > color_distances[0]:
            dominant_color = "red"
        elif color_distances[1] > color_distances[0] and color_distances[1] > color_distances[2]:
            dominant_color = "green"
        elif color_distances[0] > color_distances[1] and color_distances[0] > color_distances[2]:
            dominant_color = "blue"
        
        # Analyze spatial distribution
        center_region = self.camera_image[height//3:2*height//3, width//3:2*width//3]
        center_brightness = np.mean(center_region)
        
        # Analyze obstacles from laser
        if self.laser_scan:
            min_distance = min(self.laser_scan) if self.laser_scan else float('inf')
        else:
            min_distance = float('inf')
        
        analysis.append(f"Observed scene with dominant color {dominant_color}")
        analysis.append(f"Center brightness: {center_brightness:.2f}")
        if min_distance < float('inf'):
            analysis.append(f"Closest obstacle: {min_distance:.2f}m ahead")
        
        return " ".join(analysis)
    
    def interpret_command_with_llm(self, user_command: str) > Dict:
        """Use LLM to interpret command in context of current state"""
        current_state = {
            "pose": self.robot_pose,
            "visual_analysis": self.analyze_visual_scene(),
            "obstacle_distance": min(self.laser_scan) if self.laser_scan else "unknown",
        }
        
        prompt = f"""
        You are a robot command interpreter. Given the user's request and the current robot state,
        determine the most appropriate robot action.
        
        Current state:
         Position: x={current_state['pose']['x']:.2f}, y={current_state['pose']['y']:.2f}
         Heading: {current_state['pose']['theta']:.2f} radians
         Visual: {current_state['visual_analysis']}
         Nearest obstacle: {current_state['obstacle_distance']:.2f}m
        
        User command: "{user_command}"
        
        Respond with ONLY a JSON object with this exact structure:
        {{
          "action": "move_forward|move_backward|turn_left|turn_right|stop|approach_object|avoid_obstacle|describe_scene",
          "parameters": {{
            "linear_speed": number,  # 1.0 to 1.0
            "angular_speed": number, # 1.0 to 1.0
            "duration": number       # seconds
          }},
          "explanation": "Brief explanation of why this action was chosen",
          "confidence": number       # 0.0 to 1.0
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt3.5turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean up markdown if present
            if response_text.startswith('```'):
                response_text = response_text.split('\n', 1)[1]
                response_text = response_text.rstrip('`')
            
            # Parse JSON
            parsed_response = json.loads(response_text)
            return parsed_response
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {
                "action": "stop",
                "parameters": {"linear_speed": 0.0, "angular_speed": 0.0, "duration": 1.0},
                "explanation": "Error in processing command",
                "confidence": 0.1
            }
    
    def execute_action(self, action_plan: Dict) > bool:
        """Execute the planned action"""
        cmd = Twist()
        
        if action_plan["confidence"] < 0.3:
            rospy.logwarn(f"Action confidence too low ({action_plan['confidence']}), not executing")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            return False
        
        action = action_plan["action"]
        params = action_plan["parameters"]
        
        rospy.loginfo(f"Executing action: {action} with params: {params}")
        
        if action == "move_forward":
            cmd.linear.x = self.linear_speed * params.get("linear_speed", 1.0)
            cmd.angular.z = 0.0
        elif action == "move_backward":
            cmd.linear.x = self.linear_speed * params.get("linear_speed", 1.0)
            cmd.angular.z = 0.0
        elif action == "turn_left":
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed * abs(params.get("angular_speed", 1.0))
        elif action == "turn_right":
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed * abs(params.get("angular_speed", 1.0))
        elif action == "stop":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif action == "approach_object":
            # Move forward cautiously
            if self.laser_scan:
                min_dist = min(self.laser_scan) if self.laser_scan else float('inf')
                if min_dist > 0.5:  # Safe distance
                    cmd.linear.x = self.linear_speed * 0.5
                else:
                    cmd.linear.x = 0.0
                    rospy.logwarn("Too close to obstacle, stopping approach")
            else:
                cmd.linear.x = 0.0
        elif action == "avoid_obstacle":
            # Turn away from nearest obstacle
            if self.laser_scan:
                # Find direction of nearest obstacle (simplified)
                front_range = self.laser_scan[len(self.laser_scan)//2  30 : len(self.laser_scan)//2 + 30]
                if front_range:
                    min_idx = front_range.index(min(front_range))
                    if min_idx < len(front_range) // 2:
                        cmd.angular.z = self.angular_speed  # Turn right
                    else:
                        cmd.angular.z = self.angular_speed  # Turn left
            cmd.linear.x = 0.0  # Don't move forward while avoiding
        elif action == "describe_scene":
            # Just analyze and log, don't move
            rospy.loginfo(f"Scene description: {self.analyze_visual_scene()}")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            rospy.logwarn(f"Unknown action: {action}")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return False
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)
        
        # For timed actions, publish for duration then stop
        if "duration" in params and params["duration"] > 0:
            rospy.sleep(params["duration"])
            # Send stop command after duration
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_cmd)
        
        return True
    
    def process_command(self, user_command: str) > bool:
        """Process a user command endtoend: interpret and execute"""
        rospy.loginfo(f"Processing command: '{user_command}'")
        
        # Interpret command with LLM
        action_plan = self.interpret_command_with_llm(user_command)
        rospy.loginfo(f"LLM action plan: {action_plan}")
        
        # Execute action
        success = self.execute_action(action_plan)
        
        if success:
            rospy.loginfo(f"Command executed successfully: {action_plan['explanation']}")
        else:
            rospy.logerr("Command execution failed")
        
        return success
    
    def run_demo(self):
        """Run a demonstration of the system"""
        rospy.loginfo("Starting LLMRobot demo")
        
        demo_commands = [
            "Tell me what you see",
            "Move forward slowly",
            "Turn left",
            "Stop",
            "Move toward the clear path",
            "Go forward 1 meter",
            "Turn right and move forward"
        ]
        
        for i, command in enumerate(demo_commands):
            rospy.loginfo(f"\nDemo Step {i+1}: {command}")
            
            if not rospy.is_shutdown():
                success = self.process_command(command)
                
                if not success:
                    rospy.logerr(f"Failed to execute command: {command}")
                    break
                
                # Wait between commands
                rospy.sleep(2.0)
            else:
                break
        
        # Final stop
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_cmd)
        
        rospy.loginfo("Demo completed")


def main():
    """Main function to run the complete system"""
    # Initialize node
    rospy.init_node('llm_robot_main', anonymous=True)
    
    # Get API key from parameter server or environment variable
    api_key = rospy.get_param('~openai_api_key', '')
    if not api_key:
        api_key = input("Enter OpenAI API key: ")
    
    if not api_key:
        rospy.logerr("No OpenAI API key provided!")
        return
    
    # Create system
    system = CompleteLLMRobotSystem(api_key)
    
    rospy.loginfo("LLMRobot system ready for commands")
    
    try:
        # Run demo or wait for commands
        system.run_demo()
        
        # Or alternatively, wait for interactive commands
        # rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
        
        # Ensure robot stops
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        system.cmd_vel_pub.publish(stop_cmd)


if __name__ == '__main__':
    main()
```

## Miniproject

Create a complete languageenabled robot system that:

1. Integrates a Large Language Model with a mobile robot
2. Implements visionlanguage processing for scene understanding
3. Enables navigation based on natural language commands
4. Demonstrates object recognition and manipulation through language
5. Handles ambiguous commands through clarification dialogues
6. Implements safety checks and validation for LLMgenerated commands
7. Evaluates the system's performance with various natural language inputs
8. Documents the system's capabilities and limitations

Your project should include:
 Complete LLM integration with robot control
 Visionlanguage processing pipeline
 Natural language command understanding
 Navigation system responding to language
 Safety and validation mechanisms
 Performance evaluation
 Demo scenarios with various language inputs

## Summary

This chapter covered the integration of Large Language Models with robotics:

 **LLM Integration**: Connecting language models to robotic systems
 **VisionLanguage Processing**: Combining visual perception with language understanding
 **Natural Language Control**: Interpreting human commands for robot action
 **Embodied AI**: Grounding language in physical robotic behavior
 **Safety Considerations**: Ensuring safe execution of LLMgenerated commands

The combination of LLMs with robotics enables natural humanrobot interaction, allowing users to command robots using everyday language rather than specialized programming. However, careful attention must be paid to grounding language in the physical world and ensuring safe, reliable robot behavior.