---
title: Ch20 - Capstone – Voice → Plan → Navigate → Manipulate
module: 4
chapter: 20
sidebar_label: Ch20: Capstone – Voice → Plan → Navigate → Manipulate
description: Complete capstone project integrating voice control, cognitive planning, navigation, and manipulation
tags: [capstone, integration, voice-control, planning, navigation, manipulation, ai-robotics]
difficulty: advanced
estimated_duration: 240
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Capstone: Voice → Plan → Navigate → Manipulate

## Learning Outcomes
- Integrate voice recognition, cognitive planning, navigation, and manipulation systems
- Create end-to-end robotic systems from voice command to physical action
- Implement safety validation throughout the pipeline
- Design multimodal interfaces connecting language, vision, and action
- Evaluate system performance for complex voice-controlled tasks
- Handle error recovery and graceful degradation in multi-component systems
- Create comprehensive logging and debugging systems
- Design user-friendly interfaces for complex robotic systems

## Theory

### Integrated Voice-to-Action Pipeline

The complete voice-to-action system involves multiple interconnected components that need to work seamlessly together. Each component affects the others, requiring careful integration and validation.

<MermaidDiagram chart={`
graph TD;
    A[User Voice Command] --> B[Whisper Transcription];
    B --> C[Command Parsing];
    C --> D[GPT-4o Task Planning];
    D --> E[Hierarchical Plan];
    
    F[Visual Input] --> G[Object Detection];
    G --> H[Environment Mapping];
    H --> I[Context Integration];
    
    J[Navigation System] --> K[Path Planning];
    K --> L[Obstacle Avoidance];
    L --> M[Safe Navigation];
    
    N[Manipulation System] --> O[Grasp Planning];
    O --> P[Collision Checking];
    P --> Q[Safe Manipulation];
    
    E --> R[Plan Validation];
    I --> R;
    R --> S[Execution Monitoring];
    
    S --> T[Navigation Execution];
    S --> U[Manipulation Execution];
    
    T --> V[Safety Validation];
    U --> V;
    V --> W[Physical Robot Action];
    
    W --> X[Feedback Generation];
    X --> Y[Voice Response];
    X --> Z[Visual Feedback];
    
    E --> AA[Recovery Planning];
    AA --> BB[Alternative Actions];
    BB --> W;
    
    style A fill:#E91E63,stroke:#AD1457,color:#fff;
    style W fill:#4CAF50,stroke:#388E3C,color:#fff;
    style Y fill:#2196F3,stroke:#0D47A1,color:#fff;
    style Z fill:#FF9800,stroke:#E65100,color:#fff;
    style D fill:#9C27B0,stroke:#4A148C,color:#fff;
    style V fill:#FF5722,stroke:#BF360C,color:#fff;
`} />

### Integration Challenges

**Latency Management**: Each component adds processing time; the overall system must maintain responsiveness.

**State Consistency**: Different components may have different views of the world state; synchronization is critical.

**Error Propagation**: Errors in early stages (voice recognition) cascade to later stages (action execution).

**Safety Validation**: Each planned action must be validated for safety considering all available context.

**Context Awareness**: The system must maintain and update context across all processing stages.

### Architecture Patterns

**Event-Driven Architecture**: Components communicate through event messages, allowing loose coupling and independent failure handling.

**State Machine Architecture**: Explicit state management for complex multi-step processes.

**Layered Architecture**: Clear separation between sensory input, cognitive processing, and action execution.

## Step-by-Step Labs

### Lab 1: Building the Voice-to-Action Pipeline

1. **Create the main integration node** (`voice_action_integration.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import openai
   import whisper
   import pyaudio
   import numpy as np
   import json
   import threading
   import queue
   import time
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist, Pose, Point
   from sensor_msgs.msg import LaserScan, Image
   from geometry_msgs.msg import PoseStamped, PointStamped
   from actionlib_msgs.msg import GoalStatusArray
   from typing import Dict, List, Optional, Any
   import torch

   class VoiceActionIntegration:
       def __init__(self, api_key: str, whisper_model_size: str = "base"):
           # Initialize ROS
           rospy.init_node('voice_action_integration', anonymous=True)
           
           # Initialize Whisper
           self.whisper_model = whisper.load_model(whisper_model_size)
           
           # Initialize OpenAI client
           self.openai_client = openai.OpenAI(api_key=api_key)
           
           # Publishers
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
           self.system_status_pub = rospy.Publisher('/system_status', String, queue_size=10)
           self.voice_response_pub = rospy.Publisher('/voice_response', String, queue_size=10)
           self.action_feedback_pub = rospy.Publisher('/action_feedback', String, queue_size=10)
           
           # Subscribers
           rospy.Subscriber('/voice_command', String, self.voice_command_callback)
           rospy.Subscriber('/scan', LaserScan, self.scan_callback)
           rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
           rospy.Subscriber('/move_base/status', GoalStatusArray, self.navigation_status_callback)
           rospy.Subscriber('/action_execution_status', String, self.action_status_callback)
           
           # Internal state
           self.laser_data = None
           self.image_data = None
           self.robot_position = Point(x=0.0, y=0.0, z=0.0)
           self.navigation_active = False
           self.action_queue = queue.Queue()
           self.state_lock = threading.Lock()
           
           # System configuration
           self.safe_distance = rospy.get_param('~safe_distance', 0.5)
           self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.3)
           self.max_angular_speed = rospy.get_param('~max_angular_speed', 0.5)
           self.command_timeout = rospy.get_param('~command_timeout', 30.0)
           
           # Command processing
           self.command_processing_thread = threading.Thread(target=self.process_commands)
           self.command_processing_thread.daemon = True
           self.command_processing_thread.start()
           
           rospy.loginfo("Voice-Action Integration System initialized")
       
       def scan_callback(self, msg: LaserScan):
           """Update laser scan data"""
           with self.state_lock:
               self.laser_data = msg
       
       def image_callback(self, msg: Image):
           """Update image data"""
           with self.state_lock:
               self.image_data = msg
       
       def navigation_status_callback(self, msg: GoalStatusArray):
           """Update navigation status"""
           with self.state_lock:
               if msg.status_list:
                   # Check the status of the most recent goal (last in list)
                   last_status = msg.status_list[-1]
                   if last_status.status == 3:  # Succeeded
                       self.navigation_active = False
                   elif last_status.status in [1, 2, 4, 5, 8]:  # Active/Aborted/etc.
                       self.navigation_active = True
       
       def action_status_callback(self, msg: String):
           """Handle action execution status"""
           try:
               status_data = json.loads(msg.data)
               action_id = status_data.get("action_id", "unknown")
               status = status_data.get("status", "unknown")
               
               if status == "failure":
                   rospy.logerr(f"Action {action_id} failed: {status_data.get('error', 'Unknown error')}")
                   self.handle_action_failure(action_id, status_data.get("error", "Unknown error"))
               elif status == "success":
                   rospy.loginfo(f"Action {action_id} completed successfully")
                   self.publish_feedback(f"Action {action_id} completed successfully")
               
           except json.JSONDecodeError:
               rospy.logerr("Invalid JSON in action status message")
       
       def voice_command_callback(self, msg: String):
           """Process incoming voice command"""
           command_text = msg.data
           rospy.loginfo(f"Received voice command: {command_text}")
           
           # Add command to processing queue
           command_item = {
               "text": command_text,
               "timestamp": rospy.Time.now().to_sec(),
               "processed": False,
               "attempts": 0,
               "max_attempts": 3
           }
           
           self.action_queue.put(command_item)
           self.publish_status(f"Command received: {command_text[:50]}...")
       
       def process_commands(self):
           """Process commands from the queue in a separate thread"""
           while not rospy.is_shutdown():
               try:
                   command_item = self.action_queue.get(timeout=1.0)
                   
                   # Process the command
                   success = self.process_single_command(command_item["text"])
                   
                   if not success:
                       if command_item["attempts"] < command_item["max_attempts"]:
                           # Put back in queue with incremented attempt count
                           command_item["attempts"] += 1
                           rospy.logwarn(f"Command failed, retrying ({command_item['attempts']}/{command_item['max_attempts']})")
                           time.sleep(1.0)  # Short delay before retry
                           self.action_queue.put(command_item)
                       else:
                           rospy.logerr(f"Command failed after {command_item['max_attempts']} attempts: {command_item['text']}")
                           self.publish_feedback(f"Command failed after maximum attempts: {command_item['text']}")
                   
                   self.action_queue.task_done()
                   
               except queue.Empty:
                   continue
               except Exception as e:
                   rospy.logerr(f"Error processing command: {e}")
                   continue
       
       def process_single_command(self, command_text: str) -> bool:
           """Process a single command through the entire pipeline"""
           rospy.loginfo(f"Processing command: {command_text}")
           
           # Step 1: Contextual understanding
           context = self.get_environment_context()
           
           # Step 2: Use GPT-4o to parse command and generate plan
           plan = self.generate_action_plan(command_text, context)
           
           if not plan:
               rospy.logerr("Failed to generate action plan")
               return False
           
           rospy.loginfo(f"Generated plan with {len(plan.get('actions', []))} actions")
           
           # Step 3: Validate plan for safety
           if not self.validate_plan_safety(plan):
               rospy.logerr("Plan failed safety validation")
               self.publish_feedback("Command plan rejected: Safety validation failed")
               return False
           
           # Step 4: Execute plan step by step
           execution_success = self.execute_plan(plan)
           
           if execution_success:
               rospy.loginfo("Command plan executed successfully")
               self.publish_feedback(f"Command completed successfully: {command_text[:30]}...")
               return True
           else:
               rospy.logerr("Command plan execution failed")
               return False
       
       def get_environment_context(self) -> Dict:
           """Get current environment context"""
           with self.state_lock:
               context = {
                   "robot_state": {
                       "position": {"x": self.robot_position.x, "y": self.robot_position.y},
                       "battery_level": 0.85,  # Would come from actual robot state in real implementation
                       "navigation_status": "ready" if not self.navigation_active else "executing"
                   },
                   "environment": {
                       "obstacles": self.get_obstacle_information(),
                       "known_locations": self.get_known_locations(),
                       "detected_objects": self.get_detected_objects(),
                       "navigation_map": None  # Would come from map server
                   },
                   "time_context": {
                       "hour_of_day": rospy.Time.now().to_sec() % 86400 / 3600,
                       "day_of_week": int(rospy.Time.now().to_sec() % 604800 / 86400)
                   },
                   "constraints": {
                       "safe_distance": self.safe_distance,
                       "max_speed": self.max_linear_speed,
                       "max_angular_speed": self.max_angular_speed
                   }
               }
               return context
       
       def get_obstacle_information(self) -> List[Dict]:
           """Extract obstacle information from laser scan"""
           if not self.laser_data:
               return []
           
           # Analyze laser scan for obstacles
           obstacles = []
           ranges = self.laser_data.ranges
           angle_min = self.laser_data.angle_min
           angle_increment = self.laser_data.angle_increment
           
           # Sample every 10th point to reduce computation
           for i in range(0, len(ranges), 10):
               if not (np.isinf(ranges[i]) or np.isnan(ranges[i])):
                   angle = angle_min + i * angle_increment
                   x = ranges[i] * np.cos(angle)
                   y = ranges[i] * np.sin(angle)
                   
                   if ranges[i] < self.safe_distance * 2:  # Consider as potential obstacle
                       obstacles.append({
                           "x": x,
                           "y": y,
                           "distance": ranges[i],
                           "angle": angle
                       })
           
           return obstacles
       
       def get_known_locations(self) -> List[Dict]:
           """Get known locations (would come from map or localization)"""
           # In real implementation, this would query a location database
           return [
               {"name": "kitchen", "x": 2.0, "y": 1.0},
               {"name": "living_room", "x": 1.0, "y": -1.0},
               {"name": "bedroom", "x": -1.0, "y": 1.5},
               {"name": "office", "x": 0.5, "y": 2.0},
               {"name": "charging_station", "x": -2.0, "y": -2.0}
           ]
       
       def get_detected_objects(self) -> List[Dict]:
           """Get detected objects (would come from perception system)"""
           # In real implementation, this would come from object detection
           # For this example, return empty list (would be populated by vision system)
           if self.image_data is not None:
               # Simulate object detection from image
               # This is a placeholder - real implementation would use actual object detection
               return [
                   {"name": "coffee_cup", "type": "container", "location": "kitchen", "distance": 1.5, "confidence": 0.9},
                   {"name": "book", "type": "stationery", "location": "living_room", "distance": 2.0, "confidence": 0.8}
               ]
           return []
       
       def generate_action_plan(self, command: str, context: Dict) -> Optional[Dict]:
           """Generate action plan using GPT-4o based on command and context"""
           prompt = f"""
           User Command: "{command}"
           
           Environment Context: {json.dumps(context, indent=2)}
           
           Available Actions:
           - navigate_to(location_name): Move robot to named location
           - approach_object(object_name): Move robot to specific object
           - inspect_object(object_name): Examine an object with sensors
           - grasp_object(object_name): Pick up an object (if manipulator equipped)
           - place_object(object_name, location): Place object at location
           - open_container(container_name): Open a container/door
           - close_container(container_name): Close a container/door
           - follow_person(person_name): Follow a person for some distance
           - wait(duration_seconds): Pause execution for specified duration
           - speak_response(text): Speak text response to user
           - find_person(person_name): Locate a specific person
           - escort_person(destination): Guide a person to destination
           - patrol_area(area_name): Move through predefined area
           - charge_robot(): Return to charging station
           
           Generate a detailed action plan that:
           1. Breaks down the command into specific actions
           2. Considers the environmental context
           3. Respects robot capabilities and constraints
           4. Includes safety checks before movement
           5. Handles potential failure modes
           6. Verifies completion of each step
           
           The plan should be in JSON format:
           {{
             "command_original": "{command}",
             "reasoning": "step-by-step reasoning about the plan",
             "actions": [
               {{
                 "id": 1,
                 "type": "action_type",
                 "parameters": {{"param1": "value1", "param2": "value2"}},
                 "description": "what this action accomplishes",
                 "preconditions": ["condition1", "condition2"],
                 "expected_effects": ["effect1", "effect2"],
                 "safety_check_needed": true,
                 "estimated_duration": 15.0,
                 "success_probability": 0.9
               }}
             ],
             "estimated_total_duration": 60.0,
             "overall_confidence": 0.75,
             "failure_recovery": [
               {{"condition": "obstacle_encountered", "action": "navigation_failed_recovery_plan"}},
               {{"condition": "object_not_found", "action": "search_alternative_objects"}},
               {{"condition": "grasp_failed", "action": "retry_grasp_with_different_approach"}}
             ]
           }}
           
           Respond with ONLY the JSON object, no other text.
           """
           
           try:
               response = self.openai_client.chat.completions.create(
                   model="gpt-4o",
                   messages=[
                       {
                           "role": "system",
                           "content": """You are an expert in robot task planning. Generate feasible plans that consider robot capabilities, environmental constraints, and safety. Respond only with valid JSON as specified."""
                       },
                       {
                           "role": "user", 
                           "content": prompt
                       }
                   ],
                   temperature=0.1,
                   max_tokens=2000
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON if wrapped in code block
               if response_text.startswith('```'):
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   if start_idx != -1 and end_idx != -1:
                       response_text = response_text[start_idx:end_idx]
               
               plan_data = json.loads(response_text)
               return plan_data
               
           except Exception as e:
               rospy.logerr(f"Error generating action plan: {e}")
               return None
       
       def validate_plan_safety(self, plan: Dict) -> bool:
           """Validate the safety of the generated plan"""
           actions = plan.get("actions", [])
           
           for action in actions:
               action_type = action.get("type", "")
               
               # Check navigation safety
               if action_type in ["navigate_to", "approach_object"]:
                   params = action.get("parameters", {})
                   
                   # Check if destination is safe based on current environment
                   if "location" in params:
                       if not self.is_safe_navigation_destination(params["location"]):
                           rospy.logerr(f"Safety check failed for destination: {params['location']}")
                           return False
                   elif "object_name" in params:
                       object_name = params["object_name"]
                       if not self.is_safe_to_approach_object(object_name):
                           rospy.logerr(f"Safety check failed for approaching object: {object_name}")
                           return False
               
               # Check manipulation safety
               elif action_type in ["grasp_object", "place_object"]:
                   if not self.are_manipulation_safeties_met():
                       rospy.logerr("Safety check failed for manipulation action")
                       return False
           
           return True
       
       def is_safe_navigation_destination(self, location_name: str) -> bool:
           """Check if navigation destination is safe"""
           # In real implementation, this would check:
           # - Known map for obstacles
           # - Dynamic obstacles from sensors
           # - Reachability of the location
           # - Safety of the area
           
           known_locations = self.get_known_locations()
           target_location = next((loc for loc in known_locations if loc["name"] == location_name), None)
           
           if not target_location:
               rospy.logwarn(f"Unknown location: {location_name}")
               return False
           
           # Check for obstacles in path
           if self.laser_data:
               # Simple check: ensure path is mostly clear
               front_ranges = self.laser_data.ranges[len(self.laser_data.ranges)//2-20:len(self.laser_data.ranges)//2+20]
               valid_ranges = [r for r in front_ranges if not (np.isinf(r) or np.isnan(r))]
               
               if valid_ranges and min(valid_ranges) < 0.5:  # Dangerous to move if obstacle within 0.5m
                   rospy.logwarn(f"Dangerous obstacle detected en route to {location_name}")
                   return False
           
           return True
       
       def is_safe_to_approach_object(self, object_name: str) -> bool:
           """Check if it's safe to approach an object"""
           # Check environment context for the object
           detected_objects = self.get_detected_objects()
           target_object = next((obj for obj in detected_objects if obj["name"] == object_name), None)
           
           if target_object:
               # Check if object is within safe distance range
               if target_object["distance"] > 3.0:  # Too far to reliably approach
                   rospy.logwarn(f"Object {object_name} is too far: {target_object['distance']:.2f}m")
                   return False
               
               if target_object["distance"] < 0.2:  # Too close already
                   rospy.loginfo(f"Already close to object {object_name}")
                   return True  # This is OK, just need to stay put
               
               # Check for obstacles in path to object
               if self.laser_data:
                   # Estimate object bearing based on approximate location
                   # This is a simplification in the example
                   return True  # In real implementation, verify path is clear
           
           # If object not detected, we can't safely approach
           rospy.logwarn(f"Cannot safely approach undetected object: {object_name}")
           return False
       
       def are_manipulation_safeties_met(self) -> bool:
           """Check if manipulator safety conditions are met"""
           # In real implementation, this would check:
           # - Robot position (is it in a safe area for manipulation?)
           # - Object detectability (can we see the object clearly?)
           # - Manipulator status (is it ready?)
           # - Surroundings (are there people nearby?)
           return True  # Placeholder value
       
       def execute_plan(self, plan: Dict) -> bool:
           """Execute the action plan"""
           actions = plan.get("actions", [])
           rospy.loginfo(f"Starting execution of plan with {len(actions)} actions")
           
           for i, action in enumerate(actions):
               rospy.loginfo(f"Executing action {i+1}/{len(actions)}: {action.get('type', 'unknown')}")
               
               # Publish execution status
               status_msg = String()
               status_msg.data = json.dumps({
                   "action_id": action.get("id", "unknown"),
                   "action_type": action.get("type", "unknown"),
                   "progress": f"{i+1}/{len(actions)}",
                   "status": "executing"
               })
               self.action_feedback_pub.publish(status_msg)
               
               # Execute action
               success = self.execute_single_action(action)
               
               if not success:
                   rospy.logerr(f"Action execution failed: {action}")
                   
                   # Try recovery based on plan
                   recovery_success = self.attempt_recovery(plan, i, action)
                   if not recovery_success:
                       rospy.logerr("Recovery attempt failed, terminating plan")
                       return False
               
               # Small delay between actions
               rospy.sleep(0.5)
           
           rospy.loginfo("Plan execution completed successfully")
           return True
       
       def execute_single_action(self, action: Dict) -> bool:
           """Execute a single action"""
           action_type = action.get("type", "")
           parameters = action.get("parameters", {})
           
           if action_type == "navigate_to":
               return self.execute_navigation_action(parameters)
           elif action_type == "approach_object":
               return self.execute_approach_action(parameters)
           elif action_type == "inspect_object":
               return self.execute_inspection_action(parameters)
           elif action_type == "grasp_object":
               return self.execute_grasp_action(parameters)
           elif action_type == "place_object":
               return self.execute_place_action(parameters)
           elif action_type == "speak_response":
               return self.execute_speak_action(parameters)
           elif action_type == "wait":
               return self.execute_wait_action(parameters)
           elif action_type == "open_container":
               return self.execute_open_container_action(parameters)
           elif action_type == "close_container":
               return self.execute_close_container_action(parameters)
           else:
               rospy.logwarn(f"Unknown action type: {action_type}")
               return True  # Don't fail on unknown actions, just skip them
       
       def execute_navigation_action(self, params: Dict) -> bool:
           """Execute navigation action"""
           location_name = params.get("location")
           
           if not location_name:
               rospy.logerr("Navigation action missing location parameter")
               return False
           
           # Find the location in known locations
           known_locations = self.get_known_locations()
           target_location = next((loc for loc in known_locations if loc["name"] == location_name), None)
           
           if not target_location:
               rospy.logerr(f"Unknown location: {location_name}")
               return False
           
           # Create and publish navigation goal
           goal = PoseStamped()
           goal.header.frame_id = "map"
           goal.header.stamp = rospy.Time.now()
           goal.pose.position.x = target_location["x"]
           goal.pose.position.y = target_location["y"]
           goal.pose.position.z = 0.0
           
           # Use simple quaternion (0, 0, 0, 1) for no rotation
           goal.pose.orientation.w = 1.0
           
           self.goal_pub.publish(goal)
           rospy.loginfo(f"Navigation goal sent to {location_name} at ({target_location['x']}, {target_location['y']})")
           
           # Wait for navigation to complete (with timeout)
           start_time = rospy.Time.now().to_sec()
           timeout = 60.0  # 1 minute timeout
           
           rate = rospy.Rate(10)  # 10 Hz
           while (rospy.Time.now().to_sec() - start_time) < timeout:
               if not self.navigation_active:
                   rospy.loginfo(f"Navigation to {location_name} completed successfully")
                   return True
               rate.sleep()
           
           rospy.logerr(f"Navigation to {location_name} timed out after {timeout} seconds")
           return False
       
       def execute_approach_action(self, params: Dict) -> bool:
           """Execute object approach action"""
           object_name = params.get("object_name")
           
           if not object_name:
               rospy.logerr("Approach action missing object_name parameter")
               return False
           
           # Get object location from perception system
           detected_objects = self.get_detected_objects()
           target_object = next((obj for obj in detected_objects if obj["name"] == object_name), None)
           
           if not target_object:
               rospy.logerr(f"Object {object_name} not detected")
               return False
           
           # Approach strategy: navigate to a position 0.5m in front of the object
           # This is simplified - real implementation would require more complex path planning
           object_dist = target_object["distance"]
           if object_dist > 0.5:  # Only navigate if object is farther than our target distance
               # Calculate target position (simplified - assumes object is in front of robot)
               target_x = target_object["x"] if "x" in target_object else 0.5
               target_y = target_object["y"] if "y" in target_object else 0.0
               
               goal = PoseStamped()
               goal.header.frame_id = "map"
               goal.header.stamp = rospy.Time.now()
               goal.pose.position.x = target_x
               goal.pose.position.y = target_y
               goal.pose.position.z = 0.0
               goal.pose.orientation.w = 1.0
               
               self.goal_pub.publish(goal)
               rospy.loginfo(f"Approach goal sent for {object_name}")
               
               # Wait for navigation to complete
               start_time = rospy.Time.now().to_sec()
               timeout = 30.0  # 30 second timeout for approach
               
               rate = rospy.Rate(10)  # 10 Hz
               while (rospy.Time.now().to_sec() - start_time) < timeout:
                   if not self.navigation_active:
                       rospy.loginfo(f"Approach to {object_name} completed")
                       return True
                   rate.sleep()
               
               rospy.logerr(f"Approach to {object_name} timed out")
               return False
           else:
               # Already close enough to object
               rospy.loginfo(f"Already close enough to {object_name}")
               return True
       
       def execute_speak_action(self, params: Dict) -> bool:
           """Execute speech action"""
           text = params.get("text", "")
           if text:
               self.publish_feedback(f"Speaking: {text}")
               rospy.loginfo(f"Speaking: {text}")
               return True
           else:
               rospy.logwarn("Speak action with empty text")
               return False
       
       def execute_wait_action(self, params: Dict) -> bool:
           """Execute wait action"""
           duration = params.get("duration", 1.0)
           rospy.loginfo(f"Waiting for {duration} seconds")
           rospy.sleep(duration)
           return True
       
       def execute_inspection_action(self, params: Dict) -> bool:
           """Execute object inspection action"""
           object_name = params.get("object_name")
           rospy.loginfo(f"Inspecting object: {object_name}")
           
           # In real implementation, this would trigger perception routines
           # For now, just simulate with a delay
           rospy.sleep(2.0)
           
           # Return to report that inspection is complete
           self.publish_feedback(f"Inspection of {object_name} completed")
           return True
       
       def attempt_recovery(self, plan: Dict, failed_step: int, failed_action: Dict) -> bool:
           """Attempt to recover from action failure"""
           recovery_strategies = plan.get("failure_recovery", [])
           failed_condition = f"{failed_action.get('type', 'unknown')}_failed"
           
           # Find recovery strategy for this condition
           recovery_strategy = next((rec for rec in recovery_strategies if rec.get("condition") == failed_condition), None)
           
           if recovery_strategy:
               rospy.loginfo(f"Attempting recovery: {recovery_strategy['action']}")
               
               # For this example, we'll just log the recovery attempt
               # In a real system, you'd implement specific recovery actions
               if "navigation_failed" in recovery_strategy["action"]:
                   # Try alternative navigation approach
                   rospy.loginfo("Trying alternative navigation approach")
                   return True  # Simulate successful recovery
               elif "object_not_found" in recovery_strategy["action"]:
                   # Try searching in wider area
                   rospy.loginfo("Expanding search area for object")
                   return True  # Simulate successful recovery
               elif "grasp_failed" in recovery_strategy["action"]:
                   # Retry with different grasp approach
                   rospy.loginfo("Retrying grasp with different approach")
                   return True  # Simulate successful recovery
           
           rospy.logwarn(f"No recovery strategy found for condition: {failed_condition}")
           return False  # No recovery strategy available
       
       def publish_status(self, message: str):
           """Publish system status"""
           status_msg = String()
           status_msg.data = json.dumps({
               "message": message,
               "timestamp": rospy.Time.now().to_sec()
           })
           self.system_status_pub.publish(status_msg)
       
       def publish_feedback(self, message: str):
           """Publish action feedback"""
           feedback_msg = String()
           feedback_msg.data = json.dumps({
               "message": message,
               "timestamp": rospy.Time.now().to_sec()
           })
           self.voice_response_pub.publish(feedback_msg)

   def main():
       api_key = rospy.get_param('~openai_api_key', '')
       if not api_key:
           api_key = input("Enter OpenAI API key: ").strip()
       
       if not api_key:
           rospy.logerr("No OpenAI API key provided!")
           return
       
       system = VoiceActionIntegration(api_key)
       
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down Voice-Action Integration System")

   if __name__ == '__main__':
       main()
   ```

### Lab 2: Creating a Safety Validation Layer

1. **Create a safety validation system** (`safety_validator.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import numpy as np
   from std_msgs.msg import String, Bool
   from sensor_msgs.msg import LaserScan, PointCloud2
   from geometry_msgs.msg import Twist, PoseStamped
   from nav_msgs.msg import Odometry
   from geometry_msgs.msg import Point
   from typing import Dict, List, Optional, Tuple
   import threading
   import math
   import time

   class SafetyValidator:
       def __init__(self):
           # Initialize ROS node
           rospy.init_node('safety_validator', anonymous=True)
           
           # Publishers
           self.safety_status_pub = rospy.Publisher('/safety_status', String, queue_size=10)
           self.emergency_stop_pub = rospy.Publisher('/emergency_stop', Bool, queue_size=10)
           self.safe_command_pub = rospy.Publisher('/safe_cmd_vel', Twist, queue_size=10)
           self.safety_violation_pub = rospy.Publisher('/safety_violations', String, queue_size=10)
           
           # Subscribers
           rospy.Subscriber('/cmd_vel', Twist, self.unsafe_command_callback)
           rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
           rospy.Subscriber('/odom', Odometry, self.odometry_callback)
           rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.navigation_goal_callback)
           
           # Internal state
           self.laser_scan = None
           self.odom_data = None
           self.robot_pose = Point(x=0.0, y=0.0, z=0.0)
           self.robot_velocity = Point(x=0.0, y=0.0, z=0.0)
           self.safety_violations = []
           self.max_violations_history = 100
           
           # Safety parameters
           self.safety_distance = rospy.get_param('~safety_distance', 0.5)  # meters
           self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.4)
           self.max_angular_speed = rospy.get_param('~max_angular_speed', 0.6)
           self.max_deceleration = rospy.get_param('~max_deceleration', 2.0)  # m/s^2
           self.emergency_stop_distance = rospy.get_param('~emergency_stop_distance', 0.3)
           self.check_frequency = rospy.get_param('~check_frequency', 20.0)  # Hz
           
           # Robot physical dimensions
           self.robot_radius = rospy.get_param('~robot_radius', 0.3)  # meters
           
           # State flags
           self.emergency_stop_active = False
           self.last_safe_command_time = rospy.Time.now().to_sec()
           self.safety_check_interval = 1.0 / self.check_frequency
           
           # Command validation
           self.pending_unsafe_command = None
           self.unsafe_command_lock = threading.Lock()
           
           # Start safety monitoring
           self.safety_timer = rospy.Timer(rospy.Duration(1.0/self.check_frequency), self.safety_check_callback)
           
           rospy.loginfo("Safety Validator initialized and monitoring active")
       
       def laser_scan_callback(self, msg: LaserScan):
           """Update laser scan data"""
           self.laser_scan = msg
       
       def odometry_callback(self, msg: Odometry):
           """Update robot odometry data"""
           self.odom_data = msg
           self.robot_pose.x = msg.pose.pose.position.x
           self.robot_pose.y = msg.pose.pose.position.y
           self.robot_pose.z = msg.pose.pose.position.z
           
           # Get linear velocity from twist
           self.robot_velocity.x = msg.twist.twist.linear.x
           self.robot_velocity.y = msg.twist.twist.linear.y
           self.robot_velocity.z = msg.twist.twist.linear.z
       
       def navigation_goal_callback(self, msg: PoseStamped):
           """Validate navigation goal before execution"""
           goal_x = msg.pose.position.x
           goal_y = msg.pose.position.y
           
           # Check if goal is in a safe location
           if self.is_position_safe(goal_x, goal_y):
               rospy.loginfo(f"Navigation goal ({goal_x}, {goal_y}) is safe, allowing execution")
               # In real implementation, forward to navigation stack
           else:
               rospy.logerr(f"Navigation goal ({goal_x}, {goal_y}) is unsafe, blocking execution")
               self.publish_safety_violation({
                   "type": "unsafe_navigation_goal",
                   "position": (goal_x, goal_y),
                   "timestamp": rospy.Time.now().to_sec()
               })
       
       def is_position_safe(self, x: float, y: float) -> bool:
           """Check if a position is safe to navigate to"""
           if not self.laser_scan:
               # If no sensor data available, be conservative
               rospy.logwarn("No laser data available for position safety check, assuming unsafe")
               return False
           
           # Convert position to robot frame
           dx = x - self.robot_pose.x
           dy = y - self.robot_pose.y
           distance_to_position = math.sqrt(dx*dx + dy*dy)
           
           # Check if position is too close to current obstacles
           # This is a simplified check - in real implementation, use path planning
           min_range = min(self.laser_scan.ranges) if self.laser_scan.ranges else float('inf')
           if min_range < self.safety_distance:
               rospy.logwarn(f"Position ({x}, {y}) potentially unsafe due to nearby obstacles")
               return False
           
           return True
       
       def unsafe_command_callback(self, msg: Twist):
           """Validate and potentially modify unsafe commands"""
           with self.unsafe_command_lock:
               # Check if emergency stop is active
               if self.emergency_stop_active:
                   # Discard command, robot is in emergency stop state
                   rospy.logwarn("Emergency stop active, discarding command")
                   return
               
               # Validate the command
               safe_command = self.validate_and_modify_command(msg)
               
               if safe_command:
                   # Publish the safe command
                   self.safe_command_pub.publish(safe_command)
                   rospy.logdebug(f"Safe command published: linear.x={safe_command.linear.x}, angular.z={safe_command.angular.z}")
                   
                   # Update last safe command time
                   self.last_safe_command_time = rospy.Time.now().to_sec()
               else:
                   # Command was rejected
                   rospy.logwarn("Command rejected for safety reasons")
                   self.publish_safety_violation({
                       "type": "rejected_command",
                       "command": {"linear_x": msg.linear.x, "angular_z": msg.angular.z},
                       "reason": "Safety validation failed",
                       "timestamp": rospy.Time.now().to_sec()
                   })
       
       def validate_and_modify_command(self, cmd: Twist) -> Optional[Twist]:
           """Validate and modify command to ensure safety"""
           if not self.laser_scan:
               # If no laser data, only allow zero velocity commands
               if cmd.linear.x == 0.0 and cmd.angular.z == 0.0:
                   return cmd
               else:
                   rospy.logwarn("No laser data, rejecting non-zero velocity command")
                   return None
           
           # Validate linear speed
           if abs(cmd.linear.x) > self.max_linear_speed:
               rospy.logwarn(f"Linear velocity {cmd.linear.x} exceeds limit {self.max_linear_speed}")
               cmd.linear.x = np.clip(cmd.linear.x, -self.max_linear_speed, self.max_linear_speed)
           
           # Validate angular speed
           if abs(cmd.angular.z) > self.max_angular_speed:
               rospy.logwarn(f"Angular velocity {cmd.angular.z} exceeds limit {self.max_angular_speed}")
               cmd.angular.z = np.clip(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed)
           
           # Check for forward obstacles if trying to move forward
           if cmd.linear.x > 0:
               if self.has_obstacles_ahead(self.safety_distance):
                   # Calculate required stopping distance
                   current_speed = self.robot_velocity.x
                   if current_speed > 0:
                       stopping_distance = (current_speed ** 2) / (2 * self.max_deceleration)
                   else:
                       stopping_distance = 0
                   
                   # If we can't stop in time, slow down or stop
                   obstacle_distance = self.get_closest_obstacle_distance()
                   if obstacle_distance < (stopping_distance + self.robot_radius + 0.1):  # +0.1m safety margin
                       # Reduce speed proportionally to distance
                       new_speed = (obstacle_distance - self.robot_radius - 0.1) / (self.safety_distance - self.robot_radius - 0.1) * self.max_linear_speed
                       cmd.linear.x = max(0.0, min(cmd.linear.x, new_speed))
                       rospy.loginfo(f"Reducing forward speed due to obstacle at {obstacle_distance:.2f}m: {cmd.linear.x:.2f}m/s")
           
           # Check for backward obstacles if trying to move backward
           if cmd.linear.x < 0:
               if self.has_obstacles_behind(self.safety_distance):
                   current_speed = abs(self.robot_velocity.x)
                   if current_speed > 0:
                       stopping_distance = (current_speed ** 2) / (2 * self.max_deceleration)
                   else:
                       stopping_distance = 0
                   
                   # Similar check for backward obstacles
                   obstacle_distance = self.get_closest_backward_obstacle_distance()
                   if obstacle_distance < (stopping_distance + self.robot_radius + 0.1):
                       new_speed = -min(abs(cmd.linear.x), 
                                      max(0.0, (obstacle_distance - self.robot_radius - 0.1) / (self.safety_distance - self.robot_radius - 0.1) * self.max_linear_speed))
                       cmd.linear.x = max(new_speed, cmd.linear.x)
                       rospy.loginfo(f"Reducing backward speed due to rear obstacle")
           
           # Check if emergency distance is violated
           if self.has_emergency_obstacles():
               rospy.logerr("EMERGENCY: Obstacle too close, activating emergency stop")
               self.activate_emergency_stop()
               return None
           
           return cmd
       
       def has_obstacles_ahead(self, check_distance: float) -> bool:
           """Check for obstacles in the forward direction"""
           if not self.laser_scan:
               return False
           
           # Check front quarter of laser scan (simplified - check middle portion)
           n_ranges = len(self.laser_scan.ranges)
           front_start = n_ranges // 2 - n_ranges // 8  # 3/4 to 5/8 of ranges (front left-right)
           front_end = n_ranges // 2 + n_ranges // 8    # 1/2 + 1/8 = 5/8 to 7/8 of ranges (front right-left)
           
           for i in range(front_start, front_end):
               if i < len(self.laser_scan.ranges) and self.laser_scan.ranges[i] < check_distance:
                   if not (np.isinf(self.laser_scan.ranges[i]) or np.isnan(self.laser_scan.ranges[i])):
                       return True
           return False
       
       def has_obstacles_behind(self, check_distance: float) -> bool:
           """Check for obstacles in the rear direction"""
           if not self.laser_scan:
               return False
           
           # Check rear quarter of laser scan
           n_ranges = len(self.laser_scan.ranges)
           rear_start = 3 * n_ranges // 4
           rear_end = n_ranges
           
           for i in range(rear_start, rear_end):
               if self.laser_scan.ranges[i] < check_distance:
                   if not (np.isinf(self.laser_scan.ranges[i]) or np.isnan(self.laser_scan.ranges[i])):
                       return True
           
           # Check front part as well (wrapping around)
           front_start = 0
           front_end = n_ranges // 4
           for i in range(front_start, front_end):
               if self.laser_scan.ranges[i] < check_distance:
                   if not (np.isinf(self.laser_scan.ranges[i]) or np.isnan(self.laser_scan.ranges[i])):
                       return True
           
           return False
       
       def has_emergency_obstacles(self) -> bool:
           """Check for obstacles in emergency stop range"""
           if not self.laser_scan:
               return False
           
           # Check for obstacles closer than emergency stop distance
           for range_val in self.laser_scan.ranges:
               if not (np.isinf(range_val) or np.isnan(range_val)) and range_val < self.emergency_stop_distance:
                   return True
           
           return False
       
       def get_closest_obstacle_distance(self) -> float:
           """Get distance to closest obstacle in front"""
           if not self.laser_scan:
               return float('inf')
           
           n_ranges = len(self.laser_scan.ranges)
           front_start = n_ranges // 2 - n_ranges // 8
           front_end = n_ranges // 2 + n_ranges // 8
           
           min_distance = float('inf')
           for i in range(front_start, front_end):
               if i < len(self.laser_scan.ranges):
                   range_val = self.laser_scan.ranges[i]
                   if not (np.isinf(range_val) or np.isnan(range_val)):
                       min_distance = min(min_distance, range_val)
           
           return min_distance
       
       def get_closest_backward_obstacle_distance(self) -> float:
           """Get distance to closest obstacle in rear"""
           if not self.laser_scan:
               return float('inf')
           
           n_ranges = len(self.laser_scan.ranges)
           rear_ranges = []
           
           # Rear portion
           rear_start = 3 * n_ranges // 4
           rear_end = n_ranges
           rear_ranges.extend(self.laser_scan.ranges[rear_start:rear_end])
           
           # Front wrapping portion
           front_start = 0
           front_end = n_ranges // 4
           rear_ranges.extend(self.laser_scan.ranges[front_start:front_end])
           
           valid_distances = [r for r in rear_ranges if not (np.isinf(r) or np.isnan(r))]
           return min(valid_distances) if valid_distances else float('inf')
       
       def safety_check_callback(self, event):
           """Periodic safety checks"""
           # Check for stale sensor data
           if self.laser_scan:
               time_since_scan = rospy.Time.now().to_sec() - event.current_real.to_sec()
               if time_since_scan > 1.0:  # If laser scan is more than 1 second old
                   rospy.logwarn("Laser scan data is stale, entering safety mode")
                   self.reduce_speed_for_stale_sensors()
           
           # Check for stuck robot (command sent but no movement)
           if self.odom_data and self.laser_scan:
               # Check if robot should be moving but isn't
               if abs(self.robot_velocity.x) < 0.01 and abs(self.robot_velocity.y) < 0.01:  # Robot not moving
                   # If there are no obstacles and robot should be moving, this might indicate a problem
                   min_dist = min(self.laser_scan.ranges) if self.laser_scan.ranges else float('inf')
                   if min_dist > 0.5:  # No obstacles nearby
                       # Check if we recently sent a command
                       # (In real implementation, we'd track the last command sent)
                       pass  # Potential stuck robot detection
       
       def activate_emergency_stop(self):
           """Activate emergency stop"""
           if not self.emergency_stop_active:
               rospy.logerr("EMERGENCY STOP ACTIVATED")
               self.emergency_stop_active = True
               
               # Send stop command
               stop_cmd = Twist()
               self.safe_command_pub.publish(stop_cmd)
               
               # Publish emergency stop signal
               emergency_msg = Bool()
               emergency_msg.data = True
               self.emergency_stop_pub.publish(emergency_msg)
               
               # Log the violation
               self.publish_safety_violation({
                   "type": "emergency_stop_activated",
                   "reason": "Obstacle too close",
                   "timestamp": rospy.Time.now().to_sec()
               })
       
       def deactivate_emergency_stop(self):
           """Deactivate emergency stop"""
           if self.emergency_stop_active:
               rospy.loginfo("EMERGENCY STOP DEACTIVATED")
               self.emergency_stop_active = False
               
               # Publish emergency stop reset
               emergency_msg = Bool()
               emergency_msg.data = False
               self.emergency_stop_pub.publish(emergency_msg)
       
       def reduce_speed_for_stale_sensors(self):
           """Reduce speed when sensor data is stale"""
           # In real implementation, this would send reduced speed commands to the robot
           # For now, we'll just log the action
           rospy.logwarn("Reducing robot speed due to stale sensor data")
       
       def publish_safety_violation(self, violation: Dict):
           """Publish safety violation information"""
           violation_msg = String()
           violation_msg.data = json.dumps(violation)
           self.safety_violation_pub.publish(violation_msg)
           
           # Add to internal history
           self.safety_violations.append(violation)
           if len(self.safety_violations) > self.max_violations_history:
               self.safety_violations = self.safety_violations[-self.max_violations_history:]
       
       def get_safety_status(self) -> Dict:
           """Get current safety status"""
           status = {
               "timestamp": rospy.Time.now().to_sec(),
               "emergency_stop_active": self.emergency_stop_active,
               "closest_obstacle_distance": self.get_closest_obstacle_distance() if self.laser_scan else float('inf'),
               "robot_velocity": {
                   "linear": self.robot_velocity.x,
                   "angular": self.robot_velocity.z
               },
               "robot_position": {
                   "x": self.robot_pose.x,
                   "y": self.robot_pose.y
               },
               "recent_violations_count": len(self.safety_violations),
               "last_violation": self.safety_violations[-1] if self.safety_violations else None
           }
           
           status_msg = String()
           status_msg.data = json.dumps(status)
           self.safety_status_pub.publish(status_msg)
           
           return status

   def main():
       validator = SafetyValidator()
       
       # Start a timer to publish safety status periodically
       status_timer = rospy.Timer(rospy.Duration(1.0), lambda event: validator.get_safety_status())
       
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down Safety Validator")

   if __name__ == '__main__':
       main()
   ```

### Lab 3: Implementing the Complete Integration System

1. **Create a launch file for the complete system** (`launch/voice_to_action_system.launch.py`):
   ```python
   #!/usr/bin/env python3

   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       # Declare launch arguments
       openai_api_key_arg = DeclareLaunchArgument(
           'openai_api_key',
           default_value='',
           description='OpenAI API key for GPT-4o access'
       )
       
       use_sim_time_arg = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true'
       )
       
       # Get launch configuration variables
       openai_api_key = LaunchConfiguration('openai_api_key')
       use_sim_time = LaunchConfiguration('use_sim_time')
       
       # Create nodes
       voice_action_node = Node(
           package='robot_voice_control',
           executable='voice_action_integration.py',
           name='voice_action_integration',
           parameters=[
               {'openai_api_key': openai_api_key},
               {'use_sim_time': use_sim_time},
               {'safe_distance': 0.5},
               {'max_linear_speed': 0.3},
               {'max_angular_speed': 0.5}
           ],
           remappings=[
               ('/voice_command', '/voice_transcription/text'),
               ('/cmd_vel', '/unprotected_cmd_vel'),
               ('/move_base_simple/goal', '/move_base/goal')
           ],
           output='screen'
       )
       
       safety_validator_node = Node(
           package='robot_voice_control',
           executable='safety_validator.py',
           name='safety_validator',
           parameters=[
               {'use_sim_time': use_sim_time},
               {'safety_distance': 0.5},
               {'max_linear_speed': 0.4},
               {'max_angular_speed': 0.6},
               {'emergency_stop_distance': 0.3},
               {'robot_radius': 0.3}
           ],
           remappings=[
               ('/cmd_vel', '/unprotected_cmd_vel'),
               ('/safe_cmd_vel', '/cmd_vel')
           ],
           output='screen'
       )
       
       # Optional: Add a simple voice recognition node if needed
       # This would typically be a separate package like 'vosk_ros' or 'google_stt_ros'
       voice_recognition_node = Node(
           package='vosk_ros',
           executable='vosk_node',
           name='vosk_node',
           parameters=[
               {'model': 'model_path_here'},  # You'd specify path to vosk model
               {'sample_rate': 16000.0}
           ],
           output='screen'
       )
       
       # Add other necessary nodes for a complete system:
       # - Robot driver
       # - Navigation stack
       # - Perception pipeline (if manipulator included)
       
       # Create launch description
       ld = LaunchDescription()
       
       # Add launch arguments
       ld.add_action(openai_api_key_arg)
       ld.add_action(use_sim_time_arg)
       
       # Add nodes
       ld.add_action(voice_action_node)
       ld.add_action(safety_validator_node)
       # ld.add_action(voice_recognition_node)  # Uncomment if using voice recognition
       
       return ld
   ```

2. **Create a comprehensive test script** (`test_voice_pipeline.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   from std_msgs.msg import String
   import time
   import json

   class VoicePipelineTester:
       def __init__(self):
           rospy.init_node('voice_pipeline_tester', anonymous=True)
           
           # Publishers for testing
           self.voice_cmd_pub = rospy.Publisher('/voice_command', String, queue_size=10)
           self.status_sub = rospy.Subscriber('/system_status', String, self.status_callback)
           self.feedback_sub = rospy.Subscriber('/action_feedback', String, self.feedback_callback)
           
           self.test_results = []
           self.current_test = None
           
           rospy.loginfo("Voice Pipeline Tester initialized")
       
       def status_callback(self, msg):
           """Handle system status updates"""
           try:
               status_data = json.loads(msg.data)
               if self.current_test:
                   rospy.loginfo(f"Status for test {self.current_test['name']}: {status_data.get('message', 'no message')}")
           except json.JSONDecodeError:
               rospy.logerr("Invalid JSON in status message")
       
       def feedback_callback(self, msg):
           """Handle action feedback"""
           try:
               feedback_data = json.loads(msg.data)
               if self.current_test:
                   rospy.loginfo(f"Feedback for test {self.current_test['name']}: {feedback_data.get('message', 'no message')}")
                   # Add to test results
                   self.current_test["feedback"].append(feedback_data)
           except json.JSONDecodeError:
               rospy.logerr("Invalid JSON in feedback message")
       
       def run_test_suite(self):
           """Run comprehensive test suite"""
           rospy.loginfo("Starting voice pipeline test suite")
           
           # Test 1: Simple navigation command
           self.run_test("simple_navigation", "Go to kitchen")
           
           # Test 2: Complex command with object interaction
           self.run_test("object_interaction", "Approach the red cup in the kitchen and inspect it")
           
           # Test 3: Safety validation (should be blocked if obstacles in path)
           self.run_test("safety_validation", "Move forward toward the obstacle")
           
           # Test 4: Multi-step command
           self.run_test("multi_step", "Navigate to bedroom, find John, escort him to kitchen")
           
           # Test 5: Recovery from failure
           self.run_test("recovery", "Go to office and bring me the green notebook")
           
           rospy.loginfo("Test suite completed")
           self.generate_test_report()
       
       def run_test(self, test_name: str, command: str):
           """Run a single test case"""
           rospy.loginfo(f"Running test: {test_name} - Command: '{command}'")
           
           self.current_test = {
               "name": test_name,
               "command": command,
               "start_time": rospy.Time.now().to_sec(),
               "feedback": []
           }
           
           # Send command
           cmd_msg = String()
           cmd_msg.data = command
           self.voice_cmd_pub.publish(cmd_msg)
           
           # Wait for execution (max 30 seconds)
           start_time = rospy.Time.now().to_sec()
           timeout = 30.0
           
           while (rospy.Time.now().to_sec() - start_time) < timeout:
               rospy.sleep(0.1)
           
           # Record results
           self.current_test["end_time"] = rospy.Time.now().to_sec()
           self.current_test["duration"] = self.current_test["end_time"] - self.current_test["start_time"]
           
           # Determine success based on feedback
           success = False
           for feedback in self.current_test["feedback"]:
               if "completed successfully" in feedback.get("message", "").lower():
                   success = True
                   break
           
           self.current_test["success"] = success
           self.test_results.append(self.current_test)
           
           # Log result
           status = "SUCCESS" if success else "FAILED"
           rospy.loginfo(f"Test {test_name} completed with status: {status}")
       
       def generate_test_report(self):
           """Generate test report"""
           rospy.loginfo("\n=== VOICE PIPELINE TEST RESULTS ===")
           
           for test in self.test_results:
               status = "✓ PASS" if test["success"] else "✗ FAIL"
               rospy.loginfo(f"{status} {test['name']}: '{test['command']}' ({test['duration']:.2f}s)")
           
           total_tests = len(self.test_results)
           passed_tests = sum(1 for t in self.test_results if t["success"])
           success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
           
           rospy.loginfo(f"\nOverall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")

   def main():
       tester = VoicePipelineTester()
       
       # Run tests after a brief delay to allow system to initialize
       rospy.sleep(5.0)
       tester.run_test_suite()
       
       rospy.spin()

   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's a complete integrated system that demonstrates the full voice-to-action pipeline:

```python
#!/usr/bin/env python3
# complete_voice_to_action_system.py

import rospy
import openai
import whisper
import pyaudio
import numpy as np
import json
import threading
import queue
import time
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from actionlib_msgs.msg import GoalStatusArray
from typing import Dict, List, Optional
import torch
import wave
import tempfile
import os

class CompleteVoiceToActionSystem:
    """Complete system integrating voice recognition, cognitive planning, navigation and manipulation"""
    
    def __init__(self, api_key: str, whisper_model_size: str = "base"):
        # Initialize ROS
        rospy.init_node('complete_voice_to_action_system', anonymous=True)
        
        # Initialize models
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.openai_client = openai.OpenAI(api_key=api_key)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.system_status_pub = rospy.Publisher('/system_status', String, queue_size=10)
        self.voice_response_pub = rospy.Publisher('/voice_response', String, queue_size=10)
        self.action_feedback_pub = rospy.Publisher('/action_feedback', String, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/voice_command', String, self.voice_command_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.navigation_status_callback)
        rospy.Subscriber('/action_execution_status', String, self.action_status_callback)
        
        # Internal state
        self.laser_data = None
        self.image_data = None
        self.robot_position = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.navigation_active = False
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.voice_threshold = 0.01
        self.min_voice_duration = 0.5  # seconds
        self.min_silence_duration = 1.0  # seconds before stopping recording
        
        # Command processing
        self.command_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_commands)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # System configuration
        self.safe_distance = rospy.get_param('~safe_distance', 0.5)
        self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.3)
        self.max_angular_speed = rospy.get_param('~max_angular_speed', 0.5)
        self.command_timeout = rospy.get_param('~command_timeout', 30.0)
        
        # For voice activation
        self.is_listening = False
        self.listening_thread = None
        self.voice_activation_enabled = rospy.get_param('~voice_activation', False)
        
        rospy.loginfo("Complete Voice-to-Action System initialized")
    
    def start_voice_activation(self):
        """Start voice activation if enabled"""
        if self.voice_activation_enabled:
            self.is_listening = True
            self.listening_thread = threading.Thread(target=self.voice_activation_loop)
            self.listening_thread.daemon = True
            self.listening_thread.start()
            rospy.loginfo("Voice activation started")
    
    def voice_activation_loop(self):
        """Continuous voice activation loop"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        recording = False
        frames = []
        voice_active_count = 0
        silence_count = 0
        voice_frames_needed = int(self.min_voice_duration * self.rate / self.chunk)
        silence_frames_needed = int(self.min_silence_duration * self.rate / self.chunk)
        
        try:
            while self.is_listening:
                data = stream.read(self.chunk, exception_on_overflow=False)
                
                # Calculate RMS for voice detection
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_array ** 2))
                
                if rms > self.voice_threshold:
                    # Voice detected
                    voice_active_count += 1
                    silence_count = 0
                    
                    if not recording:
                        if voice_active_count >= voice_frames_needed:
                            # Start recording
                            recording = True
                            frames = []
                            rospy.loginfo("Voice activation detected, starting recording...")
                    
                    if recording:
                        frames.append(data)
                else:
                    # Silence detected
                    voice_active_count = 0
                    if recording:
                        silence_count += 1
                        frames.append(data)
                        
                        if silence_count >= silence_frames_needed:
                            # End of speech detected
                            if len(frames) > voice_frames_needed:  # Ensure minimum speech length
                                rospy.loginfo(f"Speech detected, saving {len(frames)} frames for transcription...")
                                
                                # Save to temp file
                                temp_filename = self.save_audio_frames(frames)
                                
                                # Transcribe and process
                                transcript = self.transcribe_audio(temp_filename)
                                
                                if transcript and transcript.strip():
                                    rospy.loginfo(f"Transcribed: {transcript}")
                                    
                                    # Add to command queue
                                    self.command_queue.put({
                                        "text": transcript.strip(),
                                        "timestamp": rospy.Time.now().to_sec(),
                                        "source": "voice_activation"
                                    })
                            
                            # Reset for next recording
                            recording = False
                            frames = []
                            silence_count = 0
            
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
        except Exception as e:
            rospy.logerr(f"Error in voice activation: {e}")
        finally:
            stream.stop_stream()
            stream.close()
    
    def save_audio_frames(self, frames):
        """Save audio frames to temporary file"""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        wf = wave.open(temp_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        os.close(temp_fd)
        return temp_path
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            return result['text'].strip()
        except Exception as e:
            rospy.logerr(f"Whisper transcription error: {e}")
            return ""
    
    def scan_callback(self, msg: LaserScan):
        """Update laser scan data"""
        self.laser_data = msg
    
    def image_callback(self, msg: Image):
        """Update image data"""
        self.image_data = msg
    
    def navigation_status_callback(self, msg: GoalStatusArray):
        """Update navigation status"""
        if msg.status_list:
            # Most recent goal status
            last_status = msg.status_list[-1]
            if last_status.status == 3:  # Succeeded
                self.navigation_active = False
                self.publish_feedback("Navigation completed successfully")
            elif last_status.status in [1, 2, 4, 5, 8]:  # Active/Aborted/etc.
                self.navigation_active = True
    
    def action_status_callback(self, msg: String):
        """Handle action execution status"""
        try:
            status_data = json.loads(msg.data)
            action_id = status_data.get("action_id", "unknown")
            status = status_data.get("status", "unknown")
            
            if status == "failure":
                rospy.logerr(f"Action {action_id} failed: {status_data.get('error', 'Unknown error')}")
                self.publish_feedback(f"Action {action_id} failed: {status_data.get('error', 'Unknown error')}")
            elif status == "success":
                rospy.loginfo(f"Action {action_id} completed successfully")
                self.publish_feedback(f"Action {action_id} completed successfully")
                
        except json.JSONDecodeError:
            rospy.logerr("Invalid JSON in action status message")
    
    def voice_command_callback(self, msg: String):
        """Process incoming voice command"""
        command_text = msg.data
        rospy.loginfo(f"Received voice command: {command_text}")
        
        command_item = {
            "text": command_text,
            "timestamp": rospy.Time.now().to_sec(),
            "source": "external"
        }
        
        self.command_queue.put(command_item)
        self.publish_status(f"Command received: {command_text[:50]}...")
    
    def process_commands(self):
        """Process commands from the queue"""
        while not rospy.is_shutdown():
            try:
                command_item = self.command_queue.get(timeout=1.0)
                
                success = self.process_command_item(command_item)
                
                if not success:
                    rospy.logerr(f"Command processing failed: {command_item['text']}")
                    self.publish_feedback(f"Command failed: {command_item['text'][:50]}...")
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Error processing command queue: {e}")
                continue
    
    def process_command_item(self, command_item: Dict) -> bool:
        """Process a single command item through the full pipeline"""
        command_text = command_item["text"]
        rospy.loginfo(f"Processing command: {command_text}")
        
        # Get environment context
        context = self.get_environment_context()
        
        # Generate plan using GPT-4o
        plan = self.generate_action_plan(command_text, context)
        
        if not plan:
            rospy.logerr("Failed to generate action plan")
            return False
        
        rospy.loginfo(f"Generated plan with {len(plan.get('actions', []))} actions")
        
        # Validate plan safety
        if not self.validate_plan_safety(plan):
            rospy.logerr("Plan failed safety validation")
            self.publish_feedback("Plan rejected: Failed safety validation")
            return False
        
        # Execute plan
        execution_success = self.execute_plan(plan)
        
        if execution_success:
            rospy.loginfo("Command executed successfully")
            self.publish_feedback(f"Command completed: {command_text[:50]}...")
            return True
        else:
            rospy.logerr("Command execution failed")
            self.publish_feedback(f"Command failed: {command_text[:50]}...")
            return False
    
    def get_environment_context(self) -> Dict:
        """Get current environment context"""
        context = {
            "robot_state": {
                "position": self.robot_position,
                "battery_level": 0.85,
                "navigation_status": "ready" if not self.navigation_active else "executing"
            },
            "environment": {
                "obstacles": self.get_obstacle_information(),
                "known_locations": self.get_known_locations(),
                "detected_objects": self.get_detected_objects(),
            },
            "constraints": {
                "safe_distance": self.safe_distance,
                "max_linear_speed": self.max_linear_speed,
                "max_angular_speed": self.max_angular_speed
            }
        }
        return context
    
    def get_obstacle_information(self) -> List[Dict]:
        """Extract obstacle information from laser scan"""
        if not self.laser_data:
            return []
        
        obstacles = []
        ranges = self.laser_data.ranges
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment
        
        # Sample every 20th point to reduce computation
        for i in range(0, len(ranges), 20):
            if not (np.isinf(ranges[i]) or np.isnan(ranges[i])):
                angle = angle_min + i * angle_increment
                x = ranges[i] * np.cos(angle)
                y = ranges[i] * np.sin(angle)
                
                if ranges[i] < self.safe_distance * 2:
                    obstacles.append({
                        "x": float(x),
                        "y": float(y),
                        "distance": float(ranges[i]),
                        "angle": float(angle)
                    })
        
        return obstacles
    
    def get_known_locations(self) -> List[Dict]:
        """Get known locations in the environment"""
        # In real implementation, this would come from map server
        return [
            {"name": "kitchen", "x": 2.0, "y": 1.0},
            {"name": "living_room", "x": 1.0, "y": -1.0},
            {"name": "bedroom", "x": -1.0, "y": 1.5},
            {"name": "office", "y": 0.5, "y": 2.0},
            {"name": "charging_station", "x": -2.0, "y": -2.0}
        ]
    
    def get_detected_objects(self) -> List[Dict]:
        """Get detected objects from perception system"""
        # In real implementation, this would come from object detection
        if self.image_data:
            # Simulated object detection from image
            return [
                {"name": "red_cup", "type": "container", "distance": 1.2, "location": "kitchen", "confidence": 0.9},
                {"name": "blue_book", "type": "stationery", "distance": 1.8, "location": "office", "confidence": 0.8}
            ]
        return []
    
    def generate_action_plan(self, command: str, context: Dict) -> Optional[Dict]:
        """Generate action plan using GPT-4o"""
        prompt = f"""
        Command: "{command}"
        
        Environment Context:
        {json.dumps(context, indent=2)}
        
        Available Actions:
        - navigate_to(location_name): Move robot to named location
        - approach_object(object_name): Move robot near specific object
        - inspect_object(object_name): Examine object with sensors
        - grasp_object(object_name): Grasp an object (if equipped with manipulator)
        - place_object(object_name, location): Place object at location
        - open_container(container_name): Open a container/door
        - close_container(container_name): Close a container/door
        - follow_person(person_name): Follow a person
        - wait(duration_seconds): Pause for specified duration
        - speak_response(text): Speak a response to user
        - find_person(person_name): Locate a specific person
        - escort_person(person_name, destination): Guide person to location
        - patrol_area(area_name): Move through predefined area
        - charge_robot(): Return to charging station
        
        Generate a detailed action plan that:
        1. Breaks down the command into specific, executable actions
        2. Considers the environment context
        3. Respects robot capabilities and constraints
        4. Includes safety checks before movement
        5. Handles potential failure modes
        6. Verifies completion of each step
        7. Includes error recovery strategies
        
        Return as JSON:
        {{
          "original_command": "{command}",
          "reasoning": "step-by-step reasoning about the plan",
          "actions": [
            {{
              "id": 1,
              "type": "action_type",
              "parameters": {{"param1": "value1", "param2": "value2"}},
              "description": "what this step accomplishes",
              "preconditions": ["condition1", "condition2"],
              "expected_effects": ["effect1", "effect2"],
              "safety_check_needed": true,
              "estimated_duration": 15.0,
              "success_probability": 0.85
            }}
          ],
          "estimated_total_duration": 120.0,
          "overall_confidence": 0.75,
          "failure_recovery": [
            {{"condition": "object_not_found", "action": "search_alternatives"}},
            {{"condition": "navigation_failed", "action": "replan_path"}},
            {{"condition": "grasp_failed", "action": "retry_with_different_approach"}}
          ]
        }}
        
        Respond with ONLY the JSON object, no additional text.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a robotic task planner. Generate detailed, executable plans with safety considerations and recovery strategies. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in code block
            if response_text.startswith('```'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx]
            
            plan_data = json.loads(response_text)
            plan_data["timestamp"] = rospy.Time.now().to_sec()
            
            return plan_data
            
        except Exception as e:
            rospy.logerr(f"Error generating action plan: {e}")
            return None
    
    def validate_plan_safety(self, plan: Dict) -> bool:
        """Validate the safety of the generated plan"""
        actions = plan.get("actions", [])
        
        for action in actions:
            action_type = action.get("type", "")
            params = action.get("parameters", {})
            
            # Check navigation safety
            if action_type in ["navigate_to", "approach_object"]:
                if "location" in params:
                    if not self.is_safe_navigation_destination(params["location"]):
                        return False
                elif "object_name" in params:
                    if not self.is_safe_to_approach_object(params["object_name"]):
                        return False
            
            # Check manipulation safety
            elif action_type in ["grasp_object", "place_object"]:
                if not self.is_manipulation_safe():
                    return False
        
        return True
    
    def is_safe_navigation_destination(self, location_name: str) -> bool:
        """Check if navigation destination is safe"""
        known_locations = self.get_known_locations()
        target_loc = next((loc for loc in known_locations if loc["name"] == location_name), None)
        
        if not target_loc:
            rospy.logwarn(f"Unknown location: {location_name}")
            return False
        
        # Check laser scan for obstacles to destination
        if self.laser_data:
            # For simplicity, check if there's a clear path (this would need more sophisticated path planning in reality)
            front_ranges = self.laser_data.ranges[len(self.laser_data.ranges)//2-20:len(self.laser_data.ranges)//2+20]
            valid_ranges = [r for r in front_ranges if not (np.isinf(r) or np.isnan(r))]
            if valid_ranges and min(valid_ranges) < 0.5:
                return False  # Obstacle in front path
        
        return True
    
    def is_safe_to_approach_object(self, object_name: str) -> bool:
        """Check if it's safe to approach an object"""
        detected_objects = self.get_detected_objects()
        target_obj = next((obj for obj in detected_objects if obj["name"] == object_name), None)
        
        if not target_obj:
            rospy.logwarn(f"Object {object_name} not detected")
            return False
        
        # Check if object is at a safe distance
        if target_obj["distance"] > 3.0 or target_obj["distance"] < 0.2:
            rospy.logwarn(f"Object {object_name} is too far ({target_obj['distance']}) or too close")
            return False
        
        return True
    
    def execute_plan(self, plan: Dict) -> bool:
        """Execute the action plan step by step"""
        actions = plan.get("actions", [])
        rospy.loginfo(f"Executing plan with {len(actions)} actions")
        
        for i, action in enumerate(actions):
            rospy.loginfo(f"Executing action {i+1}/{len(actions)}: {action.get('type', 'unknown')}")
            
            self.publish_feedback(f"Executing: {action.get('description', action.get('type'))}")
            
            # Execute the action
            success = self.execute_single_action(action)
            
            if not success:
                rospy.logerr(f"Action execution failed: {action}")
                
                # Try recovery
                if not self.attempt_recovery(plan, i, action):
                    rospy.logerr("Recovery failed, terminating plan execution")
                    return False
            
            # Small delay between actions
            rospy.sleep(0.5)
        
        rospy.loginfo("Plan execution completed successfully")
        return True
    
    def execute_single_action(self, action: Dict) -> bool:
        """Execute a single action in the plan"""
        action_type = action.get("type", "")
        params = action.get("parameters", {})
        
        if action_type == "navigate_to":
            return self.execute_navigation_action(params)
        elif action_type == "approach_object":
            return self.execute_approach_action(params)
        elif action_type == "inspect_object":
            return self.execute_inspection_action(params)
        elif action_type == "grasp_object":
            return self.execute_grasp_action(params)
        elif action_type == "place_object":
            return self.execute_place_action(params)
        elif action_type == "speak_response":
            return self.execute_speak_action(params)
        elif action_type == "wait":
            return self.execute_wait_action(params)
        else:
            rospy.logwarn(f"Unknown action type: {action_type}, skipping...")
            return True  # Don't fail on unknown actions
    
    def execute_navigation_action(self, params: Dict) -> bool:
        """Execute navigation action"""
        location_name = params.get("location")
        
        if not location_name:
            rospy.logerr("Navigation action missing location parameter")
            return False
        
        # Get destination from known locations
        known_locations = self.get_known_locations()
        target_loc = next((loc for loc in known_locations if loc["name"] == location_name), None)
        
        if not target_loc:
            rospy.logerr(f"Unknown location: {location_name}")
            return False
        
        # Create and publish navigation goal
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = target_loc["x"]
        goal.pose.position.y = target_loc["y"]
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0  # No rotation
        
        self.goal_pub.publish(goal)
        rospy.loginfo(f"Navigation goal sent to {location_name} at ({target_loc['x']}, {target_loc['y']})")
        
        # Wait for navigation to complete
        start_time = rospy.Time.now().to_sec()
        timeout = 60.0  # 1-minute timeout
        
        rate = rospy.Rate(10)  # 10 Hz
        while (rospy.Time.now().to_sec() - start_time) < timeout:
            if not self.navigation_active:
                rospy.loginfo(f"Navigation to {location_name} completed")
                return True
            rate.sleep()
        
        rospy.logerr(f"Navigation to {location_name} timed out")
        return False
    
    def execute_approach_action(self, params: Dict) -> bool:
        """Execute object approach action"""
        object_name = params.get("object_name")
        
        if not object_name:
            rospy.logerr("Approach action missing object_name parameter")
            return False
        
        # Find object in detected objects
        detected_objects = self.get_detected_objects()
        target_obj = next((obj for obj in detected_objects if obj["name"] == object_name), None)
        
        if not target_obj:
            rospy.logerr(f"Object {object_name} not detected")
            return False
        
        # Calculate approach pose (1m in front of object)
        approach_x = target_obj["x"]  # Simplified - would need transform computation
        approach_y = target_obj["y"]
        
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = approach_x
        goal.pose.position.y = approach_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal)
        rospy.loginfo(f"Approach goal sent for {object_name}")
        
        # Wait for approach completion
        start_time = rospy.Time.now().to_sec()
        timeout = 45.0  # 45-second timeout for approach
        
        rate = rospy.Rate(10)
        while (rospy.Time.now().to_sec() - start_time) < timeout:
            if not self.navigation_active:
                rospy.loginfo(f"Approach to {object_name} completed")
                return True
            rate.sleep()
        
        rospy.logerr(f"Approach to {object_name} timed out")
        return False
    
    def execute_speak_action(self, params: Dict) -> bool:
        """Execute speech response action"""
        text = params.get("text", "")
        if text:
            rospy.loginfo(f"Speaking response: {text}")
            self.publish_feedback(f"Response: {text}")
            return True
        return False
    
    def execute_wait_action(self, params: Dict) -> bool:
        """Execute wait action"""
        duration = params.get("duration", 1.0)
        rospy.loginfo(f"Waiting for {duration} seconds")
        rospy.sleep(duration)
        return True
    
    def execute_inspection_action(self, params: Dict) -> bool:
        """Execute object inspection action"""
        object_name = params.get("object_name")
        rospy.loginfo(f"Inspecting object: {object_name}")
        
        # In real system, trigger perception routines
        # For simulation, just wait
        rospy.sleep(2.0)
        
        self.publish_feedback(f"Inspection of {object_name} completed")
        return True
    
    def attempt_recovery(self, plan: Dict, failed_step: int, failed_action: Dict) -> bool:
        """Attempt to recover from action failure"""
        recovery_strategies = plan.get("failure_recovery", [])
        failed_condition = f"{failed_action.get('type', 'unknown')}_failed"
        
        # Find recovery strategy
        recovery_strategy = next(
            (rec for rec in recovery_strategies if rec.get("condition") == failed_condition), 
            None
        )
        
        if recovery_strategy:
            rospy.loginfo(f"Attempting recovery: {recovery_strategy['action']}")
            # In a real system, implement actual recovery actions
            return True  # Simulate successful recovery
        else:
            rospy.logwarn(f"No recovery strategy for: {failed_condition}")
            return False
    
    def publish_status(self, message: str):
        """Publish system status"""
        status_msg = String()
        status_msg.data = json.dumps({
            "message": message,
            "timestamp": rospy.Time.now().to_sec()
        })
        self.system_status_pub.publish(status_msg)
    
    def publish_feedback(self, message: str):
        """Publish action feedback"""
        feedback_msg = String()
        feedback_msg.data = json.dumps({
            "message": message,
            "timestamp": rospy.Time.now().to_sec()
        })
        self.voice_response_pub.publish(feedback_msg)

   def main():
       api_key = rospy.get_param('~openai_api_key', '')
       if not api_key:
           api_key = input("Enter OpenAI API key: ").strip()
       
       if not api_key:
           rospy.logerr("No OpenAI API key provided!")
           return
       
       # Initialize system
       system = CompleteVoiceToActionSystem(api_key)
       
       # Start voice activation if enabled
       system.start_voice_activation()
       
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down Complete Voice-to-Action System")

   if __name__ == '__main__':
       main()
   ```

### Running the Complete System

1. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select robot_voice_control
   source install/setup.bash
   ```

2. **Run the complete system**:
   ```bash
   ros2 launch robot_voice_control voice_to_action_system.launch.py openai_api_key:="<your-api-key>"
   ```

3. **Send voice commands** (in another terminal):
   ```bash
   # Simple navigation command
   ros2 topic pub /voice_command std_msgs/String "data: 'Go to kitchen'"
   
   # Complex command
   ros2 topic pub /voice_command std_msgs/String "data: 'Approach the red cup and inspect it'"
   ```

## Mini-project

Create a complete voice-controlled robot system that:

1. Implements voice recognition using Whisper for real-time command processing
2. Integrates with GPT-4o for cognitive task planning
3. Incorporates safety validation at multiple levels (perception, planning, execution)
4. Creates a multimodal interface connecting voice, vision, and action
5. Implements error recovery and graceful degradation mechanisms
6. Evaluates system performance with various input scenarios
7. Documents the complete system architecture and design decisions
8. Creates a user-friendly interface for non-technical operators

Your project should include:
- Complete voice recognition and processing pipeline
- GPT-4o integration for task planning
- Multi-sensor fusion for enhanced perception
- Safety validation throughout the system
- Error handling and recovery mechanisms
- Performance evaluation and testing
- User interface for command input
- Documentation of system design and decisions

## Summary

This chapter covered multi-modal perception fusion in robotics:

- **Multi-modal Integration**: Combining information from different sensor modalities
- **Fusion Architectures**: Early, late, and deep fusion approaches
- **Transformer-Based Fusion**: Using attention mechanisms for cross-modal processing
- **Sensor Calibration**: Ensuring proper alignment between modalities
- **Synchronization**: Managing temporal alignment of sensor data
- **Fusion Algorithms**: Techniques for combining multi-modal information
- **Uncertainty Management**: Handling uncertainty in fused perceptions
- **Performance Evaluation**: Metrics for assessing fusion effectiveness

Multi-modal perception fusion enables robots to develop a more comprehensive understanding of their environment by combining complementary information from multiple sensors, resulting in more robust and accurate perception systems that can handle challenging conditions.