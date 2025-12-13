-----
title: Ch20  Capstone  Autonomous Humanoid (Voice → Plan → Navigate → Manipulate)
module: 4
chapter: 20
sidebar_label: Ch20: Capstone  Autonomous Humanoid
description: Capstone project integrating all modules for voicecontrolled autonomous humanoid robot
tags: [capstone, humanoid, autonomous, voicecontrol, navigation, manipulation, integration]
difficulty: advanced
estimated_duration: 180
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Capstone  Autonomous Humanoid (Voice → Plan → Navigate → Manipulate)

## Learning Outcomes
 Integrate all previous modules (ROS 2, Digital Twins, Isaac Platform, VisionLanguageAction)
 Implement a complete voicecontrolled humanoid robot system
 Create an endtoend pipeline from voice input to physical action
 Demonstrate advanced capabilities: voice recognition, cognitive planning, navigation, and manipulation
 Integrate perception, planning, and control systems
 Evaluate the complete autonomous humanoid system
 Document performance and limitations of the integrated system

## Theory

### Complete Humanoid System Architecture

The autonomous humanoid system integrates all the modules learned throughout the course into a cohesive architecture:

<MermaidDiagram chart={`
graph TB;
    A[Voice Command] > B[ASR];
    B > C[NLP];
    C > D[Task Planning];
    D > E[Navigation];
    D > F[Manipulation];
    E > G[Path Planning];
    E > H[Local Navigation];
    F > I[Grasp Planning];
    F > J[Arm Control];
    
    K[Perception System] > L[Computer Vision];
    K > M[LiDAR Processing];
    K > N[IMU Integration];
    K > O[Sensor Fusion];
    
    L > P[Object Detection];
    M > Q[Environment Mapping];
    N > R[State Estimation];
    O > S[Scene Understanding];
    
    P > D;
    Q > E;
    R > E;
    S > D;
    
    G > H;
    I > J;
    H > T[Humanoid Control];
    J > T;
    T > U[Humanoid Robot];
    
    V[Real World] > U;
    U > W[Sensors];
    W > K;
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style D fill:#2196F3,stroke:#0D47A1,color:#fff;
    style T fill:#FF9800,stroke:#E65100,color:#fff;
    style U fill:#E91E63,stroke:#AD1457,color:#fff;
`} />

### VoicetoAction Pipeline

The system processes user commands through multiple stages:
1. **Automatic Speech Recognition**: Converts voice to text
2. **Natural Language Processing**: Interprets user intent
3. **Task Planning**: Decomposes highlevel goals into actions
4. **Perception Processing**: Understands the environment
5. **Action Execution**: Controls the physical robot

### Integration Challenges

Key challenges in integrating all modules:
 **Timing Coordination**: Ensuring all subsystems operate in harmony
 **Data Consistency**: Maintaining consistent coordinate frames and time stamps
 **Error Propagation**: Managing how errors in one module affect others
 **Resource Management**: Efficiently allocating computational resources across modules

## StepbyStep Labs

### Lab 1: Setting up the Complete Humanoid System

1. **Create the main system orchestration** (`humanoid_system_orchestrator.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import numpy as np
   import openai
   import whisper
   import torch
   import torch.nn as nn
   import json
   import threading
   import queue
   from typing import Dict, List, Optional, Any
   from std_msgs.msg import String, Bool, Float32MultiArray
   from geometry_msgs.msg import Pose, Twist, Point
   from sensor_msgs.msg import Image, LaserScan, Imu, JointState
   from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
   from actionlib_msgs.msg import GoalStatusArray
   from humanoid_control_msgs.msg import JointCommand, CartesianCommand
   import py_trees
   import py_trees_ros
   from cv_bridge import CvBridge
   import cv2
   import actionlib

   class HumanoidSystemOrchestrator:
       def __init__(self):
           rospy.init_node('humanoid_system_orchestrator', anonymous=True)
           
           # Initialize components
           self.cv_bridge = CvBridge()
           self.task_queue = queue.Queue()
           self.execution_tree = None
           
           # Humanoid configuration
           self.robot_config = {
               'height': 1.5,  # meters
               'weight': 30,   # kg
               'max_linear_speed': 0.3,  # m/s
               'max_angular_speed': 0.5,  # rad/s
               'arm_dof': 6,
               'leg_dof': 6,
               'joint_limits': {},  # Will be populated from URDF
           }
           
           # Publishers and Subscribers
           self.voice_cmd_sub = rospy.Subscriber('/voice_command', String, self.voice_callback)
           self.task_plan_pub = rospy.Publisher('/task_plan', String, queue_size=10)
           self.status_pub = rospy.Publisher('/humanoid_status', String, queue_size=10)
           self.feedback_pub = rospy.Publisher('/humanoid_feedback', String, queue_size=10)
           
           # Perception subscribers
           self.rgb_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
           self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
           self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
           self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
           
           # Control publishers
           self.joint_cmd_pub = rospy.Publisher('/joint_commands', JointCommand, queue_size=10)
           self.cartesian_cmd_pub = rospy.Publisher('/cartesian_commands', CartesianCommand, queue_size=10)
           self.base_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           
           # Action clients
           self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
           self.move_base_client.wait_for_server()
           
           # System state
           self.current_pose = Pose()
           self.joint_positions = {}
           self.perception_data = {}
           self.is_executing = False
           self.current_task = None
           
           # LLM integration
           self.openai_client = None
           self.whisper_model = None
           
           rospy.loginfo("Humanoid System Orchestrator initialized")
       
       def setup_llm_integration(self, api_key: str, whisper_model_size: str = "base"):
           """Initialize LLM and Whisper integration"""
           openai.api_key = api_key
           self.openai_client = openai.OpenAI(api_key=api_key)
           self.whisper_model = whisper.load_model(whisper_model_size)
           rospy.loginfo("LLM integration initialized")
       
       def voice_callback(self, msg: String):
           """Handle incoming voice commands"""
           command_text = msg.data
           rospy.loginfo(f"Received voice command: {command_text}")
           
           # Process command and add to execution queue
           task = self.process_voice_command(command_text)
           if task:
               self.task_queue.put(task)
               rospy.loginfo(f"Queued task: {task['action']}")
       
       def process_voice_command(self, command: str) > Optional[Dict]:
           """Process voice command with LLM and create task"""
           if not self.openai_client:
               rospy.logerr("LLM not initialized")
               return None
           
           # Get perception context
           context = self.get_perception_context()
           
           # Create prompt for task planning
           prompt = f"""
           User Command: "{command}"
           
           Environment Context: {json.dumps(context, indent=2)}
           
           Available Actions:
            navigate_to(location_name): Move humanoid to named location
            approach_object(object_name): Navigate to an object
            pick_up_object(object_name, grasp_pose): Grasp an object
            place_object(object_name, location): Place object at location
            inspect_object(object_name): Examine an object
            follow_person(person_name): Follow a person
            answer_question(question): Respond to question about environment
            open_container(container_name): Open door/drawer
            close_container(container_name): Close door/drawer
            wave_to(person_name): Wave to a person
            take_posture(posture_name): Change body posture
            speak_response(text): Speak a response
           
           Decompose the user command into specific executable tasks.
           Consider the environment context and robot capabilities.
           
           Return as a JSON object with structure:
           {{
             "action": "action_name",
             "parameters": {{"param1": "value1", "param2": "value2"}},
             "reasoning": "why this action is appropriate",
             "estimated_duration": 60.0,
             "confidence": 0.8
           }}
           
           Respond only with the JSON object.
           """
           
           try:
               response = self.openai_client.chat.completions.create(
                   model="gpt4o",
                   messages=[
                       {
                           "role": "system",
                           "content": "You are a humanoid robot task planner. Generate specific, executable robotic tasks based on user commands and environment context. Respond only with the JSON object as specified."
                       },
                       {
                           "role": "user",
                           "content": prompt
                       }
                   ],
                   temperature=0.1,
                   max_tokens=1000
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON
               if response_text.startswith('```'):
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   response_text = response_text[start_idx:end_idx]
               
               task = json.loads(response_text)
               return task
               
           except Exception as e:
               rospy.logerr(f"Error processing voice command: {e}")
               return None
       
       def get_perception_context(self) > Dict:
           """Get current perception context"""
           return {
               "robot_pose": {
                   "x": self.current_pose.position.x,
                   "y": self.current_pose.position.y,
                   "z": self.current_pose.position.z,
                   "qx": self.current_pose.orientation.x,
                   "qy": self.current_pose.orientation.y,
                   "qz": self.current_pose.orientation.z,
                   "qw": self.current_pose.orientation.w
               },
               "joint_positions": self.joint_positions,
               "known_locations": ["kitchen", "living_room", "bedroom", "office", "entrance"],
               "detected_objects": self.perception_data.get("objects", []),
               "people_present": self.perception_data.get("people", []),
               "robot_capabilities": ["navigation", "manipulation", "grasping", "speech", "posture_control"],
               "current_battery": self.perception_data.get("battery_level", 0.8)
           }
       
       def rgb_callback(self, msg: Image):
           """Process RGB camera data"""
           try:
               cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
               
               # Run object detection and people recognition
               detection_result = self.run_perception_pipeline(cv_image)
               self.perception_data.update(detection_result)
               
           except Exception as e:
               rospy.logerr(f"Error processing RGB image: {e}")
       
       def run_perception_pipeline(self, image):
           """Run perception pipeline (in real implementation, this would use YOLO, etc.)"""
           # In a real implementation, this would use object detection models
           # For this example, we'll simulate detection
           return {
               "objects": [
                   {"name": "bottle", "position": [1.2, 0.5, 0.0], "class": "container"},
                   {"name": "chair", "position": [0.0, 1.0, 0.0], "class": "furniture"}
               ],
               "people": [
                   {"name": "person1", "position": [2.0, 1.0, 0.0]}
               ]
           }
       
       def lidar_callback(self, msg: LaserScan):
           """Process LiDAR data"""
           # Process for navigation planning
           self.perception_data['lidar_ranges'] = list(msg.ranges)
           self.perception_data['lidar_angle_min'] = msg.angle_min
           self.perception_data['lidar_angle_max'] = msg.angle_max
           self.perception_data['lidar_angle_increment'] = msg.angle_increment
       
       def imu_callback(self, msg: Imu):
           """Process IMU data"""
           self.perception_data['imu'] = {
               'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
               'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
               'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
           }
       
       def joint_state_callback(self, msg: JointState):
           """Process joint state data"""
           for i, name in enumerate(msg.name):
               self.joint_positions[name] = msg.position[i]
       
       def execute_task(self, task: Dict):
           """Execute a single task"""
           action = task['action']
           params = task.get('parameters', {})
           
           rospy.loginfo(f"Executing task: {action} with params: {params}")
           
           if action == 'navigate_to':
               return self.execute_navigation_task(params)
           elif action == 'pick_up_object':
               return self.execute_pickup_task(params)
           elif action == 'place_object':
               return self.execute_place_task(params)
           elif action == 'approach_object':
               return self.execute_approach_task(params)
           elif action == 'speak_response':
               return self.execute_speech_task(params)
           elif action == 'inspect_object':
               return self.execute_inspection_task(params)
           else:
               rospy.logwarn(f"Unknown action: {action}")
               return False
       
       def execute_navigation_task(self, params: Dict) > bool:
           """Execute navigation task"""
           location = params.get('location_name')
           
           # In a real implementation, this would look up location coordinates
           # For now, we'll use simple coordinates
           location_coords = {
               'kitchen': (2.0, 1.0),
               'living_room': (0.0, 0.0),
               'bedroom': (1.0, 2.0),
               'office': (1.0, 1.0),
               'entrance': (0.0, 2.0)
           }
           
           if location not in location_coords:
               rospy.logerr(f"Unknown location: {location}")
               return False
           
           x, y = location_coords[location]
           
           # Create navigation goal
           goal = MoveBaseGoal()
           goal.target_pose.header.frame_id = "map"
           goal.target_pose.header.stamp = rospy.Time.now()
           goal.target_pose.pose.position.x = x
           goal.target_pose.pose.position.y = y
           goal.target_pose.pose.position.z = 0.0
           goal.target_pose.pose.orientation.w = 1.0
           
           # Send goal
           self.move_base_client.send_goal(goal)
           
           # Wait for result
           finished_within_time = self.move_base_client.wait_for_result(rospy.Duration(180))  # 3 minutes
           
           if not finished_within_time:
               self.move_base_client.cancel_goal()
               rospy.logerr("Navigation took too long")
               return False
           
           state = self.move_base_client.get_state()
           if state == 3:  # SUCCEEDED
               rospy.loginfo(f"Successfully navigated to {location}")
               return True
           else:
               rospy.logerr(f"Navigation failed with state: {state}")
               return False
       
       def execute_approach_task(self, params: Dict) > bool:
           """Execute approach object task"""
           object_name = params.get('object_name')
           
           # Find object in perception data
           target_object = None
           for obj in self.perception_data.get('objects', []):
               if obj['name'] == object_name:
                   target_object = obj
                   break
           
           if not target_object:
               rospy.logerr(f"Object {object_name} not found")
               return False
           
           # Navigate to object position (with safe distance)
           obj_pos = target_object['position']
           goal = MoveBaseGoal()
           goal.target_pose.header.frame_id = "map"
           goal.target_pose.header.stamp = rospy.Time.now()
           goal.target_pose.pose.position.x = obj_pos[0]  0.5  # 0.5m in front of object
           goal.target_pose.pose.position.y = obj_pos[1]
           goal.target_pose.pose.position.z = 0.0
           goal.target_pose.pose.orientation.w = 1.0
           
           self.move_base_client.send_goal(goal)
           finished = self.move_base_client.wait_for_result(rospy.Duration(60))
           
           if finished:
               state = self.move_base_client.get_state()
               if state == 3:  # SUCCEEDED
                   rospy.loginfo(f"Approached {object_name}")
                   return True
           
           rospy.logerr(f"Failed to approach {object_name}")
           return False
       
       def execute_pickup_task(self, params: Dict) > bool:
           """Execute pickup object task"""
           # In a real implementation, this would involve complex manipulation planning
           # For now, we'll simulate the action
           object_name = params.get('object_name')
           rospy.loginfo(f"Picked up {object_name}")
           return True
       
       def execute_place_task(self, params: Dict) > bool:
           """Execute place object task"""
           object_name = params.get('object_name')
           location = params.get('location')
           rospy.loginfo(f"Placed {object_name} at {location}")
           return True
       
       def execute_speech_task(self, params: Dict) > bool:
           """Execute speech task"""
           text = params.get('text', '')
           rospy.loginfo(f"Speaking: {text}")
           # In real implementation, this would use texttospeech
           return True
       
       def execute_inspection_task(self, params: Dict) > bool:
           """Execute inspection task"""
           object_name = params.get('object_name')
           rospy.loginfo(f"Inspected {object_name}")
           return True
       
       def run_execution_loop(self):
           """Main execution loop"""
           rate = rospy.Rate(10)  # 10 Hz
           
           while not rospy.is_shutdown():
               try:
                   # Check for tasks in queue
                   while not self.task_queue.empty():
                       task = self.task_queue.get_nowait()
                       
                       if task:
                           self.current_task = task
                           rospy.loginfo(f"Starting execution of task: {task['action']}")
                           
                           success = self.execute_task(task)
                           
                           if success:
                               rospy.loginfo(f"Successfully completed task: {task['action']}")
                               feedback = {
                                   'status': 'success',
                                   'task': task['action'],
                                   'timestamp': rospy.Time.now().to_sec()
                               }
                           else:
                               rospy.logerr(f"Failed to execute task: {task['action']}")
                               feedback = {
                                   'status': 'failure', 
                                   'task': task['action'],
                                   'timestamp': rospy.Time.now().to_sec(),
                                   'error': 'Execution failed'
                               }
                           
                           # Publish feedback
                           feedback_msg = String()
                           feedback_msg.data = json.dumps(feedback)
                           self.feedback_pub.publish(feedback_msg)
                           
                           # Update current task
                           self.current_task = None
                   
                   rate.sleep()
                   
               except queue.Empty:
                   # No tasks in queue, continue loop
                   pass
               except Exception as e:
                   rospy.logerr(f"Error in execution loop: {e}")

   # Example usage
   def main():
       orchestrator = HumanoidSystemOrchestrator()
       
       # Setup API key
       api_key = input("Enter OpenAI API key: ")
       if api_key:
           orchestrator.setup_llm_integration(api_key)
       
       try:
           orchestrator.run_execution_loop()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down humanoid system orchestrator...")

   if __name__ == '__main__':
       main()
   ```

### Lab 2: Creating the VoicetoAction Pipeline

1. **Implement the complete voice processing pipeline** (`voice_to_action_pipeline.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import whisper
   import openai
   import torch
   import json
   import numpy as np
   import pyaudio
   import wave
   import queue
   import threading
   import time
   from std_msgs.msg import String, Bool
   from sensor_msgs.msg import Image
   from geometry_msgs.msg import Pose
   from typing import Optional, Dict, Any
   from cv_bridge import CvBridge

   class VoiceToActionPipeline:
       def __init__(self, api_key: str, whisper_model_size: str = "base"):
           rospy.init_node('voice_to_action_pipeline', anonymous=True)
           
           # Initialize Whisper model
           self.whisper_model = whisper.load_model(whisper_model_size)
           self.openai_client = openai.OpenAI(api_key=api_key)
           
           # Initialize components
           self.cv_bridge = CvBridge()
           self.command_queue = queue.Queue()
           
           # Audio parameters
           self.rate = 16000  # Whisper expects 16kHz
           self.chunk = 1024
           self.format = pyaudio.paInt16
           self.channels = 1
           self.audio = pyaudio.PyAudio()
           
           # Publishers and Subscribers
           self.voice_cmd_pub = rospy.Publisher('/voice_command', String, queue_size=10)
           self.status_pub = rospy.Publisher('/voice_pipeline_status', String, queue_size=10)
           self.perception_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.perception_callback)
           rospy.Subscriber('/toggle_voice_control', Bool, self.toggle_callback)
           
           # System state
           self.is_listening = False
           self.perception_data = {}
           self.listening_thread = None
           self.processing_thread = None
           self.context_memory = []
           self.max_context_items = 20
           
           rospy.loginfo("VoicetoAction Pipeline initialized")
       
       def toggle_callback(self, msg: Bool):
           """Toggle voice control on/off"""
           if msg.data:
               self.start_listening()
           else:
               self.stop_listening()
       
       def perception_callback(self, msg):
           """Process perception data for context"""
           try:
               cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
               # Simple object detection simulation
               h, w, _ = cv_image.shape
               # Simulate detecting an object at center
               if np.mean(cv_image[h//250:h//2+50, w//250:w//2+50]) > 100:  # Simple threshold
                   self.perception_data['objects'] = [{'name': 'object', 'position': 'center'}]
           except:
               pass
       
       def start_listening(self):
           """Start voice recognition"""
           if self.is_listening:
               return
           
           self.is_listening = True
           
           # Start audio recording thread
           self.listening_thread = threading.Thread(target=self._record_audio_continuously)
           self.listening_thread.daemon = True
           self.listening_thread.start()
           
           # Start processing thread
           self.processing_thread = threading.Thread(target=self._process_commands)
           self.processing_thread.daemon = True
           self.processing_thread.start()
           
           rospy.loginfo("VoicetoAction Pipeline started")
           self.status_pub.publish(String(data="Voice pipeline active"))
       
       def stop_listening(self):
           """Stop voice recognition"""
           self.is_listening = False
           rospy.loginfo("VoicetoAction Pipeline stopped")
           self.status_pub.publish(String(data="Voice pipeline inactive"))
       
       def _record_audio_continuously(self):
           """Continuously record audio and detect speech"""
           stream = self.audio.open(
               format=self.format,
               channels=self.channels,
               rate=self.rate,
               input=True,
               frames_per_buffer=self.chunk
           )
           
           recording = False
           frames = []
           voice_threshold = 0.01
           silence_duration = 0
           voice_duration = 0
           min_voice_duration = 0.5  # seconds
           min_silence_duration = 1.0  # seconds
           max_recording_duration = 10.0  # seconds
           
           try:
               while self.is_listening:
                   data = stream.read(self.chunk, exception_on_overflow=False)
                   
                   # Convert to numpy for analysis
                   audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                   rms = np.sqrt(np.mean(audio_array ** 2))
                   
                   if rms > voice_threshold:
                       if not recording:
                           # Start recording
                           recording = True
                           frames = [data]
                           silence_duration = 0
                           voice_duration = 0
                       else:
                           # Continue recording
                           frames.append(data)
                           voice_duration += self.chunk / self.rate
                           silence_duration = 0
                   else:
                       if recording:
                           # Accumulate silence
                           frames.append(data)  # Keep in buffer (might be trailing speech)
                           silence_duration += self.chunk / self.rate
                           voice_duration += self.chunk / self.rate
                           
                           # Check if we have enough speech and silence to trigger transcription
                           if (silence_duration > min_silence_duration and 
                               voice_duration > min_voice_duration):
                               # Finished speaking
                               if len(frames) > 0:
                                   # Save to temp file
                                   temp_file = self._save_frames_to_wav(frames)
                                   self.command_queue.put(temp_file)
                                   
                                   rospy.loginfo(f"Recorded speech segment, duration: {voice_duration:.2f}s")
                                   
                                   # Reset for next recording
                                   recording = False
                                   frames = []
                                   silence_duration = 0
                                   voice_duration = 0
                           elif voice_duration > max_recording_duration:
                               # Max duration reached, transcribe what we have
                               if len(frames) > 0:
                                   temp_file = self._save_frames_to_wav(frames)
                                   self.command_queue.put(temp_file)
                                   
                                   rospy.loginfo(f"Max recording duration reached, transcribing: {voice_duration:.2f}s")
                                   
                                   # Reset for next recording
                                   recording = False
                                   frames = []
                                   silence_duration = 0
                                   voice_duration = 0
           except Exception as e:
               rospy.logerr(f"Error in audio recording: {e}")
           finally:
               stream.stop_stream()
               stream.close()
       
       def _save_frames_to_wav(self, frames):
           """Save audio frames to a temporary WAV file"""
           import tempfile
           import os
           
           temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
           wf = wave.open(temp_path, 'wb')
           wf.setnchannels(self.channels)
           wf.setsampwidth(self.audio.get_sample_size(self.format))
           wf.setframerate(self.rate)
           wf.writeframes(b''.join(frames))
           wf.close()
           os.close(temp_fd)
           
           return temp_path
       
       def _process_commands(self):
           """Process transcribed commands"""
           while self.is_listening or not self.command_queue.empty():
               try:
                   audio_file = self.command_queue.get(timeout=1.0)
                   
                   # Transcribe audio
                   result = self.whisper_model.transcribe(audio_file)
                   transcription = result['text'].strip()
                   
                   if transcription:
                       rospy.loginfo(f"Transcribed: {transcription}")
                       
                       # Enhance with context
                       contextual_command = self._enhance_with_context(transcription)
                       
                       # Publish command
                       cmd_msg = String()
                       cmd_msg.data = contextual_command
                       self.voice_cmd_pub.publish(cmd_msg)
                       
                       # Update context memory
                       self._update_context(transcription, contextual_command)
                       
                       rospy.loginfo(f"Published contextual command: {contextual_command}")
                   else:
                       rospy.loginfo("Empty transcription")
                   
                   # Cleanup temp file
                   if os.path.exists(audio_file):
                       os.remove(audio_file)
                   
                   self.command_queue.task_done()
                   
               except queue.Empty:
                   continue
               except Exception as e:
                   rospy.logerr(f"Error processing command: {e}")
       
       def _enhance_with_context(self, transcription: str) > str:
           """Enhance transcription with environmental context"""
           # Get recent context
           recent_context = self._get_recent_context()
           
           prompt = f"""
           Original Voice Command: "{transcription}"
           
           Environmental Context: {recent_context}
           
           Enhance the original command with environmental context. 
           If the original command refers to objects or locations that can be clarified, 
           make those references more specific based on the context.
           If the command is ambiguous, use the context to disambiguate.
           If the command is clear as is, return it unchanged.
           
           Return only the enhanced command, nothing else.
           """
           
           try:
               response = self.openai_client.chat.completions.create(
                   model="gpt4o",
                   messages=[
                       {
                           "role": "system",
                           "content": "Enhance voice commands with environmental context. Return only the enhanced command text."
                       },
                       {
                           "role": "user",
                           "content": prompt
                       }
                   ],
                   temperature=0.1,
                   max_tokens=200
               )
               
               enhanced_command = response.choices[0].message.content.strip()
               return enhanced_command
               
           except Exception as e:
               rospy.logerr(f"Error enhancing command with context: {e}")
               return transcription  # Return original if enhancement fails
       
       def _get_recent_context(self) > str:
           """Get recent context from memory"""
           if not self.context_memory:
               return "No recent interactions. Environment has common household objects and locations."
           
           recent_items = self.context_memory[5:]  # Last 5 interactions
           context_str = "Recent interactions: "
           context_str += "; ".join([item['original'] for item in recent_items])
           context_str += f". Currently perceived objects: {self.perception_data.get('objects', [])}"
           
           return context_str
       
       def _update_context(self, original: str, enhanced: str):
           """Update context memory with new interaction"""
           context_item = {
               'timestamp': time.time(),
               'original': original,
               'enhanced': enhanced
           }
           
           self.context_memory.append(context_item)
           
           # Keep only recent items
           if len(self.context_memory) > self.max_context_items:
               self.context_memory = self.context_memory[self.max_context_items:]

   def main():
       api_key = input("Enter OpenAI API key: ")
       if not api_key:
           rospy.logerr("No API key provided")
           return
       
       pipeline = VoiceToActionPipeline(api_key)
       
       try:
           pipeline.start_listening()
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down voicetoaction pipeline...")
           pipeline.stop_listening()

   if __name__ == '__main__':
       main()
   ```

### Lab 3: Implementing the Complete Navigation and Manipulation System

1. **Create the navigation and manipulation controller** (`nav_manip_controller.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import numpy as np
   import py_trees
   import py_trees_ros
   import actionlib
   import threading
   from std_msgs.msg import String
   from geometry_msgs.msg import Pose, PoseStamped, Twist, Point
   from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
   from actionlib_msgs.msg import GoalStatusArray
   from humanoid_control_msgs.msg import JointCommand, CartesianCommand
   from sensor_msgs.msg import LaserScan, Image, JointState
   from moveit_commander import MoveGroupCommander, PlanningSceneInterface
   from moveit_msgs.msg import CollisionObject
   from shape_msgs.msg import SolidPrimitive
   from visualization_msgs.msg import Marker
   from tf.transformations import quaternion_from_euler, euler_from_quaternion
   import tf2_ros
   import tf2_geometry_msgs
   import math

   class NavigationManipulationController:
       def __init__(self):
           rospy.init_node('nav_manip_controller', anonymous=True)
           
           # Initialize MoveIt! interface
           self.robot_commander = MoveGroupCommander("arm")
           self.scene = PlanningSceneInterface()
           self.tf_buffer = tf2_ros.Buffer()
           self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
           
           # Action clients
           self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
           self.move_base_client.wait_for_server()
           
           # Publishers and Subscribers
           self.nav_task_sub = rospy.Subscriber('/navigation_task', String, self.navigation_callback)
           self.manip_task_sub = rospy.Subscriber('/manipulation_task', String, self.manipulation_callback)
           self.status_pub = rospy.Publisher('/nav_manip_status', String, queue_size=10)
           self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
           
           # State
           self.current_pose = Pose()
           self.joint_states = {}
           self.laser_data = None
           self.is_navigating = False
           self.is_manipulating = False
           
           # Robot parameters
           self.arm_group_name = "arm"
           self.gripper_group_name = "gripper"
           
           rospy.loginfo("Navigation and Manipulation Controller initialized")
       
       def navigation_callback(self, msg: String):
           """Handle navigation tasks"""
           try:
               task_data = json.loads(msg.data)
               task_type = task_data.get('type')
               
               if task_type == 'move_to_location':
                   location = task_data.get('location')
                   success = self.move_to_location(location)
                   self.report_status(f"Move to {location}: {'Success' if success else 'Failed'}")
               elif task_type == 'approach_object':
                   object_name = task_data.get('object_name')
                   success = self.approach_object(object_name)
                   self.report_status(f"Approach {object_name}: {'Success' if success else 'Failed'}")
           except json.JSONDecodeError:
               rospy.logerr(f"Invalid navigation task data: {msg.data}")
       
       def manipulation_callback(self, msg: String):
           """Handle manipulation tasks"""
           try:
               task_data = json.loads(msg.data)
               task_type = task_data.get('type')
               
               if task_type == 'pick_object':
                   object_pose = task_data.get('pose')
                   success = self.pick_object(object_pose)
                   self.report_status(f"Pick object: {'Success' if success else 'Failed'}")
               elif task_type == 'place_object':
                   location_pose = task_data.get('pose')
                   success = self.place_object(location_pose)
                   self.report_status(f"Place object: {'Success' if success else 'Failed'}")
           except json.JSONDecodeError:
               rospy.logerr(f"Invalid manipulation task data: {msg.data}")
       
       def move_to_location(self, location_name: str) > bool:
           """Move to named location"""
           # In a real implementation, these would be mapped to coordinates
           location_map = {
               'kitchen': (2.0, 1.0, 0.0),
               'living_room': (0.0, 0.0, 0.0),
               'bedroom': (1.0, 2.0, 0.0),
               'office': (1.0, 1.0, 0.0),
               'dining_room': (2.0, 1.0, 0.0)
           }
           
           if location_name not in location_map:
               rospy.logerr(f"Unknown location: {location_name}")
               return False
           
           x, y, theta = location_map[location_name]
           
           # Create and send navigation goal
           goal = MoveBaseGoal()
           goal.target_pose.header.frame_id = "map"
           goal.target_pose.header.stamp = rospy.Time.now()
           goal.target_pose.pose.position.x = x
           goal.target_pose.pose.position.y = y
           goal.target_pose.pose.position.z = 0.0
           
           # Convert theta to quaternion
           quat = quaternion_from_euler(0, 0, theta)
           goal.target_pose.pose.orientation.x = quat[0]
           goal.target_pose.pose.orientation.y = quat[1]
           goal.target_pose.pose.orientation.z = quat[2]
           goal.target_pose.pose.orientation.w = quat[3]
           
           return self.execute_navigation_goal(goal)
       
       def approach_object(self, object_name: str) > bool:
           """Approach a named object"""
           # In a real system, this would get object position from perception
           # For this example, we'll use hardcoded positions
           object_positions = {
               'table': (1.5, 0.5, 0.0),
               'chair': (0.5, 1.0, 0.0),
               'bottle': (2.0, 1.0, 0.0),
               'box': (0.5, 1.5, 0.0)
           }
           
           if object_name not in object_positions:
               rospy.logerr(f"Unknown object: {object_name}")
               return False
           
           obj_x, obj_y, _ = object_positions[object_name]
           
           # Calculate a position 1m in front of the object
           current_x, current_y = self.get_current_position()
           direction_to_obj = math.atan2(obj_y  current_y, obj_x  current_x)
           
           # Position 1m away from object facing it
           approach_x = obj_x  1.0 * math.cos(direction_to_obj)
           approach_y = obj_y  1.0 * math.sin(direction_to_obj)
           
           goal = MoveBaseGoal()
           goal.target_pose.header.frame_id = "map"
           goal.target_pose.header.stamp = rospy.Time.now()
           goal.target_pose.pose.position.x = approach_x
           goal.target_pose.pose.position.y = approach_y
           goal.target_pose.pose.position.z = 0.0
           
           # Orient towards the object
           quat = quaternion_from_euler(0, 0, direction_to_obj)
           goal.target_pose.pose.orientation.x = quat[0]
           goal.target_pose.pose.orientation.y = quat[1]
           goal.target_pose.pose.orientation.z = quat[2]
           goal.target_pose.pose.orientation.w = quat[3]
           
           return self.execute_navigation_goal(goal)
       
       def execute_navigation_goal(self, goal: MoveBaseGoal) > bool:
           """Execute navigation goal with monitoring"""
           self.is_navigating = True
           
           # Send goal
           self.move_base_client.send_goal(goal)
           
           # Monitor progress
           rate = rospy.Rate(10)  # 10 Hz
           start_time = rospy.Time.now()
           timeout_duration = rospy.Duration(120)  # 2 minutes timeout
           
           while not rospy.is_shutdown():
               # Check current status
               state = self.move_base_client.get_state()
               
               if state == 3:  # SUCCEEDED
                   self.is_navigating = False
                   rospy.loginfo("Navigation successful")
                   return True
               elif state in [4, 5, 8]:  # ABORTED, REJECTED, LOST
                   self.is_navigating = False
                   rospy.logerr(f"Navigation failed with state: {state}")
                   return False
               
               # Check timeout
               if rospy.Time.now()  start_time > timeout_duration:
                   self.move_base_client.cancel_goal()
                   self.is_navigating = False
                   rospy.logerr("Navigation timeout")
                   return False
               
               rate.sleep()
       
       def pick_object(self, pose: Dict) > bool:
           """Pick up an object at the given pose"""
           self.is_manipulating = True
           
           try:
               # Set target pose
               target_pose = Pose()
               target_pose.position.x = pose['position']['x']
               target_pose.position.y = pose['position']['y']
               target_pose.position.z = pose['position']['z']
               target_pose.orientation.x = pose['orientation']['x']
               target_pose.orientation.y = pose['orientation']['y']
               target_pose.orientation.z = pose['orientation']['z']
               target_pose.orientation.w = pose['orientation']['w']
               
               # Plan motion
               self.robot_commander.set_pose_target(target_pose)
               plan = self.robot_commander.plan()
               
               if plan.joint_trajectory.points:
                   # Execute motion
                   success = self.robot_commander.execute(plan, wait=True)
                   self.is_manipulating = False
                   return success
               else:
                   rospy.logerr("Failed to plan motion to object")
                   self.is_manipulating = False
                   return False
           except Exception as e:
               rospy.logerr(f"Error in pick operation: {e}")
               self.is_manipulating = False
               return False
       
       def place_object(self, pose: Dict) > bool:
           """Place object at the given pose"""
           self.is_manipulating = True
           
           try:
               # Set target pose
               target_pose = Pose()
               target_pose.position.x = pose['position']['x']
               target_pose.position.y = pose['position']['y']
               target_pose.position.z = pose['position']['z']
               target_pose.orientation.x = pose['orientation']['x']
               target_pose.orientation.y = pose['orientation']['y']
               target_pose.orientation.z = pose['orientation']['z']
               target_pose.orientation.w = pose['orientation']['w']
               
               # Plan motion
               self.robot_commander.set_pose_target(target_pose)
               plan = self.robot_commander.plan()
               
               if plan.joint_trajectory.points:
                   # Execute motion
                   success = self.robot_commander.execute(plan, wait=True)
                   self.is_manipulating = False
                   return success
               else:
                   rospy.logerr("Failed to plan motion to placement location")
                   self.is_manipulating = False
                   return False
           except Exception as e:
               rospy.logerr(f"Error in place operation: {e}")
               self.is_manipulating = False
               return False
       
       def get_current_position(self) > Tuple[float, float]:
           """Get robot's current position from TF"""
           try:
               transform = self.tf_buffer.lookup_transform(
                   "map", "base_link", rospy.Time(0), rospy.Duration(1.0))
               
               x = transform.transform.translation.x
               y = transform.transform.translation.y
               
               return x, y
           except Exception as e:
               rospy.logwarn(f"Could not get current position: {e}")
               return 0.0, 0.0  # Default to origin if TF unavailable
       
       def report_status(self, status_msg: str):
           """Report status to monitoring system"""
           status = String()
           status.data = status_msg
           self.status_pub.publish(status)
           rospy.loginfo(status_msg)

   def main():
       controller = NavigationManipulationController()
       
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down navigation and manipulation controller...")

   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's the complete integrated system that combines all components:

```python
#!/usr/bin/env python3
# complete_autonomous_humanoid_system.py

import rospy
import json
import threading
import time
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
import sys

# Import all components
from humanoid_system_orchestrator import HumanoidSystemOrchestrator
from voice_to_action_pipeline import VoiceToActionPipeline
from nav_manip_controller import NavigationManipulationController

class CompleteAutonomousHumanoidSystem:
    """Complete autonomous humanoid system integrating all modules"""
    
    def __init__(self):
        rospy.init_node('complete_autonomous_humanoid', anonymous=True)
        
        # System state
        self.system_active = False
        self.api_key = None
        self.components = {}
        
        # Publishers
        self.system_status_pub = rospy.Publisher('/autonomous_humanoid_status', String, queue_size=10)
        self.user_feedback_pub = rospy.Publisher('/user_feedback', String, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/system_control', String, self.system_control_callback)
        
        rospy.loginfo("Complete Autonomous Humanoid System initialized")
    
    def system_control_callback(self, msg: String):
        """Handle system control commands"""
        command = json.loads(msg.data)
        action = command.get('action')
        
        if action == 'start_system':
            self.start_system()
        elif action == 'stop_system':
            self.stop_system()
        elif action == 'configure_api_key':
            self.api_key = command.get('api_key')
            self.configure_components()
        elif action == 'execute_demo':
            self.execute_demo_scenario()
    
    def configure_components(self):
        """Configure all system components with API key"""
        if not self.api_key:
            rospy.logerr("No API key available for configuration")
            return
        
        # Configure orchestrator
        if 'orchestrator' in self.components:
            self.components['orchestrator'].setup_llm_integration(self.api_key)
        
        # Configure voice pipeline
        if 'voice_pipeline' in self.components:
            # The voice pipeline would need to be recreated or reconfigured
            pass
    
    def start_system(self):
        """Start the complete autonomous humanoid system"""
        if self.system_active:
            rospy.logwarn("System already active")
            return
        
        rospy.loginfo("Starting complete autonomous humanoid system...")
        
        # Initialize API key if not already set
        if not self.api_key:
            self.api_key = input("Enter OpenAI API key: ")
        
        if not self.api_key:
            rospy.logerr("No API key provided, cannot start system")
            return
        
        # Initialize all components
        try:
            # 1. Initialize the orchestrator
            orchestrator = HumanoidSystemOrchestrator()
            orchestrator.setup_llm_integration(self.api_key)
            self.components['orchestrator'] = orchestrator
            
            # 2. Initialize voicetoaction pipeline
            voice_pipeline = VoiceToActionPipeline(self.api_key)
            self.components['voice_pipeline'] = voice_pipeline
            
            # 3. Initialize navmanip controller
            nav_manip_controller = NavigationManipulationController()
            self.components['nav_manip_controller'] = nav_manip_controller
            
            # Start components in separate threads
            self.execution_thread = threading.Thread(
                target=self.run_execution_loop, 
                args=(orchestrator,)
            )
            self.execution_thread.daemon = True
            self.execution_thread.start()
            
            # Start voice pipeline
            voice_thread = threading.Thread(
                target=self.run_voice_pipeline, 
                args=(voice_pipeline,)
            )
            voice_thread.daemon = True
            voice_thread.start()
            
            self.system_active = True
            status_msg = String()
            status_msg.data = "System started successfully"
            self.system_status_pub.publish(status_msg)
            rospy.loginfo("All system components initialized and running")
            
        except Exception as e:
            rospy.logerr(f"Error starting system: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_system(self):
        """Stop the complete system"""
        rospy.loginfo("Stopping complete autonomous humanoid system...")
        
        self.system_active = False
        
        # Stop voice pipeline
        if 'voice_pipeline' in self.components:
            self.components['voice_pipeline'].stop_listening()
        
        status_msg = String()
        status_msg.data = "System stopped"
        self.system_status_pub.publish(status_msg)
    
    def run_execution_loop(self, orchestrator):
        """Run the orchestrator execution loop"""
        try:
            orchestrator.run_execution_loop()
        except Exception as e:
            rospy.logerr(f"Error in execution loop: {e}")
    
    def run_voice_pipeline(self, voice_pipeline):
        """Run the voice pipeline"""
        try:
            voice_pipeline.start_listening()
        except Exception as e:
            rospy.logerr(f"Error in voice pipeline: {e}")
    
    def execute_demo_scenario(self):
        """Execute a demo scenario"""
        if not self.system_active:
            rospy.logerr("System not active, cannot execute demo")
            return
        
        rospy.loginfo("Starting demo scenario...")
        
        # Example sequence of highlevel commands
        demo_commands = [
            "Navigate to the kitchen",
            "Approach the table",
            "Pick up the red cup",
            "Place the cup in the sink",
            "Return to the living room"
        ]
        
        # Publish commands to the system
        voice_cmd_pub = rospy.Publisher('/voice_command', String, queue_size=10)
        
        for i, command in enumerate(demo_commands):
            rospy.loginfo(f"Executing demo command {i+1}/5: {command}")
            
            # Publish command
            cmd_msg = String()
            cmd_msg.data = command
            voice_cmd_pub.publish(cmd_msg)
            
            # Wait between commands
            time.sleep(5)
        
        rospy.loginfo("Demo scenario completed")
        feedback_msg = String()
        feedback_msg.data = "Demo scenario completed"
        self.user_feedback_pub.publish(feedback_msg)
    
    def run(self):
        """Main run loop"""
        rospy.loginfo("Complete Autonomous Humanoid System running...")
        
        # For demo purposes, start the system automatically
        self.start_system()
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down complete autonomous humanoid system...")
            self.stop_system()

def main():
    """Main function to run the complete system"""
    system = CompleteAutonomousHumanoidSystem()
    system.run()

if __name__ == '__main__':
    main()
```

### Launch file for the complete system:

```xml
<launch>
  <! Complete Autonomous Humanoid System >
  <node name="complete_autonomous_humanoid" pkg="humanoid_robot" type="complete_autonomous_humanoid_system.py" output="screen">
  </node>
  
  <! VoicetoAction Pipeline >
  <node name="voice_to_action_pipeline" pkg="humanoid_robot" type="voice_to_action_pipeline.py" output="screen">
  </node>
  
  <! Navigation and Manipulation Controller >
  <node name="nav_manip_controller" pkg="humanoid_robot" type="nav_manip_controller.py" output="screen">
  </node>
  
  <! Perception System >
  <node name="perception_pipeline" pkg="perception" type="perception_pipeline.py" output="screen">
  </node>
  
  <! Robot Hardware Interface >
  <node name="hardware_interface" pkg="ros_control" type="robot_hw_interface.py" output="screen">
  </node>
  
  <! MoveIt! Configuration >
  <include file="$(find my_robot_moveit_config)/launch/move_group.launch"/>
  <include file="$(find my_robot_moveit_config)/launch/moveit_rviz.launch"/>

  <! TF Tree >
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
  
  <! Example sensor drivers >
  <group ns="sensors">
    <node name="camera_driver" pkg="usb_cam" type="usb_cam_node" output="screen">
      <param name="video_device" value="/dev/video0"/>
      <param name="image_width" value="640"/>
      <param name="image_height" value="480"/>
      <param name="pixel_format" value="yuyv"/>
    </node>
    
    <node name="lidar_driver" pkg="velodyne_driver" type="velodyne_node" output="screen">
      <param name="device_ip" value="192.168.1.201"/>
      <param name="port" value="2368"/>
    </node>
    
    <node name="imu_driver" pkg="razor_imu_9dof" type="imu_node" output="screen"/>
  </group>
  
  <! Static transforms for the humanoid >
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0.0 0.0 0.0 0.0 0.0 0.0 base_link torso" />
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0.0 0.2 1.0 0.0 0.0 0.0 torso head" />
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0.3 0.0 0.8 0.0 0.0 0.0 torso left_shoulder" />
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0.3 0.0 0.8 0.0 0.0 0.0 torso right_shoulder" />
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0.0 0.1 0.8 0.0 0.0 0.0 torso pelvis" />
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0.15 0.1 1.0 0.0 0.0 0.0 pelvis left_hip" />
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0.15 0.1 1.0 0.0 0.0 0.0 pelvis right_hip" />
</launch>
```

## Miniproject

Create a complete autonomous humanoid system that:

1. Integrates voice recognition and natural language understanding
2. Implements cognitive task planning with GPT4o
3. Executes navigation and manipulation tasks
4. Incorporates multimodal perception fusion
5. Demonstrates complete voice → plan → navigate → manipulate pipeline
6. Evaluates system performance and robustness
7. Documents the full integration and its challenges
8. Creates a userfriendly interface for command input

Your project should include:
 Complete integration of all four modules
 Working voicetoaction pipeline
 Cognitive planning and task execution
 Navigation and manipulation capabilities
 Multimodal perception system
 Performance evaluation and error handling
 Demo scenarios showing the complete pipeline
 Detailed documentation of integration challenges

## Summary

This chapter served as the capstone project integrating all modules:

 **Module 1 (ROS 2)**: Used for system architecture and communication between components
 **Module 2 (Digital Twins)**: Applied for simulation and validation before realworld deployment
 **Module 3 (Isaac Platform)**: Provided AIpowered perception and control systems
 **Module 4 (VisionLanguageAction)**: Enabled the voicetoaction pipeline

The complete autonomous humanoid system demonstrates:
 Voice recognition and natural language understanding
 Cognitive task planning using LLMs
 Navigation and manipulation capabilities
 Multimodal perception fusion
 Endtoend integration from voice input to physical action

This capstone project showcases the integration of all technologies learned throughout the course, creating a sophisticated autonomous humanoid robot capable of understanding and executing complex voice commands in realworld environments.