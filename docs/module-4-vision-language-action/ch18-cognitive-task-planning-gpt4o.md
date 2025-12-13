-----
title: Ch18  Cognitive Task Planning with GPT4o
module: 4
chapter: 18
sidebar_label: Ch18: Cognitive Task Planning with GPT4o
description: Implementing cognitive task planning using GPT4o for complex robotic manipulation and navigation
tags: [gpt4o, cognitiveplanning, taskplanning, robotics, aiplanning, hierarchicaltasknetwork, ros2]
difficulty: advanced
estimated_duration: 150
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Cognitive Task Planning with GPT4o

## Learning Outcomes
 Implement cognitive task planning using OpenAI's GPT4o model
 Design hierarchical task decomposition for complex robotic tasks
 Integrate LLMbased planning with traditional robotic planning systems
 Create contextaware task planners that understand the environment
 Implement plan execution monitoring and recovery
 Design feedback mechanisms for plan refinement
 Evaluate cognitive planning performance and robustness
 Handle ambiguity and uncertainty in task specifications

## Theory

### Cognitive Task Planning in Robotics

Cognitive task planning involves highlevel reasoning about complex tasks that require understanding of the environment, objects, and their relationships. This contrasts with lowlevel motion planning which deals with trajectory generation and collision avoidance.

<MermaidDiagram chart={`
graph TD;
    A[HighLevel Task] > B[Task Decomposition];
    B > C[Subtask Identification];
    C > D[Primitive Action Sequences];
    D > E[LowLevel Controllers];
    
    F[Environmental Context] > G[Perception System];
    G > H[Object Recognition];
    H > I[Spatial Reasoning];
    
    B > J[Context Integration];
    C > J;
    I > J;
    
    J > K[Plan Generation];
    K > L[Plan Validation];
    L > M[Plan Execution];
    M > N[Monitoring & Recovery];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style K fill:#2196F3,stroke:#0D47A1,color:#fff;
    style M fill:#FF9800,stroke:#E65100,color:#fff;
`} />

### GPT4o for Task Planning

GPT4o provides several advantages for cognitive task planning:

 **World Knowledge**: Understanding of physical objects, their properties, and relationships
 **Logical Reasoning**: Ability to reason about cause and effect, preconditions and effects
 **Contextual Understanding**: Ability to understand the environment and adapt plans accordingly
 **Natural Language Interface**: Easy specification of tasks using natural language

### Hierarchical Task Networks (HTNs)

HTNs break down complex tasks into hierarchically organized subtasks. The planning process involves decomposing highlevel tasks into more specific subtasks until primitive actions are reached.

### Plan Representation

Plans can be represented as:
 **Sequential**: Linear sequence of actions
 **Partial Order**: Actions with temporal constraints
 **Contingent**: Plans with conditional branches based on sensing results
 **Hierarchical**: Tree structure of decomposed tasks

## StepbyStep Labs

### Lab 1: Setting up GPT4o for Task Planning

1. **Install required dependencies**:
   ```bash
   pip install openai==1.3.5
   pip install langchain langchainopenai
   pip install numpy scipy
   pip install gymnasium
   pip install ros2 rospy
   pip install py_trees  # Behavior trees library
   ```

2. **Create a GPT4o task planning interface** (`gpt_task_planner.py`):
   ```python
   #!/usr/bin/env python3

   import openai
   import json
   import rospy
   import py_trees
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Pose
   from actionlib_msgs.msg import GoalStatusArray
   from typing import Dict, List, Optional, Any
   import time
   import yaml

   class GPTTaskPlanner:
       def __init__(self, api_key: str, model_name: str = "gpt4o"):
           openai.api_key = api_key
           self.client = openai.OpenAI(api_key=api_key)
           self.model_name = model_name
           
           # Initialize ROS node
           rospy.init_node('gpt_task_planner', anonymous=True)
           
           # Publishers and Subscribers
           self.plan_pub = rospy.Publisher('/generated_plan', String, queue_size=10)
           self.status_pub = rospy.Publisher('/planner_status', String, queue_size=10)
           self.feedback_pub = rospy.Publisher('/planner_feedback', String, queue_size=10)
           rospy.Subscriber('/task_request', String, self.task_request_callback)
           rospy.Subscriber('/execution_feedback', String, self.execution_feedback_callback)
           
           # Environment state
           self.environment_state = {}
           self.current_plan = None
           self.current_execution_step = 0
           
           # Task planning configuration
           self.max_replanning_attempts = 3
           self.plan_validity_timeout = 30  # seconds
           
           rospy.loginfo(f"GPT Task Planner initialized with model: {model_name}")
       
       def task_request_callback(self, msg: String):
           """Handle incoming task requests"""
           rospy.loginfo(f"Received task request: {msg.data}")
           
           task_description = msg.data
           
           # Plan the task
           plan = self.generate_plan(task_description)
           
           if plan:
               self.current_plan = plan
               self.current_execution_step = 0
               self.publish_plan(plan)
               self.publish_status("Plan generated and published")
           else:
               self.publish_status("Failed to generate plan")
       
       def execution_feedback_callback(self, msg: String):
           """Handle execution feedback"""
           try:
               feedback_data = json.loads(msg.data)
               status = feedback_data.get("status", "")
               step = feedback_data.get("step", 0)
               
               if status == "failure":
                   rospy.loginfo(f"Plan step {step} failed, attempting recovery")
                   self.handle_failure(step, feedback_data.get("error", ""))
               elif status == "success":
                   rospy.loginfo(f"Plan step {step} completed successfully")
                   self.current_execution_step += 1
               
           except json.JSONDecodeError:
               rospy.logerr("Failed to parse execution feedback")
       
       def generate_plan(self, task_description: str) > Optional[Dict]:
           """Generate a task plan using GPT4o"""
           # Get current environment context
           context = self.get_environment_context()
           
           # Construct planning prompt
           prompt = self.construct_planning_prompt(task_description, context)
           
           try:
               response = self.client.chat.completions.create(
                   model=self.model_name,
                   messages=[
                       {
                           "role": "system",
                           "content": """You are an expert robotic task planner. Generate detailed, executable plans for robotic tasks. 
                           Your response should be a valid JSON object with the following structure:
                           {
                             "task": "original task description",
                             "reasoning": "detailed reasoning about the plan",
                             "plan": [
                               {
                                 "step": 1,
                                 "description": "what robot should do",
                                 "action": "action_type",
                                 "parameters": {"key": "value"},
                                 "preconditions": ["condition1", "condition2"],
                                 "effects": ["effect1", "effect2"]
                               }
                             ],
                             "estimated_time": 120.0,
                             "confidence": 0.9
                           }
                           
                           Actions should be specific, executable robot commands.
                           Precondition and effect descriptions should be in plain English.
                           Do not include any text before or after the JSON."""
                       },
                       {
                           "role": "user",
                           "content": prompt
                       }
                   ],
                   temperature=0.1,  # Low temperature for consistency
                   max_tokens=2000
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON from response (if wrapped in markdown)
               if response_text.startswith('```'):
                   # Find the first '{' and last '}'
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   if start_idx != 1 and end_idx != 1:
                       response_text = response_text[start_idx:end_idx]
                   else:
                       rospy.logerr("Could not extract JSON from response")
                       return None
               
               plan_data = json.loads(response_text)
               rospy.loginfo(f"Generated plan with {len(plan_data['plan'])} steps")
               return plan_data
               
           except Exception as e:
               rospy.logerr(f"Error generating plan: {e}")
               return None
       
       def construct_planning_prompt(self, task_description: str, context: Dict) > str:
           """Construct prompt for task planning"""
           prompt = f"""
           Task: {task_description}
           
           Environment Context: {json.dumps(context, indent=2)}
           
           Available Actions:
            move_to_location(location_name): Move robot to named location
            pick_object(object_name, grasp_pose): Pick up an object with specific grasp
            place_object(object_name, location, placement_pose): Place object at location
            open_object(object_name): Open a container or door
            close_object(object_name): Close a container or door
            detect_object(object_name): Detect presence of object in environment
            navigate_to_object(object_name): Navigate close to an object
            inspect_object(object_name): Inspect an object for damage or quality
            transport_object(from_location, to_location): Transport an object
           
           Generate a detailed plan to accomplish the task. The plan should be executable and consider:
           1. Preconditions for each step
           2. Expected effects of each action
           3. Potential failure modes
           4. Environmental constraints
           5. Object affordances
           
           The plan should be specific enough for direct robot execution.
           """
           return prompt
       
       def get_environment_context(self) > Dict:
           """Get current environment context"""
           # This would be populated from various ROS topics in a real implementation
           return {
               "known_locations": ["kitchen", "living_room", "bedroom", "office"],
               "robot_pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
               "detected_objects": [
                   {"name": "apple", "location": "kitchen_table", "type": "food"},
                   {"name": "book", "location": "living_room_table", "type": "stationery"},
                   {"name": "water_bottle", "location": "desk", "type": "drink"}
               ],
               "available_tools": ["arm", "grasping_gripper"],
               "current_inventory": [],
               "time_of_day": "daytime"
           }
       
       def publish_plan(self, plan: Dict):
           """Publish the generated plan"""
           plan_msg = String()
           plan_msg.data = json.dumps(plan, indent=2)
           self.plan_pub.publish(plan_msg)
       
       def publish_status(self, status: str):
           """Publish planner status"""
           status_msg = String()
           status_msg.data = status
           self.status_pub.publish(status_msg)
       
       def handle_failure(self, step_num: int, error: str):
           """Handle plan execution failure"""
           rospy.loginfo(f"Handling failure in step {step_num}: {error}")
           
           # Attempt recovery
           recovery_plan = self.generate_recovery_plan(self.current_plan, step_num, error)
           
           if recovery_plan:
               rospy.loginfo("Recovery plan generated, publishing...")
               self.publish_plan(recovery_plan)
               self.publish_status(f"Recovery plan generated for step {step_num}")
           else:
               rospy.logerr(f"Failed to generate recovery plan for step {step_num}")
               self.publish_status(f"Recovery failed for step {step_num}")
       
       def generate_recovery_plan(self, original_plan: Dict, failed_step: int, error: str) > Optional[Dict]:
           """Generate a recovery plan after failure"""
           # Get environment context at the time of failure
           context = self.get_environment_context()
           
           prompt = f"""
           Original Task: {original_plan['task']}
           
           Failed Step: {failed_step}
           Original Step Plan: {json.dumps(original_plan['plan'][failed_step1], indent=2)}
           Error: {error}
           
           Current Environment: {json.dumps(context, indent=2)}
           
           Generate a recovery plan to address the failure and continue task completion.
           The recovery plan should:
           1. Address the specific failure that occurred
           2. Resume task execution from an appropriate point
           3. Consider the current environment state
           4. Be executable by the robot
           
           Return the plan in the same JSON format as before.
           """
           
           try:
               response = self.client.chat.completions.create(
                   model=self.model_name,
                   messages=[
                       {
                           "role": "system",
                           "content": """You are an expert robotic task recovery planner. Generate recovery plans that address specific failures and resume task execution. 
                           Respond with valid JSON as specified in the original planning system."""
                       },
                       {
                           "role": "user",
                           "content": prompt
                       }
                   ],
                   temperature=0.1,
                   max_tokens=1500
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON
               if response_text.startswith('```'):
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   if start_idx != 1 and end_idx != 1:
                       response_text = response_text[start_idx:end_idx]
                   else:
                       rospy.logerr("Could not extract JSON from recovery response")
                       return None
               
               recovery_plan = json.loads(response_text)
               return recovery_plan
               
           except Exception as e:
               rospy.logerr(f"Error generating recovery plan: {e}")
               return None
       
       def run(self):
           """Run the planner node"""
           rospy.loginfo("GPT Task Planner running...")
           rospy.spin()

   if __name__ == '__main__':
       # Initialize with OpenAI API key
       api_key = rospy.get_param("~openai_api_key", "")
       if not api_key:
           api_key = input("Enter OpenAI API key: ")
       
       if not api_key:
           rospy.logerr("No OpenAI API key provided!")
           exit(1)
       
       planner = GPTTaskPlanner(api_key)
       planner.run()
   ```

### Lab 2: Implementing Hierarchical Task Networks

1. **Create a hierarchical task network planner** (`htn_planner.py`):
   ```python
   #!/usr/bin/env python3

   import openai
   import json
   import rospy
   import py_trees
   from std_msgs.msg import String
   from typing import Dict, List, Optional, Union
   import time

   class HTNPlanner:
       def __init__(self, api_key: str, model_name: str = "gpt4o"):
           self.client = openai.OpenAI(api_key=api_key)
           self.model_name = model_name
           
           # Initialize ROS
           rospy.init_node('htn_planner', anonymous=True)
           
           # Publishers and Subscribers
           self.plan_pub = rospy.Publisher('/htn_plan', String, queue_size=10)
           self.status_pub = rospy.Publisher('/htn_status', String, queue_size=10)
           rospy.Subscriber('/htn_task_request', String, self.task_request_callback)
           
           # Task templates for decomposition
           self.task_templates = {
               "move_object": {
                   "subtasks": [
                       {"name": "navigate_to_object", "parameters": ["object_name", "navigation_pose"]},
                       {"name": "grasp_object", "parameters": ["object_name", "grasp_pose"]},
                       {"name": "navigate_to_destination", "parameters": ["destination", "navigation_pose"]},
                       {"name": "release_object", "parameters": ["placement_pose"]}
                   ]
               },
               "assemble_objects": {
                   "subtasks": [
                       {"name": "find_component", "parameters": ["component_type"]},
                       {"name": "transport_component", "parameters": ["component", "assembly_station"]},
                       {"name": "align_component", "parameters": ["component", "assembly_point"]},
                       {"name": "fasten_component", "parameters": ["component", "fastening_method"]}
                   ]
               },
               "inspect_area": {
                   "subtasks": [
                       {"name": "navigate_to_area", "parameters": ["area_name"]},
                       {"name": "perform_visual_inspection", "parameters": ["inspection_targets"]},
                       {"name": "analyze_results", "parameters": ["collected_data"]},
                       {"name": "report_findings", "parameters": ["analysis_results"]}
                   ]
               }
           }
           
           rospy.loginfo("HTN Planner initialized")
       
       def task_request_callback(self, msg: String):
           """Handle task request and generate HTN plan"""
           rospy.loginfo(f"Processing HTN task: {msg.data}")
           
           plan = self.generate_htn_plan(msg.data)
           
           if plan:
               rospy.loginfo(f"Generated HTN plan with {len(plan['tasks'])} tasks")
               plan_msg = String()
               plan_msg.data = json.dumps(plan, indent=2)
               self.plan_pub.publish(plan_msg)
           else:
               rospy.logerr("Failed to generate HTN plan")
       
       def generate_htn_plan(self, task_description: str) > Optional[Dict]:
           """Generate hierarchical task network plan"""
           # Get environment context
           context = self.get_environment_context()
           
           # First, determine the highlevel task type
           task_type = self.identify_task_type(task_description)
           
           if task_type:
               # Decompose into subtasks based on identified task type
               plan = self.decompose_task(task_type, task_description, context)
               return plan
           else:
               # Use GPT4o to figure out the task structure
               return self.generate_plan_with_gpt(task_description, context)
       
       def identify_task_type(self, task_description: str) > Optional[str]:
           """Identify the type of task from description"""
           # Check against known task templates
           for task_template, _ in self.task_templates.items():
               # Simple keywordbased classification
               keywords = {
                   "move_object": ["move", "transport", "carry", "place", "put"],
                   "assemble_objects": ["assemble", "build", "construct", "put together"],
                   "inspect_area": ["inspect", "check", "examine", "survey"]
               }
               
               for keyword in keywords.get(task_template, []):
                   if keyword in task_description.lower():
                       return task_template
           
           return None
       
       def decompose_task(self, task_type: str, task_description: str, context: Dict) > Dict:
           """Decompose task into hierarchical subtasks"""
           if task_type in self.task_templates:
               template = self.task_templates[task_type]
               
               # Generate detailed plan based on template
               plan = {
                   "task": task_description,
                   "task_type": task_type,
                   "tasks": [],
                   "context": context,
                   "timestamp": time.time()
               }
               
               # Instantiate subtasks with appropriate parameters
               for i, subtask_template in enumerate(template["subtasks"]):
                   subtask = self.instantiate_subtask(subtask_template, context, task_description)
                   subtask["id"] = i + 1
                   plan["tasks"].append(subtask)
               
               return plan
           else:
               # Fallback to GPT4o for unknown task types
               return self.generate_plan_with_gpt(task_description, context)
       
       def instantiate_subtask(self, template: Dict, context: Dict, original_task: str) > Dict:
           """Instantiate a subtask template with specific parameters"""
           # Use GPT4o to determine specific parameters based on context
           prompt = f"""
           Task: {original_task}
           
           Environment Context: {json.dumps(context, indent=2)}
           
           Subtask Template: {json.dumps(template, indent=2)}
           
           Generate specific parameters for this subtask based on the environment context.
           Return a JSON object with the subtask name and filledin parameters.
           
           Example:
           {{
             "name": "navigate_to_object",
             "parameters": {{
               "object_name": "red_cup",
               "navigation_pose": {{ "x": 1.5, "y": 2.0, "theta": 0.0 }}
             }},
             "preconditions": ["robot_is_idle", "path_exists"],
             "effects": ["robot_at_object", "object_in_view"]
           }}
           """
           
           try:
               response = self.client.chat.completions.create(
                   model=self.model_name,
                   messages=[
                       {
                           "role": "system",
                           "content": "You are an expert in robot task decomposition. Generate specific subtasks with concrete parameters based on environment context."
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
               
               return json.loads(response_text)
               
           except Exception as e:
               rospy.logerr(f"Error instantiating subtask: {e}")
               # Return a basic template with generic parameters
               return {
                   "name": template["name"],
                   "parameters": {param: f"<{param}>" for param in template.get("parameters", [])},
                   "preconditions": [],
                   "effects": []
               }
       
       def generate_plan_with_gpt(self, task_description: str, context: Dict) > Optional[Dict]:
           """Generate plan using GPT4o for unknown task types"""
           prompt = f"""
           Task: {task_description}
           
           Environment Context: {json.dumps(context, indent=2)}
           
           Available Actions:
            move_to_location(location)
            pick_object(object_name, grasp_type)
            place_object(object_name, location)
            navigate_to_object(object_name)
            inspect_object(object_name)
            open_container(container_name)
            close_container(container_name)
            detect_object(object_name)
            transport_object(from_location, to_location)
            assemble_parts(part1, part2)
            disassemble_part(object)
            charge_robot(at_charging_station)
           
           Generate a hierarchical task network plan for this task.
           The plan should include:
           1. Highlevel task decomposition
           2. Specific subtasks with parameters
           3. Precondition and effect definitions
           4. Temporal dependencies between tasks
           
           Return as a JSON object with 'tasks' array containing subtasks.
           """
           
           try:
               response = self.client.chat.completions.create(
                   model=self.model_name,
                   messages=[
                       {
                           "role": "system",
                           "content": """Generate detailed hierarchical task networks for robotic tasks. 
                           Return valid JSON with a 'tasks' array containing subtasks at different levels of abstraction. 
                           Each task should have name, parameters, preconditions, and effects."""
                       },
                       {
                           "role": "user",
                           "content": prompt
                       }
                   ],
                   temperature=0.1,
                   max_tokens=1500
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON
               if response_text.startswith('```'):
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   response_text = response_text[start_idx:end_idx]
               
               plan_data = json.loads(response_text)
               plan_data["task"] = task_description
               plan_data["task_type"] = "unknown_generated"
               plan_data["context"] = context
               plan_data["timestamp"] = time.time()
               
               return plan_data
               
           except Exception as e:
               rospy.logerr(f"Error generating HTN plan: {e}")
               return None
       
       def get_environment_context(self) > Dict:
           """Get current environment context"""
           # This would be populated from sensors in a real implementation
           return {
               "robot_pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
               "known_locations": [
                   {"name": "kitchen_table", "pose": {"x": 1.0, "y": 1.0, "theta": 0.0}},
                   {"name": "desk", "pose": {"x": 2.0, "y": 0.5, "theta": 0.0}},
                   {"name": "charging_station", "pose": {"x": 0.0, "y": 2.0, "theta": 1.57}}
               ],
               "detected_objects": [
                   {"name": "coffee_mug", "location": "kitchen_table", "properties": {"color": "white", "type": "container"}},
                   {"name": "notebook", "location": "desk", "properties": {"color": "black", "type": "stationery"}}
               ],
               "robot_capabilities": ["navigation", "manipulation", "grasping", "inspection"]
           }

   if __name__ == '__main__':
       api_key = rospy.get_param("~openai_api_key", "")
       if not api_key:
           api_key = input("Enter OpenAI API key: ")
       
       if not api_key:
           rospy.logerr("No OpenAI API key provided!")
           exit(1)
       
       planner = HTNPlanner(api_key)
       planner.run()
   ```

### Lab 3: Integrating with Behavior Trees for Execution

1. **Create a behavior tree execution system** (`behavior_tree_executor.py`):
   ```python
   #!/usr/bin/env python3

   import py_trees
   import py_trees.console as console
   import rospy
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Pose
   from actionlib_msgs.msg import GoalStatusArray
   from typing import Dict, List
   import json
   import time

   class BTExecutionManager:
       def __init__(self):
           rospy.init_node('bt_execution_manager', anonymous=True)
           
           # Publishers and Subscribers
           self.status_pub = rospy.Publisher('/bt_execution_status', String, queue_size=10)
           self.feedback_pub = rospy.Publisher('/execution_feedback', String, queue_size=10)
           rospy.Subscriber('/htn_plan', String, self.plan_callback)
           rospy.Subscriber('/bt_control', String, self.control_callback)
           
           # Execution state
           self.root_behavior_tree = None
           self.current_plan = None
           self.execution_active = False
           self.execution_start_time = 0
           
           rospy.loginfo("BT Execution Manager initialized")
       
       def plan_callback(self, msg: String):
           """Handle incoming HTN plan"""
           try:
               plan = json.loads(msg.data)
               self.current_plan = plan
               
               # Convert HTN plan to behavior tree
               self.root_behavior_tree = self.convert_plan_to_bt(plan)
               
               if self.root_behavior_tree:
                   rospy.loginfo("Plan converted to behavior tree successfully")
                   self.publish_status("Plan ready for execution")
                   self.publish_status(f"Root BT: {self.root_behavior_tree.name}")
               else:
                   rospy.logerr("Failed to convert plan to behavior tree")
                   self.publish_status("Plan conversion failed")
                   
           except json.JSONDecodeError:
               rospy.logerr("Failed to parse plan JSON")
       
       def control_callback(self, msg: String):
           """Handle execution control commands"""
           command = msg.data.lower().strip()
           
           if command == "start":
               self.start_execution()
           elif command == "stop":
               self.stop_execution()
           elif command == "pause":
               self.pause_execution()
           elif command == "resume":
               self.resume_execution()
           else:
               rospy.logwarn(f"Unknown command: {command}")
       
       def convert_plan_to_bt(self, plan: Dict) > Optional[py_trees.behaviour.Behaviour]:
           """Convert HTN plan to behavior tree"""
           try:
               # Create a sequence for the main plan
               main_sequence = py_trees.composites.Sequence(name="MainPlan")
               
               # Add tasks to the sequence
               for task in plan.get("tasks", []):
                   bt_task = self.create_task_behavior(task)
                   if bt_task:
                       main_sequence.add_child(bt_task)
               
               return main_sequence
               
           except Exception as e:
               rospy.logerr(f"Error converting plan to BT: {e}")
               return None
       
       def create_task_behavior(self, task: Dict) > Optional[py_trees.behaviour.Behaviour]:
           """Create a behavior tree node for a task"""
           task_name = task.get("name", "unknown_task")
           task_params = task.get("parameters", {})
           
           # Create appropriate behavior based on task name
           if "navigate" in task_name:
               return NavigationBehavior(task_name, **task_params)
           elif "pick" in task_name or "grasp" in task_name:
               return GraspingBehavior(task_name, **task_params)
           elif "place" in task_name or "release" in task_name:
               return PlacingBehavior(task_name, **task_params)
           elif "inspect" in task_name:
               return InspectionBehavior(task_name, **task_params)
           elif "open" in task_name:
               return OpenContainerBehavior(task_name, **task_params)
           elif "close" in task_name:
               return CloseContainerBehavior(task_name, **task_params)
           elif "detect" in task_name:
               return DetectionBehavior(task_name, **task_params)
           elif "transport" in task_name:
               return TransportBehavior(task_name, **task_params)
           else:
               # Generic task executor
               return GenericTaskBehavior(task_name, task.get("action", "unknown"), task_params)
       
       def start_execution(self):
           """Start executing the plan"""
           if not self.root_behavior_tree:
               rospy.logerr("No plan to execute")
               self.publish_status("No plan to execute")
               return
           
           self.execution_active = True
           self.execution_start_time = time.time()
           
           # Setup tree
           self.root_behavior_tree.setup_with_descendants()
           
           rate = rospy.Rate(10)  # 10 Hz
           
           rospy.loginfo("Starting plan execution...")
           self.publish_status("Execution started")
           
           while self.execution_active and not rospy.is_shutdown():
               try:
                   # Tick the tree
                   self.root_behavior_tree.tick_once()
                   
                   # Check for completion
                   if self.root_behavior_tree.status == py_trees.common.Status.SUCCESS:
                       rospy.loginfo("Plan execution completed successfully!")
                       self.publish_status("Execution completed successfully")
                       self.execution_active = False
                       break
                   elif self.root_behavior_tree.status == py_trees.common.Status.FAILURE:
                       rospy.logerr("Plan execution failed!")
                       self.publish_feedback({"status": "failure", "error": "Task failed"})
                       self.publish_status("Execution failed")
                       self.execution_active = False
                       break
                   
                   # Publish feedback
                   self.publish_feedback({
                       "status": "running",
                       "tree_status": str(self.root_behavior_tree.status),
                       "elapsed_time": time.time()  self.execution_start_time
                   })
                   
                   rate.sleep()
                   
               except Exception as e:
                   rospy.logerr(f"Error during execution: {e}")
                   self.publish_status(f"Execution error: {e}")
                   self.stop_execution()
                   break
       
       def stop_execution(self):
           """Stop plan execution"""
           self.execution_active = False
           if self.root_behavior_tree:
               # Cancel any running actions
               self.root_behavior_tree.stop(py_trees.common.Status.INVALID)
           self.publish_status("Execution stopped")
       
       def pause_execution(self):
           """Pause plan execution"""
           self.execution_active = False
           self.publish_status("Execution paused")
       
       def resume_execution(self):
           """Resume plan execution"""
           if self.root_behavior_tree:
               self.execution_active = True
               rospy.loginfo("Resuming plan execution...")
               self.publish_status("Execution resumed")
           else:
               rospy.logwarn("No plan to resume")
       
       def publish_status(self, status: str):
           """Publish execution status"""
           status_msg = String()
           status_msg.data = status
           self.status_pub.publish(status_msg)
       
       def publish_feedback(self, feedback_data: Dict):
           """Publish execution feedback"""
           feedback_msg = String()
           feedback_msg.data = json.dumps(feedback_data)
           self.feedback_pub.publish(feedback_msg)

   # Behavior Tree Actions
   class NavigationBehavior(py_trees.behaviour.Behaviour):
       def __init__(self, name, location=None, navigation_pose=None, **kwargs):
           super(NavigationBehavior, self).__init__(name)
           self.location = location
           self.navigation_pose = navigation_pose or {}
           self.success_probability = kwargs.get("success_probability", 0.95)
       
       def setup(self, **kwargs):
           self.logger.debug(f"NavigationBehavior [{self.name}].setup()") # Debug logger
       
       def update(self):
           # Simulate navigation action
           # In a real robot, this would call navigation stack or move_base
           rospy.loginfo(f"Navigating to {self.location or self.navigation_pose}")
           
           # Simulate success/failure based on probability
           if rospy.Time.now().to_sec() % 100 < (1  self.success_probability) * 100:
               # Simulate occasional failure
               self.feedback_message = "Navigation failed"
               return py_trees.common.Status.FAILURE
           else:
               # Simulate time delay
               rospy.sleep(1.0)
               self.feedback_message = f"Reached {self.location or 'specified pose'}"
               return py_trees.common.Status.SUCCESS
       
       def terminate(self, new_status):
           self.logger.debug(f"NavigationBehavior [{self.name}].terminate({new_status})")

   class GraspingBehavior(py_trees.behaviour.Behaviour):
       def __init__(self, name, object_name=None, grasp_pose=None, **kwargs):
           super(GraspingBehavior, self).__init__(name)
           self.object_name = object_name
           self.grasp_pose = grasp_pose or {}
           self.success_probability = kwargs.get("success_probability", 0.85)
       
       def setup(self, **kwargs):
           self.logger.debug(f"GraspingBehavior [{self.name}].setup()")
       
       def update(self):
           rospy.loginfo(f"Attempting to grasp {self.object_name}")
           
           # Simulate success/failure
           import random
           if random.random() > self.success_probability:
               self.feedback_message = f"Failed to grasp {self.object_name}"
               return py_trees.common.Status.FAILURE
           else:
               rospy.sleep(2.0)  # Simulate grasping time
               self.feedback_message = f"Successfully grasped {self.object_name}"
               return py_trees.common.Status.SUCCESS
       
       def terminate(self, new_status):
           self.logger.debug(f"GraspingBehavior [{self.name}].terminate({new_status})")

   class PlacingBehavior(py_trees.behaviour.Behaviour):
       def __init__(self, name, object_name=None, placement_pose=None, **kwargs):
           super(PlacingBehavior, self).__init__(name)
           self.object_name = object_name
           self.placement_pose = placement_pose or {}
       
       def setup(self, **kwargs):
           self.logger.debug(f"PlacingBehavior [{self.name}].setup()")
       
       def update(self):
           rospy.loginfo(f"Placing {self.object_name}")
           rospy.sleep(1.5)  # Simulate placing time
           self.feedback_message = f"Successfully placed {self.object_name}"
           return py_trees.common.Status.SUCCESS
       
       def terminate(self, new_status):
           self.logger.debug(f"PlacingBehavior [{self.name}].terminate({new_status})")

   class InspectionBehavior(py_trees.behaviour.Behaviour):
       def __init__(self, name, inspection_targets=None, **kwargs):
           super(InspectionBehavior, self).__init__(name)
           self.targets = inspection_targets or []
       
       def setup(self, **kwargs):
           self.logger.debug(f"InspectionBehavior [{self.name}].setup()")
       
       def update(self):
           rospy.loginfo(f"Inspecting targets: {self.targets}")
           rospy.sleep(3.0)  # Simulate inspection time
           self.feedback_message = f"Inspection complete for {len(self.targets)} targets"
           return py_trees.common.Status.SUCCESS
       
       def terminate(self, new_status):
           self.logger.debug(f"InspectionBehavior [{self.name}].terminate({new_status})")

   # Additional behaviors can be added similarly...

   class GenericTaskBehavior(py_trees.behaviour.Behaviour):
       def __init__(self, name, action_type="unknown", parameters=None):
           super(GenericTaskBehavior, self).__init__(name)
           self.action_type = action_type
           self.parameters = parameters or {}
       
       def setup(self, **kwargs):
           self.logger.debug(f"GenericTaskBehavior [{self.name}].setup()")
       
       def update(self):
           rospy.loginfo(f"Executing generic task: {self.action_type} with params: {self.parameters}")
           rospy.sleep(1.0)  # Simulate task execution
           self.feedback_message = f"Completed {self.action_type}"
           return py_trees.common.Status.SUCCESS
       
       def terminate(self, new_status):
           self.logger.debug(f"GenericTaskBehavior [{self.name}].terminate({new_status})")

   def main():
       manager = BTExecutionManager()
       
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down BT Execution Manager")

   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's a complete system that ties GPTbased planning with behavior tree execution:

```python
#!/usr/bin/env python3
# complete_cognitive_planning_system.py

import openai
import json
import rospy
import py_trees
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose
from actionlib_msgs.msg import GoalStatusArray
import time
import threading
import queue

class CompleteCognitivePlanningSystem:
    """Complete cognitive planning system using GPT4o and behavior trees"""
    
    def __init__(self, api_key: str):
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = "gpt4o"
        
        # Initialize ROS
        rospy.init_node('complete_cognitive_planner', anonymous=True)
        
        # Publishers and Subscribers
        self.plan_pub = rospy.Publisher('/complete_plan', String, queue_size=10)
        self.status_pub = rospy.Publisher('/cognitive_planner_status', String, queue_size=10)
        self.feedback_pub = rospy.Publisher('/cognitive_feedback', String, queue_size=10)
        self.bt_command_pub = rospy.Publisher('/bt_control', String, queue_size=10)
        
        rospy.Subscriber('/cognitive_task', String, self.task_callback)
        rospy.Subscriber('/cognitive_status', String, self.status_feedback_callback)
        
        # Internal state
        self.active_plan = None
        self.plan_queue = queue.Queue()
        self.execution_thread = None
        self.is_executing = False
        
        # System parameters
        self.max_plan_steps = 50  # Prevent infinite plans
        self.default_success_probability = 0.9
        self.max_replanning_attempts = 3
        
        rospy.loginfo("Complete Cognitive Planning System initialized")
    
    def task_callback(self, msg: String):
        """Handle incoming cognitive tasks"""
        rospy.loginfo(f"Received cognitive task: {msg.data}")
        
        # Generate plan using GPT4o
        plan = self.generate_cognitive_plan(msg.data)
        
        if plan:
            self.active_plan = plan
            rospy.loginfo(f"Generated plan with {len(plan.get('tasks', []))} steps")
            
            # Publish plan for BT execution
            plan_msg = String()
            plan_msg.data = json.dumps(plan, indent=2)
            self.plan_pub.publish(plan_msg)
            
            # Trigger execution
            self.start_execution()
        else:
            rospy.logerr("Failed to generate cognitive plan")
            self.publish_status("Plan generation failed")
    
    def status_feedback_callback(self, msg: String):
        """Handle feedback from execution system"""
        try:
            feedback = json.loads(msg.data)
            status = feedback.get('status', '')
            
            if status == 'failure':
                rospy.loginfo("Plan execution failed, attempting replanning")
                self.handle_execution_failure(feedback)
            elif status == 'success':
                rospy.loginfo("Plan execution completed successfully")
                self.publish_status("Plan execution completed")
            elif status == 'running':
                progress = feedback.get('progress', 0)
                self.publish_status(f"Execution in progress: {progress}%")
        except json.JSONDecodeError:
            rospy.logerr("Failed to parse status feedback")
    
    def generate_cognitive_plan(self, task_description: str) > dict:
        """Generate cognitive plan using GPT4o"""
        # Get environment context
        context = self.get_environment_context()
        
        # Construct detailed planning prompt
        prompt = f"""
        Task: {task_description}
        
        Environment Context: {json.dumps(context, indent=2)}
        
        Available Robot Capabilities:
         Navigation: Move to locations, avoid obstacles
         Manipulation: Pick/place objects, open/close containers
         Perception: Detect objects, inspect areas, recognize faces
         Communication: Speak, listen, interact with humans
         Learning: Adapt to new situations, remember preferences
        
        HighLevel Actions:
         move_to(location): Navigate to named location
         pick_object(object, grasp_type): Grasp an object
         place_object(object, location): Place object at location
         detect_object(object_type): Find an object in the environment
         navigate_to_object(object_type): Move near an object
         open_container(container): Open a door or drawer
         close_container(container): Close a door or drawer
         inspect_area(area): Survey a specific area
         interact_with_person(person): Approach and talk to someone
         find_person(person_name): Locate a specific person
         deliver_object(object, recipient): Transport and hand over object
         charge_robot(): Return to charging station
         wait_for(time_period): Pause execution
        
        Generate a comprehensive cognitive plan that:
        1. Breaks down the task into logical highlevel steps
        2. Considers environmental context and constraints
        3. Accounts for robot capabilities and limitations
        4. Includes error handling and recovery strategies
        5. Estimates time and resource requirements
        6. Specifies success conditions for each step
        
        Return as JSON with structure:
        {{
          "task": "original task description",
          "reasoning": "stepbystep reasoning for the plan",
          "plan": [
            {{
              "step_id": 1,
              "description": "what this step accomplishes",
              "action": "high_level_action",
              "parameters": {{"param1": "value1", "param2": "value2"}},
              "context_requirements": ["condition1", "condition2"],
              "expected_outcomes": ["outcome1", "outcome2"],
              "timeout": 30,  # seconds
              "success_probability": 0.95,
              "recovery_plan": ["alternative_step1", "alternative_step2"]
            }}
          ],
          "estimated_total_time": 180,  # seconds
          "confidence": 0.85,
          "constraints": ["constraint1", "constraint2"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """Generate detailed cognitive plans for robotic tasks. 
                        Each plan should be comprehensive, considering environmental context, 
                        robot capabilities, and potential failure modes. 
                        Return only valid JSON as specified."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON
            if response_text.startswith('```'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                response_text = response_text[start_idx:end_idx]
            
            plan_data = json.loads(response_text)
            
            # Validate plan structure
            if self.validate_plan(plan_data):
                rospy.loginfo(f"Generated valid cognitive plan with {len(plan_data['plan'])} steps")
                return plan_data
            else:
                rospy.logerr("Generated plan failed validation")
                return {}
        except Exception as e:
            rospy.logerr(f"Error generating cognitive plan: {e}")
            return {}
    
    def validate_plan(self, plan: dict) > bool:
        """Validate the structure of a generated plan"""
        required_fields = ['task', 'plan', 'confidence']
        if not all(field in plan for field in required_fields):
            rospy.logwarn("Plan missing required fields")
            return False
        
        if not isinstance(plan['plan'], list):
            rospy.logwarn("Plan steps not in list format")
            return False
        
        if len(plan['plan']) > self.max_plan_steps:
            rospy.logwarn(f"Plan exceeds maximum step count ({self.max_plan_steps})")
            return False
        
        # Validate each step
        for i, step in enumerate(plan['plan']):
            required_step_fields = ['step_id', 'action', 'parameters']
            if not all(field in step for field in required_step_fields):
                rospy.logwarn(f"Step {i} missing required fields")
                return False
        
        if not (0.0 <= plan.get('confidence', 0.5) <= 1.0):
            rospy.logwarn("Plan confidence out of range [0,1]")
            return False
        
        return True
    
    def get_environment_context(self) > dict:
        """Get current environment context for planning"""
        # This would be populated from various sensors and state topics in a real system
        return {
            "robot_state": {
                "current_location": "charging_station",
                "battery_level": 0.75,
                "gripper_status": "open",
                "arm_position": "home",
                "navigation_status": "ready"
            },
            "known_locations": [
                {"name": "kitchen", "coordinates": [1.0, 1.0]},
                {"name": "living_room", "coordinates": [2.0, 0.5]},
                {"name": "bedroom", "coordinates": [0.5, 2.0]},
                {"name": "office", "coordinates": [1.5, 2.5]},
                {"name": "charging_station", "coordinates": [0.0, 0.0]}
            ],
            "detected_objects": [
                {"name": "red_apple", "type": "fruit", "location": "kitchen", "pose": [1.1, 1.1]},
                {"name": "blue_book", "type": "stationery", "location": "living_room", "pose": [2.1, 0.6]},
                {"name": "white_cup", "type": "container", "location": "kitchen", "pose": [1.2, 1.0]}
            ],
            "known_people": [
                {"name": "john", "location": "office", "last_seen": "20231015T10:30:00Z"},
                {"name": "mary", "location": "kitchen", "last_seen": "20231015T10:25:00Z"}
            ],
            "time_context": {
                "hour_of_day": 10,  # 24hour format
                "day_of_week": "monday",
                "month": "october",
                "is_daylight": True
            },
            "environment_constraints": {
                "fragile_objects": ["glassware", "electronics"],
                "restricted_areas": ["private_office"],
                "noise_limitations": ["during_meeting_hours"],
                "charging_deadline": "20231015T18:00:00Z"
            }
        }
    
    def start_execution(self):
        """Start executing the current plan"""
        if not self.active_plan or self.is_executing:
            return
        
        rospy.loginfo("Starting cognitive plan execution")
        self.is_executing = True
        self.publish_status("Plan execution started")
        
        # Publish to BT executor
        plan_msg = String()
        plan_msg.data = json.dumps(self.active_plan, indent=2)
        self.plan_pub.publish(plan_msg)
        
        # Send start command to BT executor
        start_cmd = String()
        start_cmd.data = "start"
        self.bt_command_pub.publish(start_cmd)
    
    def handle_execution_failure(self, feedback: dict):
        """Handle plan execution failure with recovery"""
        step_failed = feedback.get('step', 0)
        error = feedback.get('error', 'unknown')
        
        rospy.loginfo(f"Handling failure at step {step_failed}: {error}")
        self.publish_status(f"Failure at step {step_failed}: {error}")
        
        # Attempt recovery up to max attempts
        attempts = 0
        while attempts < self.max_replanning_attempts:
            recovery_plan = self.generate_recovery_plan(step_failed, error)
            
            if recovery_plan:
                rospy.loginfo(f"Recovery plan generated (attempt {attempts + 1})")
                
                # Update active plan with recovery plan
                self.active_plan = recovery_plan
                
                # Restart execution
                self.start_execution()
                return
            else:
                attempts += 1
                rospy.logwarn(f"Recovery attempt {attempts} failed, trying again...")
        
        # If all recovery attempts fail
        rospy.logerr(f"All {self.max_replanning_attempts} recovery attempts failed")
        self.publish_status(f"Recovery failed after {self.max_replanning_attempts} attempts")
        self.is_executing = False
    
    def generate_recovery_plan(self, failed_step: int, error: str) > dict:
        """Generate a recovery plan using GPT4o"""
        context = self.get_environment_context()
        
        prompt = f"""
        Original Task: {self.active_plan['task']}
        
        Failed Step: {failed_step}
        Error: {error}
        Current Environment: {json.dumps(context, indent=2)}
        
        Available Recovery Strategies:
         Retry step with different parameters
         Skip problematic step if possible
         Alternative approach to achieve same goal
         Abort task and return to safe state
         Request human assistance
         Charge robot if battery is low
        
        Generate a recovery plan that addresses the specific failure and attempts to continue task completion.
        The recovery plan should:
        1. Address the immediate failure
        2. Adapt the overall strategy if needed
        3. Consider current environment state
        4. Be executable by the robot
        
        Return in the same JSON format as the original plan.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate recovery plans for failed robotic tasks. Return valid JSON with a complete plan structure."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON
            if response_text.startswith('```'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                response_text = response_text[start_idx:end_idx]
            
            recovery_plan = json.loads(response_text)
            
            if self.validate_plan(recovery_plan):
                return recovery_plan
            else:
                rospy.logerr("Recovery plan failed validation")
                return {}
                
        except Exception as e:
            rospy.logerr(f"Error generating recovery plan: {e}")
            return {}
    
    def publish_status(self, status: str):
        """Publish system status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
    
    def run(self):
        """Run the cognitive planning system"""
        rospy.loginfo("Cognitive Planning System running...")
        rospy.spin()

def main():
    """Main function to run the complete system"""
    api_key = input("Enter OpenAI API key: ")
    if not api_key:
        rospy.logerr("No API key provided!")
        return
    
    system = CompleteCognitivePlanningSystem(api_key)
    system.run()

if __name__ == '__main__':
    main()
```

## Miniproject

Create a complete cognitive planning system that:

1. Implements GPT4obased hierarchical task planning
2. Integrates with behavior tree execution for plan execution
3. Handles complex multistep tasks with dependencies
4. Implements plan monitoring and execution feedback
5. Creates contextaware planning considering environment state
6. Implements recovery mechanisms for plan failures
7. Evaluates the quality and efficiency of generated plans
8. Demonstrates the system with complex multirobot scenarios

Your project should include:
 Complete GPT4o integration for cognitive planning
 Behavior tree execution system
 Contextaware environment model
 Plan monitoring and feedback mechanisms
 Recovery and replanning capabilities
 Performance evaluation metrics
 Demo scenarios with complex tasks

## Summary

This chapter covered cognitive task planning using GPT4o for robotics:

 **Cognitive Planning**: Highlevel reasoning about complex robotic tasks
 **GPT4o Integration**: Using large language models for plan generation
 **Hierarchical Decomposition**: Breaking complex tasks into manageable subtasks
 **Behavior Trees**: Executing plans with robust control structures
 **Context Awareness**: Adapting plans based on environment state
 **Recovery Mechanisms**: Handling failures and adapting plans
 **Performance Evaluation**: Assessing plan quality and execution success

Cognitive task planning with LLMs represents a significant advancement in robotics, enabling robots to understand complex tasks expressed in natural language and decompose them into executable actions while considering environmental context and potential failure modes.