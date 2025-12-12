---
title: Ch19 - Cognitive Task Planning with GPT-4o
module: 4
chapter: 19
sidebar_label: Ch19: Cognitive Task Planning with GPT-4o
description: Implementing high-level cognitive task planning using OpenAI GPT models for robotics
tags: [gpt-4o, cognitive-planning, task-planning, robotics, ai-planning, hierarchical-task-network, ros2]
difficulty: advanced
estimated_duration: 120
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Cognitive Task Planning with GPT-4o

## Learning Outcomes
- Understand cognitive task planning in robotics using LLMs
- Implement GPT-4o integration for high-level task decomposition
- Create hierarchical task networks with LLM guidance
- Develop context-aware planning systems
- Integrate LLM-based planning with traditional ROS 2 planning systems
- Implement plan validation and safety checks
- Handle multi-step task execution and monitoring
- Create fallback and recovery mechanisms for plan failures
- Evaluate planning performance and success rates

## Theory

### Cognitive Task Planning Fundamentals

Cognitive task planning involves high-level reasoning about complex tasks that require understanding of the environment, objects, and their relationships. Traditional robotics planning focuses on low-level motion planning, while cognitive planning addresses the "what" and "why" of robot actions.

<MermaidDiagram chart={`
graph TD;
    A[Cognitive Task Planning] --> B[Task Decomposition];
    A --> C[Context Understanding];
    A --> D[Knowledge Integration];
    A --> E[Reasoning & Inference];
    
    B --> F[High-Level Goals];
    B --> G[Intermediate Steps];
    B --> H[Primitive Actions];
    
    C --> I[Environmental State];
    C --> J[Object Affordances];
    C --> K[Robot Capabilities];
    
    D --> L[Common Sense];
    D --> M[World Knowledge];
    D --> N[Physical Laws];
    
    E --> O[Logical Reasoning];
    E --> P[Causal Inference];
    E --> Q[Plan Validation];
    
    R[Robot] --> S[Traditional Planner];
    T[GPT-4o] --> U[Cognitive Planner];
    
    S --> V[Low-Level Motion];
    U --> W[High-Level Reasoning];
    U --> X[Context Integration];
    
    V --> Y[Path Planning];
    W --> Z[Task Sequencing];
    X --> AA[Constraint Checking];
    
    BB[Human] --> CC[Natural Language Task];
    CC --> U;
    W --> DD[Robot Action Sequence];
    DD --> R;
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style T fill:#2196F3,stroke:#0D47A1,color:#fff;
    style CC fill:#FF9800,stroke:#E65100,color:#fff;
    style DD fill:#E91E63,stroke:#AD1457,color:#fff;
`} />

### GPT-4o for Task Planning

GPT-4o provides several advantages for cognitive task planning:

- **World Knowledge**: Comprehensive understanding of physical objects, their properties, and relationships
- **Logical Reasoning**: Ability to reason about cause and effect, preconditions, and effects
- **Contextual Understanding**: Incorporation of environmental context into planning
- **Natural Language Interface**: Direct translation of human instructions to robot actions
- **Flexibility**: Ability to handle novel situations and adapt plans

### Hierarchical Task Networks (HTNs)

HTNs decompose complex tasks into hierarchically organized subtasks. The planning process involves:

1. **Task Decomposition**: Breaking high-level goals into subtasks
2. **Constraint Propagation**: Maintaining consistency across subtask relationships
3. **Resource Allocation**: Ensuring resource requirements are met
4. **Temporal Scheduling**: Ordering subtasks according to dependencies

### Integration with Traditional Planning

Cognitive planning should integrate with traditional robotic planning systems:

- **High-Level**: LLM handles task decomposition and contextual reasoning
- **Mid-Level**: ROS navigation stack handles path planning and execution
- **Low-Level**: Motion controllers handle trajectory execution and feedback

### Context Integration

For effective task planning, GPT-4o must incorporate context from:

- **Current State**: Robot pose, battery level, internal state
- **Environmental State**: Sensor data from cameras, LiDAR, etc.
- **Historical Context**: Previous tasks, learned preferences
- **Constraint Information**: Safety requirements, operational constraints

## Step-by-Step Labs

### Lab 1: Setting up GPT-4o Integration for Planning

1. **Install required dependencies**:
   ```bash
   pip install openai==1.3.5
   pip install langchain langchain-openai
   pip install python-dotenv
   pip install ros2 rospy std_msgs geometry_msgs actionlib_msgs
   pip install py_trees  # For behavior trees integration
   ```

2. **Create a GPT-4o planning interface** (`gpt_planning_interface.py`):
   ```python
   #!/usr/bin/env python3

   import openai
   import rospy
   import json
   import time
   from std_msgs.msg import String
   from geometry_msgs.msg import PoseStamped
   from sensor_msgs.msg import LaserScan
   from actionlib_msgs.msg import GoalStatusArray
   from typing import Dict, List, Optional
   import threading
   import queue

   class GPTPlanningInterface:
       def __init__(self, api_key: str):
           # Initialize OpenAI client
           self.client = openai.OpenAI(api_key=api_key)
           self.model_name = "gpt-4o"
           
           # Initialize ROS
           rospy.init_node('gpt_planning_interface', anonymous=True)
           
           # Publishers and Subscribers
           self.plan_pub = rospy.Publisher('/gpt_generated_plan', String, queue_size=10)
           self.status_pub = rospy.Publisher('/gpt_planner_status', String, queue_size=10)
           self.feedback_pub = rospy.Publisher('/gpt_planning_feedback', String, queue_size=10)
           
           rospy.Subscriber('/task_request', String, self.task_request_callback)
           rospy.Subscriber('/robot_state', String, self.robot_state_callback)
           rospy.Subscriber('/environment_state', String, self.environment_state_callback)
           rospy.Subscriber('/execution_feedback', String, self.execution_feedback_callback)
           
           # Internal state
           self.robot_state = {}
           self.environment_state = {}
           self.current_task = None
           self.current_plan = None
           self.plan_queue = queue.Queue()
           
           # Planning parameters
           self.max_retries = 3
           self.planning_timeout = 30.0
           
           rospy.loginfo("GPT Planning Interface initialized")
       
       def task_request_callback(self, msg: String):
           """Handle new task requests"""
           task_data = json.loads(msg.data) if msg.data.startswith('{') else {"task": msg.data}
           task_description = task_data.get("task", "")
           priority = task_data.get("priority", "normal")
           
           rospy.loginfo(f"Received task request: {task_description} (Priority: {priority})")
           
           # Add to planning queue
           self.plan_queue.put({
               "task": task_description,
               "priority": priority,
               "timestamp": rospy.Time.now().to_sec(),
               "task_id": time.time_ns()  # Unique identifier
           })
       
       def robot_state_callback(self, msg: String):
           """Update robot state"""
           try:
               self.robot_state = json.loads(msg.data)
               rospy.logdebug(f"Updated robot state: {self.robot_state.keys()}")
           except json.JSONDecodeError:
               rospy.logerr("Failed to decode robot state")
       
       def environment_state_callback(self, msg: String):
           """Update environment state"""
           try:
               self.environment_state = json.loads(msg.data)
               rospy.logdebug(f"Updated environment state: {self.environment_state.keys()}")
           except json.JSONDecodeError:
               rospy.logerr("Failed to decode environment state")
       
       def execution_feedback_callback(self, msg: String):
           """Handle execution feedback for plan monitoring"""
           try:
               feedback = json.loads(msg.data)
               task_id = feedback.get("task_id")
               status = feedback.get("status")  # "success", "failure", "in_progress"
               step_completed = feedback.get("step_completed", -1)
               
               if status == "failure" and self.current_plan and self.current_plan.get("task_id") == task_id:
                   rospy.logwarn(f"Current plan failed at step {step_completed}")
                   self.handle_plan_failure(task_id, step_completed, feedback.get("error", ""))
               
           except json.JSONDecodeError:
               rospy.logerr("Failed to decode execution feedback")
       
       def handle_plan_failure(self, task_id: str, step_num: int, error: str):
           """Handle plan execution failure"""
           rospy.loginfo(f"Handling plan failure for task {task_id}, step {step_num}")
           
           # Generate recovery plan
           recovery_plan = self.generate_recovery_plan(task_id, step_num, error)
           
           if recovery_plan:
               rospy.loginfo("Recovery plan generated successfully")
               self.publish_plan(recovery_plan)
               self.publish_feedback({
                   "message": f"Recovery plan generated for failed step {step_num}",
                   "type": "recovery",
                   "task_id": task_id
               })
           else:
               rospy.logerr("Failed to generate recovery plan")
               self.publish_feedback({
                   "message": f"Could not generate recovery plan for failure at step {step_num}",
                   "type": "error",
                   "task_id": task_id
               })
       
       def generate_recovery_plan(self, task_id: str, failed_step: int, error: str) -> Optional[Dict]:
           """Generate recovery plan when original plan fails"""
           # Get context at time of failure
           context = {
               "robot_state": self.robot_state,
               "environment_state": self.environment_state,
               "original_task": self.current_task,
               "failed_step": failed_step,
               "error": error
           }
           
           prompt = f"""
           Original Task: {self.current_task}
           
           Plan Failed at Step: {failed_step}
           Error Encountered: {error}
           
           Current Robot State: {json.dumps(self.robot_state, indent=2)}
           Current Environment State: {json.dumps(self.environment_state, indent=2)}
           
           Available Robot Capabilities:
           - Navigation: move_to_location, navigate_to_object
           - Manipulation: grasp_object, place_object, open_container, close_container
           - Perception: detect_object, inspect_area
           - Communication: speak_text, play_sound
           
           Generate a recovery plan that:
           1. Addresses the specific failure that occurred
           2. Attempts to resume the original task if possible
           3. Includes alternative strategies if the original approach failed
           4. Considers the current robot state and environment
           5. Is executable by the robot
           
           Return the plan as JSON with the structure:
           {{
             "task_id": "original_task_id",
             "task": "original task with recovery context",
             "plan_type": "recovery",
             "recovery_strategy": "strategy_used",
             "steps": [
               {{
                 "id": 1,
                 "action": "action_type",
                 "parameters": {{"param1": "value1"}},
                 "description": "what this step does",
                 "preconditions": ["condition1", "condition2"],
                 "expected_effects": ["effect1", "effect2"],
                 "estimated_duration": 10.0
               }}
             ],
             "estimated_completion_time": 120.0,
             "confidence": 0.8,
             "fallback_options": ["option1", "option2"]
           }}
           
           Only return the JSON plan, no other text.
           """
           
           try:
               response = self.client.chat.completions.create(
                   model=self.model_name,
                   messages=[
                       {
                           "role": "system",
                           "content": "You are a robot task recovery planner. Generate recovery plans that address specific failures and attempt to continue task completion. Only return valid JSON."
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
               
               # Extract JSON if wrapped in code block
               if response_text.startswith('```'):
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   if start_idx != -1 and end_idx != -1:
                       response_text = response_text[start_idx:end_idx]
               
               recovery_plan = json.loads(response_text)
               recovery_plan["timestamp"] = rospy.Time.now().to_sec()
               
               return recovery_plan
               
           except Exception as e:
               rospy.logerr(f"Error generating recovery plan: {e}")
               return None
       
       def run_planning_loop(self):
           """Main planning loop to process tasks asynchronously"""
           rate = rospy.Rate(1)  # 1 Hz polling of task queue
           
           while not rospy.is_shutdown():
               try:
                   # Get next task from queue
                   task_request = self.plan_queue.get(timeout=1.0)
                   
                   # Generate plan for task
                   plan = self.generate_plan(task_request)
                   
                   if plan:
                       self.current_plan = plan
                       self.current_task = task_request["task"]
                       self.publish_plan(plan)
                       self.publish_status(f"Plan generated for task ID: {task_request['task_id']}")
                   else:
                       self.publish_status(f"Failed to generate plan for task ID: {task_request['task_id']}")
                       self.publish_feedback({
                           "message": "Plan generation failed",
                           "type": "error",
                           "task_id": task_request['task_id']
                       })
                   
                   self.plan_queue.task_done()
                   
               except queue.Empty:
                   continue
               except Exception as e:
                   rospy.logerr(f"Error in planning loop: {e}")
                   continue
               
               rate.sleep()
       
       def generate_plan(self, task_request: Dict) -> Optional[Dict]:
           """Generate a plan using GPT-4o"""
           task_description = task_request["task"]
           
           # Construct planning context
           context = {
               "robot_capabilities": self.robot_state.get("capabilities", []),
               "current_robot_state": self.robot_state,
               "environment_objects": self.environment_state.get("objects", []),
               "known_locations": self.environment_state.get("locations", []),
               "previous_experiences": self.environment_state.get("historical_tasks", []),
               "safety_constraints": ["avoid_obstacles", "maintain_battery_above_20_percent"],
               "time_constraints": task_request.get("deadline", "none")
           }
           
           # Create detailed planning prompt
           prompt = f"""
           Task to Plan: {task_description}
           
           Context Information:
           Robot Capabilities: {json.dumps(context['robot_capabilities'])}
           Current Robot State: {json.dumps(context['current_robot_state'])}
           Environmental Objects: {json.dumps(context['environment_objects'])}
           Known Locations: {json.dumps(context['known_locations'])}
           Safety Constraints: {json.dumps(context['safety_constraints'])}
           Time Constraints: {context['time_constraints']}
           
           Available High-Level Actions:
           - navigate: Move robot to a location or object
             Parameters: target_location, speed, safety_margin
           - manipulate: Manipulate objects
             Parameters: action (grasp/place/open/close), object_name, grasp_pose
           - perceive: Use sensors to gather information
             Parameters: type (detect/inspect/recognize), target_object
           - communicate: Interact with humans or systems
             Parameters: message_type, content, target_recipient
           - wait: Pause execution
             Parameters: duration, condition_to_monitor
           
           Generate a detailed plan that:
           1. Breaks the high-level task into a sequence of specific actions
           2. Considers the robot's capabilities and current state
           3. Takes environmental context into account
           4. Respects safety and temporal constraints
           5. Handles potential failure modes
           6. Includes success criteria for each step
           
           The plan should be in JSON format:
           {{
             "task_id": "{task_request['task_id']}",
             "task": "{task_description}",
             "plan_type": "high_level",
             "steps": [
               {{
                 "id": 1,
                 "action": "action_type",
                 "parameters": {{"param1": "value1", "param2": "value2"}},
                 "description": "Detailed description of what this step accomplishes",
                 "preconditions": [
                   "conditions that must be true before step execution"
                 ],
                 "expected_effects": [
                   "expected outcomes of the step execution"
                 ],
                 "execution_constraints": [
                   "time limits, safety requirements, etc."
                 ],
                 "failure_recovery": [
                   "actions to take if this step fails"
                 ],
                 "success_criteria": [
                   "how to verify step completion"
                 ],
                 "estimated_duration": 15.0,
                 "confidence": 0.8
               }}
             ],
             "total_estimated_duration": 120.0,
             "overall_confidence": 0.75,
             "risk_factors": ["factor1", "factor2"],
             "validation_steps": [
               "steps to verify plan feasibility before execution"
             ]
           }}
           
           Only return the JSON plan with no other text.
           """
           
           try:
               response = self.client.chat.completions.create(
                   model=self.model_name,
                   messages=[
                       {
                           "role": "system", 
                           "content": "You are an advanced robotic task planner. Generate detailed, executable plans that consider robot capabilities, environmental context, and safety constraints. Respond only with valid JSON."
                       },
                       {
                           "role": "user",
                           "content": prompt
                       }
                   ],
                   temperature=0.1,  # Low temperature for consistency
                   max_tokens=2500
               )
               
               response_text = response.choices[0].message.content.strip()
               
               # Extract JSON from response
               if response_text.startswith('```'):
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   if start_idx != -1 and end_idx != -1:
                       response_text = response_text[start_idx:end_idx]
               
               plan_data = json.loads(response_text)
               plan_data["timestamp"] = rospy.Time.now().to_sec()
               
               # Validate plan structure
               if self.validate_plan_structure(plan_data):
                   rospy.loginfo(f"Generated valid plan with {len(plan_data['steps'])} steps")
                   return plan_data
               else:
                   rospy.logerr("Generated plan failed validation")
                   return None
                   
           except Exception as e:
               rospy.logerr(f"Error generating plan: {e}")
               return None
       
       def validate_plan_structure(self, plan: Dict) -> bool:
           """Validate the structure of a generated plan"""
           required_fields = ["task_id", "task", "steps", "overall_confidence"]
           step_required_fields = ["id", "action", "parameters", "description", "success_criteria"]
           
           # Check top-level fields
           if not all(field in plan for field in required_fields):
               rospy.logerr("Plan missing required top-level fields")
               return False
           
           # Check plan steps
           if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
               rospy.logerr("Plan has no steps or steps not in list format")
               return False
           
           # Validate each step
           for step in plan["steps"]:
               if not all(field in step for field in step_required_fields):
                   rospy.logerr(f"Step {step.get('id', 'unknown')} missing required fields")
                   return False
               
               # Validate action type (should be in known action types)
               valid_actions = ["navigate", "manipulate", "perceive", "communicate", "wait"]
               if step["action"] not in valid_actions:
                   rospy.logwarn(f"Unknown action type in step {step['id']}: {step['action']}")
           
           return True
       
       def publish_plan(self, plan: Dict):
           """Publish generated plan"""
           plan_msg = String()
           plan_msg.data = json.dumps(plan, indent=2)
           self.plan_pub.publish(plan_msg)
       
       def publish_status(self, status: str):
           """Publish planner status"""
           status_msg = String()
           status_msg.data = json.dumps({
               "status": status,
               "timestamp": rospy.Time.now().to_sec()
           })
           self.status_pub.publish(status_msg)
       
       def publish_feedback(self, feedback: Dict):
           """Publish planning feedback"""
           feedback_msg = String()
           feedback_msg.data = json.dumps(feedback, indent=2)
           self.feedback_pub.publish(feedback_msg)

   class GPTPlanningNode:
       def __init__(self):
           self.planner_interface = None
       
       def run(self):
           # Get API key from parameter
           api_key = rospy.get_param('~openai_api_key', '')
           if not api_key:
               api_key = input("Enter OpenAI API key: ").strip()
           
           if not api_key:
               rospy.logerr("No OpenAI API key provided!")
               return
           
           self.planner_interface = GPTPlanningInterface(api_key)
           
           # Start planning loop in separate thread
           planning_thread = threading.Thread(target=self.planner_interface.run_planning_loop)
           planning_thread.daemon = True
           planning_thread.start()
           
           rospy.loginfo("GPT Planning Node running...")
           rospy.spin()

   def main():
       rospy.init_node('gpt_planning_node', anonymous=True)
       node = GPTPlanningNode()
       node.run()

   if __name__ == '__main__':
       main()
   ```

### Lab 2: Creating Hierarchical Task Networks with GPT-4o

1. **Create an HTN planner with GPT-4o guidance** (`htn_gpt_planner.py`):
   ```python
   #!/usr/bin/env python3

   import openai
   import rospy
   import json
   import time
   from std_msgs.msg import String
   from geometry_msgs.msg import PoseStamped
   from actionlib_msgs.msg import GoalID
   from typing import Dict, List, Optional, Tuple
   import threading
   import queue
   import re

   class GPTHierarchicalPlanner:
       def __init__(self, api_key: str):
           # Initialize OpenAI client
           self.client = openai.OpenAI(api_key=api_key)
           self.model_name = "gpt-4o"
           
           # Initialize ROS
           rospy.init_node('gpt_hierarchical_planner', anonymous=True)
           
           # Publishers and Subscribers
           self.high_level_plan_pub = rospy.Publisher('/hl_plan', String, queue_size=10)
           self.low_level_plan_pub = rospy.Publisher('/ll_plan', String, queue_size=10)
           self.plan_status_pub = rospy.Publisher('/plan_status', String, queue_size=10)
           
           rospy.Subscriber('/high_level_task', String, self.hl_task_callback)
           rospy.Subscriber('/task_decomposition_request', String, self.decomposition_request_callback)
           rospy.Subscriber('/plan_execution_feedback', String, self.execution_feedback_callback)
           
           # Internal state
           self.current_plan = None
           self.plan_history = []
           self.max_history_size = 100
           
           # Knowledge base for planning
           self.knowledge_base = {
               "object_affordances": {
                   "cup": ["grasp", "move", "fill", "contain"],
                   "table": ["support", "place_on", "surface_for"],
                   "drawer": ["open", "close", "contain", "access"],
                   "door": ["open", "close", "pass_through", "barrier"],
                   "bottle": ["grasp", "pour", "contain", "manipulate"],
                   "book": ["grasp", "read", "organize", "locate"],
                   "chair": ["occupy", "move", "support", "sit_on"]
               },
               "spatial_relations": [
                   "left_of", "right_of", "in_front_of", "behind", 
                   "next_to", "on_top_of", "under", "inside",
                   "near", "far", "between", "above", "below"
               ],
               "robot_capabilities": {
                   "navigation": {
                       "actions": ["navigate_to", "go_to", "move_to", "reach"],
                       "constraints": ["no_collision", "reachable", "known_map"]
                   },
                   "manipulation": {
                       "actions": ["grasp", "place", "pick", "put_down", "open", "close"],
                       "constraints": ["reachable", "graspable", "manipulable"]
                   },
                   "perception": {
                       "actions": ["detect", "recognize", "inspect", "localize"],
                       "constraints": ["visible", "sensor_range"]
                   }
               }
           }
           
           rospy.loginfo("GPT Hierarchical Planner initialized")
       
       def hl_task_callback(self, msg: String):
           """Handle high-level task requests"""
           try:
               task_data = json.loads(msg.data)
               task_description = task_data["task"]
               priority = task_data.get("priority", "normal")
               deadline = task_data.get("deadline", None)
               
               rospy.loginfo(f"High-level task received: {task_description}")
               
               # Generate hierarchical plan
               plan = self.generate_hierarchical_plan(task_description, priority, deadline)
               
               if plan:
                   self.current_plan = plan
                   self.add_to_plan_history(plan)
                   self.publish_hierarchical_plan(plan)
                   self.publish_plan_status("Plan generated and published")
               else:
                   rospy.logerr("Failed to generate hierarchical plan")
                   self.publish_plan_status("Plan generation failed")
                   
           except json.JSONDecodeError:
               rospy.logerr("Invalid JSON in high-level task message")
           except KeyError as e:
               rospy.logerr(f"Missing key in task data: {e}")
       
       def generate_hierarchical_plan(self, task_description: str, priority: str = "normal", deadline: Optional[float] = None) -> Optional[Dict]:
           """Generate hierarchical plan using GPT-4o"""
           # Get environmental context
           context = self.get_environment_context()
           
           # Create detailed prompt for hierarchical planning
           prompt = f"""
           High-Level Task: {task_description}
           
           Environmental Context: {json.dumps(context, indent=2)}
           
           Robot Capabilities: {json.dumps(self.knowledge_base["robot_capabilities"], indent=2)}
           Object Affordances: {json.dumps(self.knowledge_base["object_affordances"], indent=2)}
           
           Task Priority: {priority}
           Deadline: {deadline if deadline is not None else 'none'}
           
           Available Task Templates:
           - NavigateToObject: move to object, perceive, prepare for manipulation
           - GraspObject: approach, grasp, verify grip
           - PlaceObject: navigate, place, verify placement
           - OpenContainer: locate, approach, open, verify
           - CloseContainer: locate, approach, close, verify
           - InspectArea: perceive, analyze, report
           - TransportObject: pick, navigate, place
           - SetUpWorkspace: arrange objects, prepare area
           - CleanUpWorkspace: collect, organize, store
           
           Generate a hierarchical task network that:
           1. Decomposes the high-level task into intermediate subtasks
           2. Further decomposes subtasks into primitive actions
           3. Considers object affordances and spatial relationships
           4. Respects robot capabilities and constraints
           5. Includes temporal dependencies and resource requirements
           6. Provides fallback strategies for failure recovery
           
           Return the plan in JSON format:
           {{
             "task_id": "unique_identifier",
             "task_description": "{task_description}",
             "priority": "{priority}",
             "deadline": {deadline if deadline is not None else 'null'},
             "hierarchy": {{
               "level_1": [  // High-level objectives
                 {{
                   "id": "L1_1",
                   "name": "subtask_name",
                   "description": "what this subtask achieves",
                   "dependencies": ["L1_0_previous_task_id"],  // Task dependencies
                   "resources_required": ["navigation", "manipulation"],
                   "estimated_duration": 30.0,
                   "confidence": 0.8
                 }}
               ],
               "level_2": [  // Action sequences for each level 1 task
                 {{
                   "id": "L2_1",
                   "parent_id": "L1_1",
                   "name": "action_sequence_name",
                   "description": "sequence of actions",
                   "actions": [  // Primitive actions
                     {{
                       "id": "A1",
                       "type": "navigatge_to_object|grasp_object|place_object|etc.",
                       "parameters": {{"object": "name", "location": "coordinate"}},
                       "preconditions": ["robot_free", "object_visible"],
                       "effects": ["robot_at_location", "object_grasped"],
                       "estimated_duration": 10.0,
                       "probability_of_success": 0.95
                     }}
                   ],
                   "ordered": true,  // Whether actions must be executed in sequence
                   "parallelizable": false,  // Whether actions can run in parallel
                   "synchronization_points": ["A3"]  // Actions that require synchronization
                 }}
               ]
             }},
             "execution_order": ["L1_1", "L1_2"],  // Order of level 1 tasks
             "resource_schedule": {{
               "navigation": ["L2_1", "L2_3"],
               "manipulation": ["L2_2"],
               "perception": ["L2_1", "L2_2"]
             }},
             "risk_assessment": [
               {{"risk": "object_not_found", "probability": 0.1, "mitigation": "search_alternatives"}},
               {{"risk": "grasp_failure", "probability": 0.05, "mitigation": "retry_different_grasp"}}
             ],
             "success_criteria": [
               "primary_object_placed_correctly",
               "workspace_cleaned_up"
             ],
             "validation_checks": [
               "precondition_verification",
               "intermediate_state_checking",
               "final_state_verification"
             ],
             "estimated_total_duration": 120.0,
             "overall_confidence": 0.7
           }}
           
           Only return the JSON plan, no other text.
           """
           
           try:
               response = self.client.chat.completions.create(
                   model=self.model_name,
                   messages=[
                       {
                           "role": "system",
                           "content": "You are a hierarchical task planner for robotics. Generate comprehensive hierarchical plans with proper task decomposition, dependencies, and validation. Respond only with valid JSON."
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
               
               # Extract JSON from response
               if response_text.startswith('```'):
                   start_idx = response_text.find('{')
                   end_idx = response_text.rfind('}') + 1
                   if start_idx != -1 and end_idx != -1:
                       response_text = response_text[start_idx:end_idx]
               
               plan_data = json.loads(response_text)
               plan_data["timestamp"] = rospy.Time.now().to_sec()
               
               # Validate plan before returning
               if self.validate_hierarchical_plan(plan_data):
                   rospy.loginfo(f"Generated hierarchical plan with {len(plan_data['hierarchy']['level_1'])} high-level tasks")
                   return plan_data
               else:
                   rospy.logerr("Generated hierarchical plan failed validation")
                   return None
                   
           except Exception as e:
               rospy.logerr(f"Error generating hierarchical plan: {e}")
               return None
       
       def get_environment_context(self) -> Dict:
           """Get current environment context for planning"""
           # In a real implementation, this would come from various ROS topics
           # and might involve querying known object locations, robot state, etc.
           return {
               "robot_state": {
                   "current_location": "home_position",
                   "battery_level": 0.85,
                   "gripper_status": "open",
                   "navigation_status": "ready"
               },
               "objects_in_environment": [
                   {"name": "coffee_cup", "type": "cup", "location": "kitchen_table", "graspable": True},
                   {"name": "water_bottle", "type": "bottle", "location": "desk", "graspable": True},
                   {"name": "book", "type": "book", "location": "shelf", "graspable": True},
                   {"name": "drawer", "type": "container", "location": "kitchen_counter", "accessible": True, "contents": []}
               ],
               "known_locations": {
                   "kitchen_table": {"x": 1.0, "y": 1.0, "theta": 0.0},
                   "desk": {"x": 2.0, "y": 0.5, "theta": 0.0},
                   "shelf": {"x": 3.0, "y": 1.5, "theta": 0.0},
                   "kitchen_counter": {"x": 0.5, "y": 0.8, "theta": 0.0}
               },
               "spatial_constraints": {
                   "narrow_corridors": [["kitchen", "office"]],
                   "blocked_areas": [],
                   "preferred_paths": []
               }
           }
       
       def validate_hierarchical_plan(self, plan: Dict) -> bool:
           """Validate hierarchical plan structure"""
           required_fields = ["task_id", "task_description", "hierarchy", "execution_order"]
           hierarchy_required = ["level_1", "level_2"]
           
           # Check top-level fields
           if not all(field in plan for field in required_fields):
               rospy.logerr("Plan missing required top-level fields")
               return False
           
           # Check hierarchy structure
           if not all(level in plan["hierarchy"] for level in hierarchy_required):
               rospy.logerr("Plan hierarchy missing required levels")
               return False
           
           # Validate level 1 tasks
           level1_tasks = plan["hierarchy"]["level_1"]
           if not isinstance(level1_tasks, list) or len(level1_tasks) == 0:
               rospy.logerr("No level 1 tasks in hierarchical plan")
               return False
           
           # Validate level 2 sequences
           level2_sequences = plan["hierarchy"]["level_2"]
           if not isinstance(level2_sequences, list):
               rospy.logerr("Level 2 sequences not in list format")
               return False
           
           # Check that level 2 sequences reference valid level 1 parents
           valid_l1_ids = {task["id"] for task in level1_tasks}
           for seq in level2_sequences:
               if seq.get("parent_id") not in valid_l1_ids:
                   rospy.logerr(f"Level 2 sequence references invalid parent: {seq.get('parent_id')}")
                   return False
           
           return True
       
       def publish_hierarchical_plan(self, plan: Dict):
           """Publish the complete hierarchical plan"""
           # Publish high-level plan
           hl_plan = {
               "task_id": plan["task_id"],
               "tasks": plan["hierarchy"]["level_1"],
               "execution_order": plan["execution_order"],
               "timestamp": plan["timestamp"]
           }
           
           hl_msg = String()
           hl_msg.data = json.dumps(hl_plan, indent=2)
           self.high_level_plan_pub.publish(hl_msg)
           
           # Publish low-level plans
           ll_plans = {
               "task_id": plan["task_id"],
               "action_sequences": plan["hierarchy"]["level_2"],
               "resource_schedule": plan["resource_schedule"],
               "timestamp": plan["timestamp"]
           }
           
           ll_msg = String()
           ll_msg.data = json.dumps(ll_plans, indent=2)
           self.low_level_plan_pub.publish(ll_msg)
       
       def add_to_plan_history(self, plan: Dict):
           """Add plan to history with size limiting"""
           self.plan_history.append(plan)
           if len(self.plan_history) > self.max_history_size:
               self.plan_history = self.plan_history[-self.max_history_size:]
       
       def publish_plan_status(self, status: str):
           """Publish plan status"""
           status_msg = String()
           status_msg.data = json.dumps({
               "status": status,
               "timestamp": rospy.Time.now().to_sec()
           })
           self.plan_status_pub.publish(status_msg)
       
       def run(self):
           """Run the hierarchical planning node"""
           rospy.loginfo("GPT Hierarchical Planner running...")
           rospy.spin()

   def main():
       api_key = rospy.get_param('~openai_api_key', '')
       if not api_key:
           api_key = input("Enter OpenAI API key: ").strip()
       
       if not api_key:
           rospy.logerr("No OpenAI API key provided!")
           return
       
       planner = GPTHierarchicalPlanner(api_key)
       planner.run()

   if __name__ == '__main__':
       main()
   ```

### Lab 3: Integrating with ROS Navigation Stack

1. **Create a planning integration node** (`planning_integration_node.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import actionlib
   from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
   from geometry_msgs.msg import PoseStamped, Twist
   from std_msgs.msg import String, Bool
   from sensor_msgs.msg import LaserScan
   from gpt_planning_interface import GPTPlanningInterface
   from htn_gpt_planner import GPTHierarchicalPlanner
   import json
   import time
   import threading
   import numpy as np

   class PlanningIntegrationNode:
       def __init__(self):
           rospy.init_node('planning_integration_node', anonymous=True)
           
           # Initialize planners
           api_key = rospy.get_param('~openai_api_key', '')
           if not api_key:
               api_key = input("Enter OpenAI API key: ").strip()
           
           if api_key:
               self.gpt_planner = GPTPlanningInterface(api_key)
               self.htn_planner = GPTHierarchicalPlanner(api_key)
           else:
               rospy.logerr("No API key provided, planners will not be initialized")
           
           # Initialize action clients
           self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
           self.move_base_client.wait_for_server()
           
           # Publishers and Subscribers
           self.nav_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           rospy.Subscriber('/gpt_generated_plan', String, self.gpt_plan_callback)
           rospy.Subscriber('/hl_plan', String, self.hl_plan_callback)
           rospy.Subscriber('/integration_control', String, self.integration_control_callback)
           rospy.Subscriber('/task_request_natural', String, self.natural_task_callback)
           
           # Internal state
           self.robot_position = PoseStamped()
           self.current_goal = None
           self.plan_execution_active = False
           self.execution_thread = None
           
           rospy.loginfo("Planning Integration Node initialized")
       
       def natural_task_callback(self, msg: String):
           """Handle natural language task requests"""
           task_description = msg.data
           rospy.loginfo(f"Received natural language task: {task_description}")
           
           # Determine if this is a high-level task or low-level command
           if self.is_high_level_task(task_description):
               # Send to hierarchical planner
               hl_task_msg = String()
               hl_task_msg.data = json.dumps({"task": task_description})
               self.high_level_task_pub.publish(hl_task_msg)
           else:
               # Send to GPT planner for direct command processing
               gpt_task_msg = String()
               gpt_task_msg.data = json.dumps({"task": task_description})
               self.gpt_task_pub.publish(gpt_task_msg)
       
       def is_high_level_task(self, description: str) -> bool:
           """Determine if a task is high-level or low-level"""
           high_level_keywords = [
               'arrange', 'organize', 'clean', 'prepare', 'assist', 'help',
               'bring me', 'fetch', 'setup', 'assemble', 'construct', 'repair',
               'maintain', 'monitor', 'coordinate', 'manage', 'plan', 'schedule',
               'organize'
           ]
           
           low_level_keywords = [
               'go to', 'move to', 'go forward', 'turn left', 'turn right',
               'stop', 'halt', 'navigate', 'move', 'go', 'turn'
           ]
           
           desc_lower = description.lower()
           
           # Check for high-level keywords
           if any(kw in desc_lower for kw in high_level_keywords):
               return True
           
           # Check for low-level keywords
           if any(kw in desc_lower for kw in low_level_keywords):
               return False
           
           # Default to high-level for ambiguous tasks
           return True
       
       def gpt_plan_callback(self, msg: String):
           """Handle GPT-generated plans"""
           try:
               plan_data = json.loads(msg.data)
               rospy.loginfo("Received GPT-generated plan")
               
               # Execute the plan
               if not self.plan_execution_active:
                   self.execute_gpt_plan(plan_data)
               else:
                   rospy.logwarn("Plan execution already active, queuing plan...")
                   # In a real system, you'd have a plan queue
                   
           except json.JSONDecodeError:
               rospy.logerr("Invalid JSON in GPT plan message")
       
       def hl_plan_callback(self, msg: String):
           """Handle hierarchical plans"""
           try:
               hl_plan_data = json.loads(msg.data)
               rospy.loginfo("Received hierarchical plan")
               
               # Execute the hierarchical plan
               if not self.plan_execution_active:
                   self.execute_hierarchical_plan(hl_plan_data)
               else:
                   rospy.logwarn("Plan execution already active, queuing plan...")
                   
           except json.JSONDecodeError:
               rospy.logerr("Invalid JSON in hierarchical plan message")
       
       def execute_gpt_plan(self, plan: Dict):
           """Execute a plan generated by GPT"""
           self.plan_execution_active = True
           
           for step in plan.get("steps", []):
               action_type = step["action"]
               parameters = step.get("parameters", {})
               
               rospy.loginfo(f"Executing step: {action_type} with {parameters}")
               
               success = False
               retries = 0
               max_retries = 3
               
               while not success and retries < max_retries:
                   if action_type == "navigate":
                       success = self.execute_navigation_step(parameters)
                   elif action_type == "manipulate":
                       success = self.execute_manipulation_step(parameters)
                   elif action_type == "perceive":
                       success = self.execute_perception_step(parameters)
                   elif action_type == "communicate":
                       success = self.execute_communication_step(parameters)
                   else:
                       rospy.logwarn(f"Unknown action type: {action_type}")
                       success = True  # Skip unknown actions
                       break
                   
                   if not success:
                       retries += 1
                       rospy.logwarn(f"Step failed, retrying ({retries}/{max_retries})...")
                       time.sleep(1.0)
               
               if not success:
                   rospy.logerr(f"Step failed after {max_retries} retries: {action_type}")
                   # Consider plan failure handling
                   break
               
               # Check for preconditions before next step
               if not self.verify_preconditions(step.get("preconditions", [])):
                   rospy.logerr("Preconditions not met for next step")
                   break
           
           self.plan_execution_active = False
           rospy.loginfo("GPT plan execution completed")
       
       def execute_navigation_step(self, params: Dict) -> bool:
           """Execute navigation step"""
           target_location = params.get("target_location")
           
           if not target_location:
               rospy.logerr("Navigation step missing target location")
               return False
           
           # Convert target location to coordinates
           target_pose = self.get_location_pose(target_location)
           if not target_pose:
               rospy.logerr(f"Unknown location: {target_location}")
               return False
           
           # Send navigation goal
           goal = MoveBaseGoal()
           goal.target_pose.header.frame_id = "map"
           goal.target_pose.header.stamp = rospy.Time.now()
           goal.target_pose.pose = target_pose
           
           # Send goal to move_base
           self.move_base_client.send_goal(goal)
           
           # Wait for result
           finished_within_time = self.move_base_client.wait_for_result(rospy.Duration(60))  # 1 minute timeout
           
           if not finished_within_time:
               self.move_base_client.cancel_goal()
               rospy.logerr("Navigation timed out")
               return False
           
           state = self.move_base_client.get_state()
           if state == actionlib.GoalStatus.SUCCEEDED:
               rospy.loginfo(f"Successfully navigated to {target_location}")
               return True
           else:
               rospy.logerr(f"Navigation failed with state: {state}")
               return False
       
       def execute_manipulation_step(self, params: Dict) -> bool:
           """Execute manipulation step (stub for real manipulation system)"""
           action = params.get("action")
           object_name = params.get("object_name")
           
           rospy.loginfo(f"Manipulation step: {action} {object_name}")
           
           # In a real system, this would interface with manipulation stack
           # For now, simulate completion
           time.sleep(2.0)  # Simulate manipulation time
           
           rospy.loginfo(f"Manipulation step completed: {action} {object_name}")
           return True
       
       def execute_perception_step(self, params: Dict) -> bool:
           """Execute perception step (stub for real perception system)"""
           perception_type = params.get("type")
           target_object = params.get("target_object", "unknown")
           
           rospy.loginfo(f"Perception step: {perception_type} {target_object}")
           
           # In a real system, this would trigger perception algorithms
           # For now, simulate perception
           time.sleep(1.0)  # Simulate perception time
           
           rospy.loginfo(f"Perception step completed: {perception_type} {target_object}")
           return True
       
       def execute_communication_step(self, params: Dict) -> bool:
           """Execute communication step (stub for real voice system)"""
           message_type = params.get("message_type", "text")
           content = params.get("content", "Hello")
           target = params.get("target_recipient", "operator")
           
           rospy.loginfo(f"Communication step: {message_type} to {target}: {content}")
           
           # In a real system, this would interface with communication stack
           # For now, simulate communication
           time.sleep(0.5)  # Simulate communication time
           
           rospy.loginfo(f"Communication step completed: {content}")
           return True
       
       def get_location_pose(self, location_name: str) -> Optional:
           """Get pose for named location"""
           # In real implementation, this would query a location database
           # For now, provide some predefined locations
           locations = {
               "home_position": (0.0, 0.0, 0.0),
               "kitchen": (1.0, 1.0, 0.0),
               "living_room": (2.0, 0.0, 0.0),
               "bedroom": (-1.0, 1.0, 0.0),
               "office": (0.0, -2.0, 1.57)
           }
           
           if location_name in locations:
               x, y, theta = locations[location_name]
               pose = Pose()
               pose.position.x = x
               pose.position.y = y
               pose.position.z = 0.0
               
               # Convert theta to quaternion
               from tf.transformations import quaternion_from_euler
               q = quaternion_from_euler(0, 0, theta)
               pose.orientation.x = q[0]
               pose.orientation.y = q[1]
               pose.orientation.z = q[2]
               pose.orientation.w = q[3]
               
               return pose
           else:
               return None
       
       def verify_preconditions(self, preconditions: List[str]) -> bool:
           """Verify preconditions are met"""
           for condition in preconditions:
               if condition == "robot_free":
                   # Check if robot is not busy
                   if self.plan_execution_active:
                       return False
               elif condition == "object_visible":
                   # This would require perception system integration
                   continue  # For now, assume object is visible
               elif condition == "reachable":
                   # This would require path planning verification
                   continue  # For now, assume reachable
               elif condition == "graspable":
                   # This would require object property verification
                   continue  # For now, assume graspable
           
           return True
       
       def integration_control_callback(self, msg: String):
           """Handle integration control commands"""
           command = msg.data.lower()
           
           if command == "execute_plan":
               # This would trigger execution of current plan
               rospy.loginfo("Integration received execute command")
               # Plan execution initiated in other callbacks
           elif command == "cancel_plan":
               # Cancel current plan execution
               self.plan_execution_active = False
               rospy.loginfo("Plan execution cancelled")
               
               # Cancel any active goals
               self.move_base_client.cancel_all_goals()
               
               # Stop robot
               stop_cmd = Twist()
               self.cmd_vel_pub.publish(stop_cmd)
           elif command == "pause_plan":
               rospy.loginfo("Plan execution paused")
               # In real implementation, would pause execution
           elif command == "resume_plan":
               rospy.loginfo("Plan execution resumed")
               # In real implementation, would resume execution
       
       def run(self):
           """Run the integration node"""
           rospy.loginfo("Planning Integration Node running...")
           rospy.spin()

   def main():
       node = PlanningIntegrationNode()
       node.run()

   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's a complete cognitive planning system that integrates GPT-4o with ROS 2:

```python
#!/usr/bin/env python3
# complete_cognitive_planning_system.py

import rospy
import openai
import json
import threading
import queue
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from actionlib_msgs.msg import GoalStatusArray
from typing import Dict, List, Optional

class CompleteCognitivePlanningSystem:
    def __init__(self, api_key: str):
        rospy.init_node('complete_cognitive_planning_system', anonymous=True)
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = "gpt-4o"
        
        # Publishers
        self.high_level_plan_pub = rospy.Publisher('/cognitive/high_level_plan', String, queue_size=10)
        self.low_level_plan_pub = rospy.Publisher('/cognitive/low_level_plan', String, queue_size=10)
        self.execution_feedback_pub = rospy.Publisher('/cognitive/execution_feedback', String, queue_size=10)
        self.status_pub = rospy.Publisher('/cognitive/status', String, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/cognitive/natural_task', String, self.natural_task_callback)
        rospy.Subscriber('/cognitive/plan_request', String, self.plan_request_callback)
        rospy.Subscriber('/cognitive/execution_status', String, self.execution_status_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        
        # Internal state
        self.laser_data = None
        self.current_plan = None
        self.plan_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        self.is_executing = False
        
        # System parameters
        self.max_planning_retries = 3
        self.planning_timeout = 30.0
        self.execution_timeout = 60.0
        self.safety_distance = 0.5  # meters
        
        # Knowledge base
        self.initialize_knowledge_base()
        
        self.publish_status("System initialized and ready")
        rospy.loginfo("Complete Cognitive Planning System initialized")
    
    def initialize_knowledge_base(self):
        """Initialize the system's knowledge base"""
        self.knowledge_base = {
            "robot_capabilities": {
                "navigation": {
                    "max_linear_speed": 0.5,
                    "max_angular_speed": 1.0,
                    "min_turn_radius": 0.2,
                    "sensor_range": 3.0
                },
                "manipulation": {
                    "max_reach": 0.8,
                    "payload_capacity": 2.0,
                    "gripper_types": ["parallel", "vacuum", "suction"]
                },
                "perception": {
                    "camera_fov": 60.0,  # degrees
                    "depth_range": [0.1, 5.0],  # meters
                    "object_detection_range": [0.2, 3.0]  # meters
                }
            },
            "environment_constraints": [
                "avoid_dynamic_obstacles",
                "maintain_personal_space",
                "respect_doorways",
                "avoid_steep_slopes",
                "stay_on_navigable_surfaces"
            ],
            "task_templates": {
                "navigation": {
                    "actions": ["navigate_to", "explore", "follow_path"],
                    "constraints": ["collision_free", "reachable", "safe_speed"]
                },
                "manipulation": {
                    "actions": ["pick_and_place", "grasp_object", "transport"],
                    "constraints": ["reachable", "graspable", "stable_grasp"]
                },
                "perception": {
                    "actions": ["detect_object", "localize_person", "inspect_area"],
                    "constraints": ["visible", "in_field_of_view", "sufficient_lighting"]
                }
            }
        }
    
    def laser_callback(self, msg: LaserScan):
        """Update laser data for safety checks"""
        self.laser_data = msg
    
    def natural_task_callback(self, msg: String):
        """Handle natural language task requests"""
        try:
            # Parse the task request
            if msg.data.startswith('{'):
                task_request = json.loads(msg.data)
                task_description = task_request["task"]
                priority = task_request.get("priority", "normal")
            else:
                task_description = msg.data
                priority = "normal"
            
            rospy.loginfo(f"Received natural task: {task_description} (Priority: {priority})")
            
            # Validate task request
            if not self.validate_task_request(task_description):
                feedback = {
                    "task_id": "unknown",
                    "status": "invalid_request",
                    "message": "Task request could not be understood",
                    "timestamp": rospy.Time.now().to_sec()
                }
                self.publish_execution_feedback(feedback)
                return
            
            # Add to planning queue
            self.plan_queue.put({
                "task": task_description,
                "priority": priority,
                "request_time": rospy.Time.now().to_sec()
            })
            
            self.publish_status(f"Task received and queued for planning: {task_description[:50]}...")
            
        except json.JSONDecodeError:
            rospy.logerr("Invalid JSON in task request")
            self.publish_feedback("Invalid task request format", level="error")
        except KeyError as e:
            rospy.logerr(f"Missing key in task request: {e}")
    
    def validate_task_request(self, task_description: str) -> bool:
        """Validate if the task request is understandable"""
        # Basic validation - ensure it has content and isn't too generic
        if not task_description or len(task_description.strip()) < 3:
            return False
        
        # Check if it contains actionable verbs
        actionable_verbs = [
            'move', 'navigate', 'go', 'drive', 'turn', 'rotate',  # movement
            'pick', 'grasp', 'grab', 'take', 'place', 'put', 'release',  # manipulation
            'find', 'locate', 'detect', 'see', 'look', 'inspect',  # perception
            'tell', 'say', 'speak', 'inform', 'notify',  # communication
            'help', 'assist', 'fetch', 'bring'  # complex tasks
        ]
        
        task_lower = task_description.lower()
        has_actionable_verb = any(verb in task_lower for verb in actionable_verbs)
        
        # Basic check - task should contain at least one actionable verb
        return has_actionable_verb
    
    def plan_request_callback(self, msg: String):
        """Handle direct plan requests"""
        try:
            plan_request = json.loads(msg.data)
            task_description = plan_request["task"]
            
            # Generate plan directly
            plan = self.generate_cognitive_plan(task_description)
            
            if plan:
                self.publish_hierarchical_plan(plan)
                self.publish_status(f"Plan generated for: {task_description[:30]}...")
            else:
                self.publish_status(f"Plan generation failed for: {task_description[:30]}...")
                
        except json.JSONDecodeError:
            rospy.logerr("Invalid JSON in plan request")
    
    def execution_status_callback(self, msg: String):
        """Handle execution status updates"""
        try:
            status = json.loads(msg.data)
            
            # Update internal execution state
            if status.get("status") == "completed":
                self.is_executing = False
                self.publish_status("Plan execution completed")
            elif status.get("status") == "failed":
                self.is_executing = False
                self.publish_status(f"Plan execution failed: {status.get('error', 'Unknown error')}")
                # Handle failure - maybe generate recovery plan
                self.handle_execution_failure(status)
            elif status.get("status") == "executing":
                self.is_executing = True
                self.publish_status(f"Executing plan: {status.get('task_id', 'unknown')}")
                
        except json.JSONDecodeError:
            rospy.logerr("Invalid JSON in execution status")
    
    def handle_execution_failure(self, status: Dict):
        """Handle plan execution failures"""
        task_id = status.get("task_id", "unknown")
        error = status.get("error", "Unknown error")
        
        rospy.logwarn(f"Handling failure for task {task_id}: {error}")
        
        # Generate recovery plan if possible
        recovery_plan = self.generate_recovery_plan(task_id, error)
        
        if recovery_plan:
            rospy.loginfo("Recovery plan generated, publishing...")
            self.publish_hierarchical_plan(recovery_plan)
            self.publish_execution_feedback({
                "task_id": task_id,
                "status": "recovery_plan_generated",
                "message": "Recovery plan published",
                "timestamp": rospy.Time.now().to_sec()
            })
        else:
            rospy.logerr("Could not generate recovery plan")
            self.publish_execution_feedback({
                "task_id": task_id,
                "status": "recovery_failed",
                "message": "Could not generate recovery plan",
                "timestamp": rospy.Time.now().to_sec()
            })
    
    def generate_recovery_plan(self, task_id: str, error: str) -> Optional[Dict]:
        """Generate a recovery plan after execution failure"""
        # In a real system, this would have access to full plan and error context
        # For now, create a simple recovery strategy
        
        prompt = f"""
        Original Task: {self.current_plan.get('task_description', 'unknown') if self.current_plan else 'unknown'}
        Error Encountered: {error}
        Current Environment Context: {json.dumps(self.get_environment_context(), indent=2)}
        
        Available Recovery Strategies:
        - Retry: Attempt the same action with different parameters
        - Alternative: Use a different approach to achieve the same goal
        - Abort: Cancel the task and return to safe state
        - Ask for Help: Request human intervention
        
        Generate a recovery plan that addresses the specific failure and attempts to continue task completion.
        Provide the plan in the same format as the original plan generation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a robot task recovery planner. Generate recovery plans that address specific failures and attempt task continuation. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith('```'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx]
            
            recovery_plan = json.loads(response_text)
            recovery_plan["task_id"] = f"{task_id}_recovery"
            recovery_plan["timestamp"] = rospy.Time.now().to_sec()
            
            return recovery_plan
            
        except Exception as e:
            rospy.logerr(f"Error generating recovery plan: {e}")
            return None
    
    def generate_cognitive_plan(self, task_description: str) -> Optional[Dict]:
        """Generate cognitive plan using GPT-4o"""
        # Get environment context
        context = self.get_environment_context()
        
        # Create detailed planning prompt
        prompt = f"""
        Task: {task_description}
        
        Environment Context: {json.dumps(context, indent=2)}
        
        Robot Capabilities: {json.dumps(self.knowledge_base["robot_capabilities"], indent=2)}
        Task Templates: {json.dumps(self.knowledge_base["task_templates"], indent=2)}
        Environment Constraints: {json.dumps(self.knowledge_base["environment_constraints"], indent=2)}
        
        Available Actions:
        Navigation Actions:
        - move_to_location(location_name): Navigate to named location
        - navigate_to_object(object_name): Navigate to object position
        - avoid_obstacles(): Execute obstacle avoidance
        - follow_path(waypoints): Follow sequence of coordinates
        - patrol_area(area_name): Navigate through predefined area
        
        Manipulation Actions:
        - grasp_object(object_name): Grasp specified object
        - place_object(object_name, location): Place object at location  
        - transport_object(object_name, from_location, to_location): Move object between locations
        - open_container(container_name): Open container/drawer/door
        - close_container(container_name): Close container/drawer/door
        - handover_object(person_name): Give object to person
        
        Perception Actions:
        - detect_object(object_type): Detect object in environment
        - recognize_object(object_instance): Confirm identity of object
        - localize_person(person_name): Find specific person
        - inspect_object(object_name): Examine object state
        - scan_area(area_name): Survey area for information
        
        Communication Actions:
        - speak_text(text): Speak text aloud
        - play_sound(sound_name): Play predefined sound
        - send_notification(message): Send message to operator
        - request_confirmation(question): Ask for human confirmation
        
        Generate a comprehensive cognitive plan that:
        1. Decomposes the high-level task into a hierarchical structure
        2. Considers environmental context and safety constraints
        3. Respects robot capabilities and limitations
        4. Includes precondition checking before each step
        5. Provides expected outcomes for verification
        6. Incorporates contingency plans for common failures
        7. Estimates execution time and confidence levels
        
        Return in JSON format:
        {{
          "task_id": "unique_identifier_based_on_time",
          "task_description": "{task_description}",
          "plan_type": "cognitive_task_plan",
          "plan_hierarchy": {{
            "high_level_tasks": [
              {{
                "id": "HL_1",
                "name": "task_name",
                "description": "what this high-level task accomplishes",
                "dependencies": ["HL_0"],  // Other high-level tasks this depends on
                "subtasks": ["ST_1_1", "ST_1_2"],  // Subtask IDs that comprise this task
                "estimated_duration": 30.0,
                "confidence": 0.85,
                "success_criteria": ["criteria1", "criteria2"]
              }}
            ],
            "subtasks": [
              {{
                "id": "ST_1_1",
                "parent_id": "HL_1",
                "action_type": "navigation|manipulation|perception|communication",
                "action": "specific_action_name",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "description": "detailed description of this step",
                "preconditions": [
                  "robot_free", 
                  "target_visible", 
                  "safety_clear", 
                  "resources_available"
                ],
                "expected_effects": [
                  "robot_at_location",
                  "object_detected",
                  "task_step_complete"
                ],
                "success_verification": [
                  "check_robot_pose",
                  "verify_object_state",
                  "confirm_completion"
                ],
                "failure_recovery": [
                  "retry_step",
                  "use_alternative_approach",
                  "abort_task"
                ],
                "estimated_duration": 10.0,
                "success_probability": 0.9
              }}
            ]
          }},
          "execution_schema": {{
            "parallelizable_steps": [["ST_1_1", "ST_1_2"]],  // Steps that can run in parallel
            "synchronization_points": ["ST_1_3"],  // Points where parallel steps must synchronize
            "critical_path": ["ST_0_1", "ST_0_2", "ST_0_3"]  // Sequence that determines total duration
          }},
          "safety_checks": [
            "validate_navigation_path",
            "check_manipulation_feasibility",
            "verify_perception_reliability"
          ],
          "resource_utilization": {{
            "navigation": ["time", "battery"],
            "manipulation": ["time", "gripper", "power"],
            "perception": ["time", "compute", "sensors"],
            "communication": ["time", "bandwidth"]
          }},
          "estimated_total_duration": 120.0,
          "overall_confidence": 0.75,
          "risk_assessment": [
            {{"risk": "object_not_found", "likelihood": 0.1, "impact": "medium", "mitigation": "search_alternatives"}},
            {{"risk": "navigation_failure", "likelihood": 0.05, "impact": "high", "mitigation": "replan_path"}},
            {{"risk": "grasp_failure", "likelihood": 0.08, "impact": "low", "mitigation": "retry_different_grasp"}}
          ],
          "validation_procedures": [
            "pre_execution_checklist",
            "during_execution_monitoring",
            "post_execution_verification"
          ],
          "timestamp": current_timestamp
        }}
        
        Only return the JSON plan with no additional text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an advanced robotic cognitive task planner. Generate detailed, executable plans that consider all constraints, capabilities, and environmental factors. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith('```'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx]
            
            plan_data = json.loads(response_text)
            plan_data["timestamp"] = rospy.Time.now().to_sec()
            
            # Validate the generated plan
            if self.validate_plan_structure(plan_data):
                rospy.loginfo(f"Generated cognitive plan with {len(plan_data['plan_hierarchy']['subtasks'])} steps")
                return plan_data
            else:
                rospy.logerr("Generated plan failed structural validation")
                return None
                
        except Exception as e:
            rospy.logerr(f"Error generating cognitive plan: {e}")
            return None
    
    def get_environment_context(self) -> Dict:
        """Get current environment context for planning"""
        context = {
            "robot_state": {
                "position": {"x": 0.0, "y": 0.0, "theta": 0.0},  # Would come from TF or localization
                "status": "idle",
                "battery_level": 0.85,
                "gripper_status": "open",
                "navigation_status": "ready"
            },
            "sensed_environment": {
                "objects": [
                    {"name": "table_1", "type": "furniture", "position": {"x": 1.0, "y": 0.0}, "reachable": True},
                    {"name": "book", "type": "object", "position": {"x": 1.1, "y": 0.1}, "graspable": True},
                    {"name": "chair", "type": "furniture", "position": {"x": 0.0, "y": 1.0}, "reachable": True}
                ],
                "obstacles": self.laser_data.ranges[:50] if self.laser_data else [],  # Sample of front detection
                "safe_zones": ["home_area", "charging_area"],
                "forbidden_zones": []
            },
            "known_locations": {
                "kitchen": {"x": 2.0, "y": 1.0, "theta": 0.0},
                "living_room": {"x": 1.0, "y": -1.0, "theta": 0.0},
                "bedroom": {"x": -1.0, "y": 0.5, "theta": 1.57},
                "office": {"x": 0.0, "y": -2.0, "theta": 3.14},
                "home_position": {"x": 0.0, "y": 0.0, "theta": 0.0},
                "charging_station": {"x": -2.0, "y": -2.0, "theta": 0.0}
            },
            "time_context": {
                "hour_of_day": 14,  # Current hour (24-hour format)
                "day_of_week": "tuesday",
                "month": "december"
            },
            "user_preferences": {
                "preferred_speed": "moderate",
                "safety_preference": "conservative",
                "interaction_mode": "autonomous"
            }
        }
        
        return context
    
    def validate_plan_structure(self, plan: Dict) -> bool:
        """Validate the structure of a generated plan"""
        required_fields = [
            "task_id", "task_description", "plan_hierarchy", 
            "estimated_total_duration", "overall_confidence"
        ]
        
        if not all(field in plan for field in required_fields):
            rospy.logerr("Plan missing required top-level fields")
            return False
        
        hierarchy = plan.get("plan_hierarchy", {})
        required_hierarchy = ["high_level_tasks", "subtasks"]
        
        if not all(field in hierarchy for field in required_hierarchy):
            rospy.logerr("Plan hierarchy missing required components")
            return False
        
        # Validate that subtasks reference valid high-level tasks
        hl_task_ids = {task["id"] for task in hierarchy["high_level_tasks"]}
        for subtask in hierarchy["subtasks"]:
            parent_id = subtask.get("parent_id")
            if parent_id and parent_id not in hl_task_ids:
                rospy.logerr(f"Subtask {subtask.get('id')} references invalid parent {parent_id}")
                return False
        
        return True
    
    def publish_hierarchical_plan(self, plan: Dict):
        """Publish the complete hierarchical plan"""
        # Split plan into high-level and low-level components for different consumers
        high_level = {
            "task_id": plan["task_id"],
            "task_description": plan["task_description"],
            "high_level_tasks": plan["plan_hierarchy"]["high_level_tasks"],
            "execution_schema": plan["execution_schema"],
            "timestamp": plan["timestamp"]
        }
        
        low_level = {
            "task_id": plan["task_id"],
            "subtasks": plan["plan_hierarchy"]["subtasks"],
            "safety_checks": plan["safety_checks"],
            "execution_schema": plan["execution_schema"],
            "timestamp": plan["timestamp"]
        }
        
        # Publish high-level plan
        hl_msg = String()
        hl_msg.data = json.dumps(high_level, indent=2)
        self.high_level_plan_pub.publish(hl_msg)
        
        # Publish low-level plan
        ll_msg = String()
        ll_msg.data = json.dumps(low_level, indent=2)
        self.low_level_plan_pub.publish(ll_msg)
    
    def publish_execution_feedback(self, feedback: Dict):
        """Publish execution feedback"""
        feedback_msg = String()
        feedback_msg.data = json.dumps(feedback, indent=2)
        self.execution_feedback_pub.publish(feedback_msg)
    
    def publish_status(self, status: str):
        """Publish system status"""
        status_msg = String()
        status_msg.data = json.dumps({
            "status": status,
            "timestamp": rospy.Time.now().to_sec()
        })
        self.status_pub.publish(status_msg)
    
    def run(self):
        """Run the cognitive planning system"""
        rospy.loginfo("Starting Complete Cognitive Planning System...")
        
        # Start planning thread for processing queued requests
        planning_thread = threading.Thread(target=self.planning_loop)
        planning_thread.daemon = True
        planning_thread.start()
        
        rospy.spin()
    
    def planning_loop(self):
        """Process planning requests in a separate thread"""
        rate = rospy.Rate(1)  # 1 Hz polling of plan queue
        
        while not rospy.is_shutdown():
            try:
                # Get next planning request
                request = self.plan_queue.get(timeout=1.0)
                
                # Generate plan
                plan = self.generate_cognitive_plan(request["task"])
                
                if plan:
                    self.current_plan = plan
                    self.publish_hierarchical_plan(plan)
                    
                    feedback = {
                        "task_id": plan["task_id"],
                        "status": "plan_generated",
                        "message": "Cognitive plan successfully generated",
                        "timestamp": rospy.Time.now().to_sec()
                    }
                else:
                    # Plan generation failed
                    feedback = {
                        "task_id": "unknown",  # Can't generate ID without plan
                        "status": "generation_failed",
                        "message": "Cognitive plan generation failed",
                        "request_task": request["task"],
                        "timestamp": rospy.Time.now().to_sec()
                    }
                
                self.publish_execution_feedback(feedback)
                self.plan_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Error in planning loop: {e}")
                continue
            
            rate.sleep()

def main():
    api_key = rospy.get_param('~openai_api_key', '')
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    
    if not api_key:
        rospy.logerr("No OpenAI API key provided!")
        return
    
    system = CompleteCognitivePlanningSystem(api_key)
    system.run()

if __name__ == '__main__':
    main()
```

### Launch file for the system:

```xml
<launch>
  <!-- Complete Cognitive Planning System -->
  <node name="complete_cognitive_planner" pkg="robot_planning" type="complete_cognitive_planning_system.py" output="screen">
    <param name="openai_api_key" value=""/>
    <param name="safety_distance" value="0.5"/>
    <param name="max_planning_retries" value="3"/>
  </node>
  
  <!-- Example: Simple robot simulation for testing -->
  <include file="$(find turtlebot3_gazebo)/launch/turtlebot3_empty_world.launch"/>
  
  <!-- TF for robot -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>
  
  <!-- Joint state publisher for simulated joints -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher">
    <param name="use_gui" value="false"/>
  </node>
</launch>
```

## Mini-project

Create a complete cognitive planning system that:

1. Implements GPT-4o-based task decomposition for complex robotic missions
2. Integrates with ROS navigation stack for execution
3. Creates hierarchical task networks with multiple abstraction levels
4. Implements plan monitoring and execution feedback
5. Handles multi-robot coordination scenarios
6. Evaluates plan success and provides learning feedback
7. Implements safety validation for all generated plans
8. Creates a visualization interface for plan monitoring

Your project should include:
- Complete GPT-4o integration for task planning
- Hierarchical task network generation
- ROS integration for plan execution
- Multi-robot coordination mechanisms
- Safety validation system
- Plan monitoring and feedback mechanisms
- Performance evaluation metrics
- Visualization interface

## Summary

This chapter covered cognitive task planning with GPT-4o:

- **Multi-modal Integration**: Fusing information from different sensor modalities
- **Early vs Late Fusion**: Different approaches to combining sensor data
- **Deep Fusion**: Learnable fusion within neural network architectures
- **Transformer-based Fusion**: Attention mechanisms for multi-modal integration
- **Sensor Calibration**: Ensuring proper alignment between modalities
- **Synchronization**: Aligning temporal data from different sensors
- **Fusion Algorithms**: Techniques for combining multi-modal information
- **Uncertainty Handling**: Managing uncertainty in fused perceptual data
- **Performance Evaluation**: Metrics for assessing fusion effectiveness

Multi-modal perception fusion enables robots to develop a more comprehensive understanding of their environment by combining complementary information from multiple sensors. This results in more robust and accurate perception systems that can handle challenging conditions where individual sensors may fail.