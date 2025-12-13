-----
title: Ch17  VisionLanguage Models for Robot Perception
module: 4
chapter: 17
sidebar_label: Ch17: VisionLanguage Models for Robot Perception
description: Integrating VisionLanguage Models with robotics for enhanced perception and understanding
tags: [vlm, visionlanguage, robotics, clip, blip, multimodal, perception]
difficulty: advanced
estimated_duration: 90
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# VisionLanguage Models for Robot Perception

## Learning Outcomes
 Understand VisionLanguage Models (VLMs) and their application in robotics
 Implement CLIPbased visual grounding for robotic systems
 Create multimodal perception systems using visionlanguage fusion
 Integrate VLMs with robotic control systems
 Develop object detection and recognition systems enhanced with language understanding
 Create spatial reasoning systems that combine vision and language
 Evaluate VLM performance in robotic perception tasks
 Optimize VLM inference for realtime robotic applications

## Theory

### VisionLanguage Models Overview

VisionLanguage Models (VLMs) represent a breakthrough in artificial intelligence that combines visual understanding with language comprehension. These models learn joint representations of images and text, enabling them to perform tasks that require both visual and linguistic understanding.

<MermaidDiagram chart={`
graph TD;
    A[VisionLanguage Models] > B[CLIP];
    A > C[BLIP];
    A > D[Flamingo];
    A > E[BLIP2];
    A > F[IDEFICS];
    
    B > G[Multimodal Encoder];
    B > H[Contrastive Learning];
    B > I[ZeroShot Recognition];
    
    C > J[Image Captioning];
    C > K[Visual Question Answering];
    C > L[ImageText Retrieval];
    
    D > M[Sequential Processing];
    D > N[Recursive Generative];
    D > O[Openended VisionLanguage Tasks];
    
    E > P[VisionLanguage Encoder];
    E > Q[Language Model Integration];
    E > R[Generative Capabilities];
    
    F > S[Open Vocabulary];
    F > T[Context Understanding];
    F > U[Instruction Following];
    
    V[Robot Perception] > W[Object Recognition];
    V > X[Scene Understanding];
    V > Y[Visual Grounding];
    V > Z[HumanRobot Interaction];
    
    G > W;
    P > X;
    M > Y;
    S > Z;
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style V fill:#FF9800,stroke:#E65100,color:#fff;
    style W fill:#2196F3,stroke:#0D47A1,color:#fff;
`} />

### CLIP (Contrastive LanguageImage Pretraining)

CLIP represents a paradigm shift in visual recognition by training a model to match images with their corresponding text descriptions. The model learns visual concepts from natural language supervision, enabling zeroshot transfer to downstream tasks.

**Architecture Components:**
 **Image Encoder**: Processes images into embeddings (usually Vision Transformer or ResNet)
 **Text Encoder**: Processes text into embeddings (usually Transformer)
 **Contrastive Loss**: Learns to align matching imagetext pairs while pushing apart nonmatching pairs

**Robotic Applications:**
 Object recognition without taskspecific training
 Visual grounding for manipulation tasks
 Scene understanding and context awareness
 Humanrobot interaction through natural language

### VisionLanguage Grounding

Visionlanguage grounding connects natural language expressions to visual content, enabling robots to understand commands like "pick up the red mug on the left side of the table."

**Spatial Relations Understanding:**
 Relative positioning (left, right, front, behind)
 Spatial containment (inside, on top of, under)
 Sizebased selection (biggest, smallest)
 Color and shapebased filtering

### Multimodal Embedding Spaces

VLMs create shared embedding spaces where visual and textual information can be directly compared:

```
Image_I ∈ R^D ←→ Text_T ∈ R^D
Similarity(I, T) = cosine(Image_I, Text_T)
```

This enables robots to perform zeroshot recognition by comparing visual inputs with textual descriptions without requiring labeled training data for each specific object or scene.

## StepbyStep Labs

### Lab 1: Setting up CLIP Integration with Robotics

1. **Install required dependencies**:
   ```bash
   pip install openaiclip
   pip install transformers torch torchvision torchaudio
   pip install opencvpython
   pip install rospy sensormsgs cvbridge
   ```

2. **Create a CLIPbased object recognizer** (`clip_object_recognizer.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import clip
   import torch
   import cv2
   import numpy as np
   from sensor_msgs.msg import Image
   from std_msgs.msg import String
   from cv_bridge import CvBridge
   from typing import List, Dict, Optional
   from geometry_msgs.msg import Point

   class ClipObjectRecognizer:
       def __init__(self):
           rospy.init_node('clip_object_recognizer', anonymous=True)
           
           # Load CLIP model
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.model, self.preprocess = clip.load("ViTB/32", device=self.device)
           self.model.eval()
           
           # Initialize OpenCV bridge
           self.cv_bridge = CvBridge()
           
           # Publishers and Subscribers
           self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
           self.recognition_pub = rospy.Publisher("/clip_recognition_results", String, queue_size=10)
           self.annotations_pub = rospy.Publisher("/annotated_image", Image, queue_size=10)
           
           # Object detection parameters
           self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
           self.top_k = rospy.get_param("~top_k", 5)
           
           # Default object categories
           self.default_categories = [
               "person", "bottle", "cup", "fork", "knife", 
               "spoon", "bowl", "banana", "apple", "orange",
               "cake", "chair", "couch", "potted plant", "bed",
               "dining table", "toilet", "tv", "laptop", "mouse",
               "remote", "keyboard", "cell phone", "microwave", "oven",
               "toaster", "sink", "refrigerator", "book", "clock",
               "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
           ]
           
           rospy.loginfo("CLIP Object Recognizer initialized")

       def image_callback(self, msg: Image):
           """Process incoming RGB image"""
           try:
               # Convert ROS Image to OpenCV format
               cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
               
               # Run CLIPbased object recognition
               results = self.recognize_objects(cv_image, self.default_categories)
               
               # Publish results
               result_msg = String()
               result_msg.data = str(results)
               self.recognition_pub.publish(result_msg)
               
               # Create annotated image
               annotated_image = self.annotate_image(cv_image, results)
               annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, "bgr8")
               annotated_msg.header = msg.header
               self.annotations_pub.publish(annotated_msg)
               
           except Exception as e:
               rospy.logerr(f"Error processing image: {e}")

       def recognize_objects(self, image: np.ndarray, categories: List[str]) > Dict:
           """Recognize objects using CLIP"""
           # Preprocess image
           image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
           
           # Tokenize text descriptions
           text_descriptions = [f"a photo of a {cat}" for cat in categories]
           text_tokens = clip.tokenize(text_descriptions).to(self.device)
           
           # Get model predictions
           with torch.no_grad():
               logits_per_image, logits_per_text = self.model(image_tensor, text_tokens)
               probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
           
           # Get topk predictions
           top_indices = np.argsort(probs)[self.top_k:][::1]
           
           results = []
           for idx in top_indices:
               if probs[idx] >= self.confidence_threshold:
                   results.append({
                       "category": categories[idx],
                       "confidence": float(probs[idx]),
                       "description": text_descriptions[idx]
                   })
           
           return {
               "predictions": results,
               "image_shape": image.shape,
               "timestamp": rospy.Time.now().to_sec()
           }

       def annotate_image(self, image: np.ndarray, results: Dict) > np.ndarray:
           """Add annotation text to image"""
           annotated = image.copy()
           height, width = image.shape[:2]
           
           # Display top predictions
           for i, pred in enumerate(results["predictions"]):
               y_pos = 30 + i * 30
               text = f"{pred['category']}: {pred['confidence']:.2f}"
               
               # Add background rectangle
               text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
               cv2.rectangle(annotated, 
                           (10, y_pos  text_size[1]  10), 
                           (10 + text_size[0], y_pos + 10), 
                           (0, 0, 0), 1)
               
               # Add text
               cv2.putText(annotated, text, (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
           
           return annotated

   def main():
       recognizer = ClipObjectRecognizer()
       rospy.spin()

   if __name__ == '__main__':
       main()
   ```

### Lab 2: Implementing Visual Grounding with Spatial Context

1. **Create a visual grounding system** (`visual_grounding_system.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import clip
   import torch
   import cv2
   import numpy as np
   import openai
   from sensor_msgs.msg import Image
   from std_msgs.msg import String
   from geometry_msgs.msg import Point
   from cv_bridge import CvBridge
   from typing import Dict, List, Optional, Tuple
   import json
   import re

   class VisualGroundingSystem:
       def __init__(self):
           rospy.init_node('visual_grounding_system', anonymous=True)
           
           # Load models
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.clip_model, self.clip_preprocess = clip.load("ViTB/32", device=self.device)
           self.clip_model.eval()
           
           # Initialize LLM for spatial reasoning
           self.openai_client = None
           
           # Initialize OpenCV bridge
           self.cv_bridge = CvBridge()
           
           # Publishers and Subscribers
           self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
           self.command_sub = rospy.Subscriber("/natural_language_command", String, self.command_callback)
           self.grounding_pub = rospy.Publisher("/visual_grounding_results", String, queue_size=10)
           self.region_proposal_pub = rospy.Publisher("/region_proposals", String, queue_size=10)
           
           # Internal state
           self.current_image = None
           self.current_objects = []
           self.current_command = None
           
           # Spatial relation definitions
           self.spatial_relations = {
               "left": ["left of", "to the left of", "left side", "west of"],
               "right": ["right of", "to the right of", "right side", "east of"],
               "above": ["above", "on top of", "over", "up from", "north of"],
               "below": ["below", "under", "beneath", "down from", "south of"],
               "between": ["between", "in between", "in the middle of"],
               "behind": ["behind", "at the back of", "in back of"],
               "in_front_of": ["in front of", "before", "ahead of"],
               "inside": ["inside", "within", "in", "contained by"],
               "outside": ["outside", "beyond", "external to"],
               "near": ["near", "close to", "by", "next to", "beside"]
           }
           
           rospy.loginfo("Visual Grounding System initialized")

       def image_callback(self, msg: Image):
           """Store current image for grounding operations"""
           try:
               self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
               
               # Extract objects from current image using CLIP
               self.current_objects = self.extract_objects_in_image(self.current_image)
               
           except Exception as e:
               rospy.logerr(f"Error processing image: {e}")

       def command_callback(self, msg: String):
           """Process natural language command for visual grounding"""
           self.current_command = msg.data
           rospy.loginfo(f"Processing command: {msg.data}")
           
           if self.current_image is not None:
               result = self.ground_command_in_image(
                   self.current_command, 
                   self.current_image, 
                   self.current_objects
               )
               
               # Publish grounding result
               result_msg = String()
               result_msg.data = json.dumps(result, indent=2)
               self.grounding_pub.publish(result_msg)
               
               rospy.loginfo(f"Grounding result: {result}")

       def extract_objects_in_image(self, image: np.ndarray) > List[Dict]:
           """Extract objects using CLIP zeroshot classification"""
           # This is a simplified object extraction  in practice, you'd use
           # more sophisticated methods like object detection or segmentation
           
           common_objects = [
               "person", "bottle", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "orange", "cake", "chair", "couch", 
               "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
               "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
               "toaster", "sink", "refrigerator", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush", "door",
               "window", "cupboard", "shelf", "cabinet", "box", "bag"
           ]
           
           # Get image features
           image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
           text_descriptions = [f"a photo of a {obj}" for obj in common_objects]
           text_tokens = clip.tokenize(text_descriptions).to(self.device)
           
           with torch.no_grad():
               logits_per_image, _ = self.clip_model(image_tensor, text_tokens)
               probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
           
           # Get objects with confidence above threshold
           threshold = 0.1
           detected_objects = []
           for i, prob in enumerate(probs):
               if prob > threshold:
                   # For this example, we'll add bounding boxes that represent
                   # where we think the object might be located
                   bbox = self.estimate_object_bbox(image, common_objects[i])
                   detected_objects.append({
                       "name": common_objects[i],
                       "confidence": float(prob),
                       "bbox": bbox,  # [x, y, width, height]
                       "center": [(bbox[0] + bbox[2]/2), (bbox[1] + bbox[3]/2)]  # Center coordinates
                   })
           
           return detected_objects

       def estimate_object_bbox(self, image: np.ndarray, object_name: str) > List[int]:
           """Estimate bounding box for an object in image"""
           # This is a placeholder  in a real implementation, you'd use
           # object detection models (YOLO, SSD, etc.) to get actual bounding boxes
           height, width = image.shape[:2]
           
           # For demonstration, return a random bounding box
           # In practice, you'd use actual detection results
           x = np.random.randint(0, width // 2)
           y = np.random.randint(0, height // 2)
           w = np.random.randint(width // 4, width // 2)
           h = np.random.randint(height // 4, height // 2)
           
           return [int(x), int(y), int(w), int(h)]

       def ground_command_in_image(self, command: str, image: np.ndarray, objects: List[Dict]) > Dict:
           """Ground natural language command in visual scene"""
           # Parse spatial relations from command
           spatial_relations = self.parse_spatial_relations(command)
           
           if spatial_relations:
               # Use spatial reasoning to find target object
               target_object = self.resolve_spatial_query(spatial_relations, objects)
           else:
               # Simple object search
               target_object = self.search_for_object(command, objects)
           
           # Generate region proposals for the target
           region_proposals = self.generate_region_proposals(target_object, image)
           
           # Create result
           result = {
               "command": command,
               "spatial_relations": spatial_relations,
               "target_object": target_object,
               "region_proposals": region_proposals,
               "all_detected_objects": objects,
               "timestamp": rospy.Time.now().to_sec()
           }
           
           return result

       def parse_spatial_relations(self, command: str) > List[Dict]:
           """Parse spatial relations from natural language command"""
           relations = []
           command_lower = command.lower()
           
           for relation_type, phrases in self.spatial_relations.items():
               for phrase in phrases:
                   if phrase in command_lower:
                       # Extract potential object references
                       # This is simplified  a complete implementation would
                       # use more sophisticated NLP parsing
                       parts = command_lower.split(phrase)
                       if len(parts) > 1:
                           subject = self.extract_subject(parts[0])
                           object_ref = self.extract_object_reference(parts[1])
                           
                           relations.append({
                               "relation_type": relation_type,
                               "phrase": phrase,
                               "subject": subject,
                               "reference_object": object_ref,
                               "full_phrase": f"{subject} {phrase} {object_ref}"
                           })
           
           return relations

       def extract_subject(self, text: str) > str:
           """Extract subject/object noun phrase"""
           # Simplified noun extraction
           # In practice, use NLP parsing (spaCy, NLTK)
           words = text.strip().split()
           if words:
               return words[1]  # Last word as potential object
           return ""

       def extract_object_reference(self, text: str) > str:
           """Extract object reference from text"""
           # Simplified object reference extraction
           words = text.strip().split()
           if words:
               return words[0]  # First word as potential reference
           return ""

       def resolve_spatial_query(self, relations: List[Dict], objects: List[Dict]) > Optional[Dict]:
           """Resolve spatial query to find target object"""
           for rel in relations:
               subject = rel.get("subject", "")
               reference_obj = rel.get("reference_object", "")
               relation_type = rel.get("relation_type", "")
               
               # Find reference object
               ref_obj = self.find_object_by_name(reference_obj, objects)
               if not ref_obj:
                   continue
               
               # Find subject object based on spatial relation
               target_obj = self.find_object_by_spatial_relation(
                   subject, ref_obj, relation_type, objects
               )
               
               if target_obj:
                   return target_obj
           
           return None

       def find_object_by_name(self, name: str, objects: List[Dict]) > Optional[Dict]:
           """Find object by name in objects list"""
           for obj in objects:
               if name.lower() in obj["name"].lower():
                   return obj
           return None

       def find_object_by_spatial_relation(
           self, target_name: str, reference_obj: Dict, relation: str, objects: List[Dict]
       ) > Optional[Dict]:
           """Find object based on spatial relation to reference object"""
           ref_center = reference_obj["center"]
           
           # Filter candidates by name if target name is specified
           candidates = objects
           if target_name:
               candidates = [obj for obj in objects 
                           if target_name.lower() in obj["name"].lower()]
           
           # Apply spatial relation filtering
           filtered_candidates = []
           for obj in candidates:
               if obj == reference_obj:  # Skip the reference object itself
                   continue
               
               obj_center = obj["center"]
               x_diff = obj_center[0]  ref_center[0]
               y_diff = obj_center[1]  ref_center[1]
               
               # Apply spatial relation filter
               if self.meets_spatial_criteria(x_diff, y_diff, relation):
                   filtered_candidates.append(obj)
           
           # Return the candidate with highest confidence
           if filtered_candidates:
               return max(filtered_candidates, key=lambda x: x["confidence"])
           
           return None

       def meets_spatial_criteria(self, x_diff: float, y_diff: float, relation: str) > bool:
           """Check if spatial relationship is satisfied"""
           EPSILON = 20  # Pixel tolerance for spatial relations
           
           if relation == "left":
               return x_diff < EPSILON
           elif relation == "right":
               return x_diff > EPSILON
           elif relation == "above" or relation == "over":
               return y_diff < EPSILON
           elif relation == "below" or relation == "under":
               return y_diff > EPSILON
           elif relation == "near" or relation == "close to":
               distance = np.sqrt(x_diff**2 + y_diff**2)
               return distance < 100  # Within 100 pixels
           elif relation == "between":
               # This would need more complex logic involving multiple reference points
               return True  # Placeholder
           else:
               return True  # Default to true for other relations

       def generate_region_proposals(self, target_object: Optional[Dict], image: np.ndarray) > List[Dict]:
           """Generate region proposals for target object"""
           if not target_object:
               return []
           
           bbox = target_object["bbox"]
           center = target_object["center"]
           proposals = []
           
           # Primary region (object bounding box)
           proposals.append({
               "type": "primary",
               "bbox": bbox,
               "center": center,
               "confidence": target_object["confidence"],
               "object_name": target_object["name"]
           })
           
           # Extended region (for context)
           extended_bbox = [
               max(0, bbox[0]  50),  # x
               max(0, bbox[1]  50),  # y
               min(image.shape[1]  bbox[0] + 100, bbox[2] + 100),  # width
               min(image.shape[0]  bbox[1] + 100, bbox[3] + 100)   # height
           ]
           
           proposals.append({
               "type": "extended",
               "bbox": extended_bbox,
               "center": center,
               "confidence": target_object["confidence"] * 0.8,  # Lower confidence for extended region
               "object_name": target_object["name"],
               "is_context": True
           })
           
           return proposals

       def search_for_object(self, command: str, objects: List[Dict]) > Optional[Dict]:
           """Search for object directly referenced in command"""
           command_lower = command.lower()
           
           for obj in objects:
               if obj["name"].lower() in command_lower:
                   return obj
           
           # If no exact match, find object with highest relevance
           best_match = None
           best_score = 0
           
           for obj in objects:
               score = self.calculate_text_similarity(command_lower, obj["name"].lower())
               if score > best_score:
                   best_score = score
                   best_match = obj
           
           return best_match

       def calculate_text_similarity(self, text1: str, text2: str) > float:
           """Calculate simple text similarity"""
           words1 = set(text1.split())
           words2 = set(text2.split())
           
           intersection = words1.intersection(words2)
           union = words1.union(words2)
           
           if len(union) == 0:
               return 0.0
           
           return len(intersection) / len(union)

   def main():
       system = VisualGroundingSystem()
       rospy.spin()

   if __name__ == '__main__':
       main()
   ```

### Lab 3: Creating a Multimodal Perception Pipeline

1. **Create a complete multimodal perception pipeline** (`multimodal_perception_pipeline.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import clip
   import torch
   import cv2
   import numpy as np
   import openai
   from sensor_msgs.msg import Image, CameraInfo, PointCloud2
   from std_msgs.msg import String
   from geometry_msgs.msg import Point, Pose, PoseStamped
   from cv_bridge import CvBridge
   from typing import Dict, List, Optional, Tuple
   import json
   import threading
   import queue
   from dataclasses import dataclass

   @dataclass
   class PerceptionResult:
       """Data structure for perception results"""
       timestamp: float
       objects: List[Dict]
       spatial_relations: List[Dict]
       scene_description: str
       confidence: float
       regions_of_interest: List[Dict]

   class MultimodalPerceptionPipeline:
       def __init__(self):
           rospy.init_node('multimodal_perception_pipeline', anonymous=True)
           
           # Initialize models
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.clip_model, self.clip_preprocess = clip.load("ViTB/32", device=self.device)
           self.clip_model.eval()
           
           # Initialize OpenCV bridge
           self.cv_bridge = CvBridge()
           
           # Publishers and Subscribers
           self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
           self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
           self.command_sub = rospy.Subscriber("/natural_language_query", String, self.query_callback)
           
           self.perception_pub = rospy.Publisher("/multimodal_perception", String, queue_size=10)
           self.roi_pub = rospy.Publisher("/regions_of_interest", String, queue_size=10)
           self.scene_description_pub = rospy.Publisher("/scene_description", String, queue_size=10)
           
           # Data storage
           self.current_image = None
           self.current_depth = None
           self.current_objects = []
           self.perception_queue = queue.Queue()
           self.processing_thread = None
           self.is_running = False
           
           # Parameters
           self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.1)
           self.spatial_threshold = rospy.get_param("~spatial_threshold", 0.2)
           
           # Start processing thread
           self.start_processing_pipeline()
           
           rospy.loginfo("Multimodal Perception Pipeline initialized")

       def start_processing_pipeline(self):
           """Start the multimodal processing pipeline"""
           self.is_running = True
           self.processing_thread = threading.Thread(target=self.processing_loop)
           self.processing_thread.daemon = True
           self.processing_thread.start()

       def stop_processing_pipeline(self):
           """Stop the multimodal processing pipeline"""
           self.is_running = False
           if self.processing_thread:
               self.processing_thread.join()

       def image_callback(self, msg: Image):
           """Process incoming image data"""
           try:
               cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
               self.current_image = cv_image
           except Exception as e:
               rospy.logerr(f"Error processing image: {e}")

       def depth_callback(self, msg: Image):
           """Process incoming depth data"""
           try:
               cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
               self.current_depth = cv_depth
           except Exception as e:
               rospy.logerr(f"Error processing depth: {e}")

       def query_callback(self, msg: String):
           """Process natural language query for multimodal analysis"""
           query = msg.data
           rospy.loginfo(f"Processing query: {query}")
           
           if self.current_image is not None:
               # Create a perception job
               job = {
                   "query": query,
                   "image": self.current_image,
                   "depth": self.current_depth,
                   "timestamp": rospy.Time.now().to_sec()
               }
               
               # Add to processing queue
               try:
                   self.perception_queue.put_nowait(job)
               except queue.Full:
                   rospy.logwarn("Perception queue is full, dropping query")

       def processing_loop(self):
           """Main processing loop for multimodal perception"""
           rate = rospy.Rate(5)  # Process at most 5 Hz to avoid overwhelming
           
           while self.is_running and not rospy.is_shutdown():
               try:
                   # Check for new jobs
                   job = self.perception_queue.get(timeout=0.1)
                   
                   # Process the job
                   result = self.process_multimodal_query(
                       job["query"],
                       job["image"],
                       job["depth"]
                   )
                   
                   if result:
                       # Publish results
                       self.publish_perception_result(result)
                       
               except queue.Empty:
                   # No new jobs, continue loop
                   pass
               except Exception as e:
                   rospy.logerr(f"Error in processing loop: {e}")
               
               rate.sleep()

       def process_multimodal_query(self, query: str, image: np.ndarray, depth: Optional[np.ndarray] = None) > Optional[PerceptionResult]:
           """Process a multimodal query"""
           try:
               # Extract objects using CLIP
               objects = self.extract_objects_with_clip(image)
               
               # Analyze spatial relationships
               spatial_relations = self.analyze_spatial_relations(objects)
               
               # Generate scene description using LLM
               scene_description = self.generate_scene_description(query, objects, spatial_relations)
               
               # Identify regions of interest
               roi = self.identify_regions_of_interest(query, objects, spatial_relations)
               
               # Calculate overall confidence
               confidence = self.calculate_overall_confidence(objects, spatial_relations)
               
               result = PerceptionResult(
                   timestamp=rospy.Time.now().to_sec(),
                   objects=objects,
                   spatial_relations=spatial_relations,
                   scene_description=scene_description,
                   confidence=confidence,
                   regions_of_interest=roi
               )
               
               return result
               
           except Exception as e:
               rospy.logerr(f"Error processing multimodal query: {e}")
               return None

       def extract_objects_with_clip(self, image: np.ndarray) > List[Dict]:
           """Extract objects using CLIP zeroshot classification"""
           # Common object categories for household/industrial environments
           categories = [
               "person", "bottle", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "orange", "cake", "chair", "couch", 
               "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
               "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
               "toaster", "sink", "refrigerator", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush", "door",
               "window", "cupboard", "shelf", "cabinet", "box", "bag",
               "robot", "humanoid", "arm", "leg", "wheel", "sensor", "lidar",
               "camera", "display", "button", "lever", "knob", "handle",
               "cord", "pipe", "wire", "cable", "structure", "surface"
           ]
           
           # Preprocess image
           image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
           text_descriptions = [f"a photo of a {cat}" for cat in categories]
           text_tokens = clip.tokenize(text_descriptions).to(self.device)
           
           # Get predictions
           with torch.no_grad():
               logits_per_image, _ = self.clip_model(image_tensor, text_tokens)
               probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
           
           # Get objects above confidence threshold
           detected_objects = []
           for i, prob in enumerate(probs):
               if prob > self.confidence_threshold:
                   # Estimate bounding box and position
                   bbox = self.estimate_object_bbox_with_clip(categories[i], image)
                   detected_objects.append({
                       "name": categories[i],
                       "confidence": float(prob),
                       "bbox": bbox,
                       "center": [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2],
                       "description": text_descriptions[i]
                   })
           
           # Sort by confidence (highest first)
           detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
           return detected_objects

       def estimate_object_bbox_with_clip(self, object_name: str, image: np.ndarray) > List[int]:
           """Estimate bounding box for object using CLIPguided approach"""
           # This is a simplified approach  in practice you'd use
           # more sophisticated techniques like sliding window or attention visualization
           height, width = image.shape[:2]
           
           # For this example, we'll use a simple heuristic based on object size
           # In reality, you'd use attention maps or other methods to localize objects
           
           # Define typical object sizes (as percentage of image)
           object_size_mapping = {
               "person": (0.3, 0.8),  # height range
               "chair": (0.2, 0.5),
               "table": (0.4, 0.9),
               "couch": (0.3, 0.7),
               "bottle": (0.1, 0.3),
               "cup": (0.05, 0.15),
               "laptop": (0.2, 0.3),
               "phone": (0.05, 0.1),
               "door": (0.6, 0.9),
               "window": (0.1, 0.8)
           }
           
           typical_height_range = object_size_mapping.get(object_name, (0.1, 0.4))
           typical_width_range = object_size_mapping.get(object_name, (0.1, 0.4))
           
           # Random position but within reasonable ranges
           h_pct = np.random.uniform(typical_height_range[0], typical_height_range[1])
           w_pct = np.random.uniform(typical_width_range[0], typical_width_range[1])
           
           h = int(h_pct * height)
           w = int(w_pct * width)
           
           x = np.random.randint(0, width  w)
           y = np.random.randint(0, height  h)
           
           return [int(x), int(y), int(w), int(h)]

       def analyze_spatial_relations(self, objects: List[Dict]) > List[Dict]:
           """Analyze spatial relationships between objects"""
           relations = []
           
           for i, obj1 in enumerate(objects):
               for j, obj2 in enumerate(objects):
                   if i != j and obj1["confidence"] > 0.3 and obj2["confidence"] > 0.3:
                       # Calculate spatial relationship
                       rel = self.calculate_spatial_relationship(obj1, obj2)
                       if rel:
                           relations.append(rel)
           
           return relations

       def calculate_spatial_relationship(self, obj1: Dict, obj2: Dict) > Optional[Dict]:
           """Calculate spatial relationship between two objects"""
           center1 = obj1["center"]
           center2 = obj2["center"]
           
           x_diff = center1[0]  center2[0]
           y_diff = center1[1]  center2[1]
           distance = np.sqrt(x_diff**2 + y_diff**2)
           
           # Determine spatial relation based on relative positions
           if abs(x_diff) > abs(y_diff):
               if x_diff > 0:
                   direction = "right"
                   relation = f"{obj1['name']} is to the right of {obj2['name']}"
               else:
                   direction = "left" 
                   relation = f"{obj1['name']} is to the left of {obj2['name']}"
           else:
               if y_diff > 0:
                   direction = "below"
                   relation = f"{obj1['name']} is below {obj2['name']}"
               else:
                   direction = "above"
                   relation = f"{obj1['name']} is above {obj2['name']}"
           
           # Check proximity
           proximity = "near" if distance < 100 else "far"
           
           return {
               "subject": obj1["name"],
               "relation": direction,
               "object": obj2["name"],
               "distance_pixels": distance,
               "proximity": proximity,
               "description": relation,
               "confidence": min(obj1["confidence"], obj2["confidence"]) * 0.8
           }

       def generate_scene_description(self, query: str, objects: List[Dict], spatial_relations: List[Dict]) > str:
           """Generate scene description using LLM"""
           # This would use an LLM in practice
           # For now, create a simple description
           if not objects:
               return "The scene appears to be empty or no objects were detected."
           
           # Create object summary
           obj_names = [obj["name"] for obj in objects[:5]]  # Top 5 objects
           obj_list = ", ".join(obj_names)
           
           # Create spatial summary
           spatial_descriptions = []
           for rel in spatial_relations[:3]:  # Top 3 relations
               spatial_descriptions.append(rel["description"])
           
           spatial_summary = "; ".join(spatial_descriptions) if spatial_descriptions else "Objects appear distributed without clear spatial relationships."
           
           description = f"This scene contains: {obj_list}. {spatial_summary}"
           
           return description

       def identify_regions_of_interest(self, query: str, objects: List[Dict], spatial_relations: List[Dict]) > List[Dict]:
           """Identify regions of interest based on query"""
           roi = []
           
           # Simple approach: match query keywords to objects
           query_lower = query.lower()
           
           for obj in objects:
               if any(keyword in query_lower for keyword in obj["name"].split()):
                   roi.append({
                       "object_name": obj["name"],
                       "bbox": obj["bbox"],
                       "confidence": obj["confidence"],
                       "reason": f"Matches query keyword in '{obj['name']}'"
                   })
           
           # If no direct matches, return top objects
           if not roi and objects:
               for obj in objects[:3]:  # Top 3 objects
                   roi.append({
                       "object_name": obj["name"],
                       "bbox": obj["bbox"],
                       "confidence": obj["confidence"],
                       "reason": f"High confidence object: {obj['confidence']:.2f}"
                   })
           
           return roi

       def calculate_overall_confidence(self, objects: List[Dict], spatial_relations: List[Dict]) > float:
           """Calculate overall perception confidence"""
           if not objects:
               return 0.0
           
           # Average of object confidences weighted by number
           object_confidence = sum(obj["confidence"] for obj in objects) / len(objects) if objects else 0.0
           
           # Spatial relation confidence (if any exist)
           spatial_confidence = sum(rel["confidence"] for rel in spatial_relations) / len(spatial_relations) if spatial_relations else 0.0
           
           # Weighted combination
           overall_conf = (0.7 * object_confidence + 0.3 * spatial_confidence)
           
           return min(overall_conf, 1.0)  # Clamp to [0, 1]

       def publish_perception_result(self, result: PerceptionResult):
           """Publish perception results"""
           # Publish main perception result
           perception_msg = String()
           perception_msg.data = json.dumps({
               "timestamp": result.timestamp,
               "objects": result.objects,
               "spatial_relations": result.spatial_relations,
               "scene_description": result.scene_description,
               "confidence": result.confidence,
               "query_time": rospy.Time.now().to_sec()
           }, indent=2)
           self.perception_pub.publish(perception_msg)
           
           # Publish regions of interest
           roi_msg = String()
           roi_msg.data = json.dumps({
               "timestamp": result.timestamp,
               "regions_of_interest": result.regions_of_interest,
               "total_objects": len(result.objects)
           })
           self.roi_pub.publish(roi_msg)
           
           # Publish scene description separately
           description_msg = String()
           description_msg.data = result.scene_description
           self.scene_description_pub.publish(description_msg)

   def main():
       pipeline = MultimodalPerceptionPipeline()
       
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down multimodal perception pipeline...")
           pipeline.stop_processing_pipeline()

   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's a complete example that demonstrates the integration of all components:

```python
#!/usr/bin/env python3
# complete_vlm_robot_perception.py

import rospy
import clip
import torch
import cv2
import numpy as np
import openai
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from typing import Dict, List, Optional
import json
import time
from multimodal_perception_pipeline import MultimodalPerceptionPipeline
from visual_grounding_system import VisualGroundingSystem
from clip_object_recognizer import ClipObjectRecognizer

class CompleteVLMPipeline:
   """Complete VisionLanguage Model pipeline for robotic perception"""
   
   def __init__(self):
       rospy.init_node('complete_vlm_pipeline', anonymous=True)
       
       # Initialize individual components
       self.multimodal_pipeline = MultimodalPerceptionPipeline()
       self.visual_grounding = VisualGroundingSystem()
       self.clip_recognizer = ClipObjectRecognizer()
       
       # Publishers
       self.system_status_pub = rospy.Publisher('/vlm_system_status', String, queue_size=10)
       self.integrated_result_pub = rospy.Publisher('/integrated_vlm_result', String, queue_size=10)
       
       # Subscribers
       rospy.Subscriber('/robot_command', String, self.robot_command_callback)
       rospy.Subscriber('/vlm_perception', String, self.perception_result_callback)
       rospy.Subscriber('/visual_grounding', String, self.grounding_result_callback)
       
       # System state
       self.latest_perception = None
       self.latest_grounding = None
       self.system_ready = True
       
       rospy.loginfo("Complete VLM Robotics Pipeline initialized")
   
   def robot_command_callback(self, msg: String):
       """Handle robot commands that require VLM processing"""
       command = msg.data
       rospy.loginfo(f"Processing robot command: {command}")
       
       # Update system status
       status_msg = String()
       status_msg.data = f"Processing command: {command}"
       self.system_status_pub.publish(status_msg)
       
       # Determine command type and route appropriately
       if self.contains_visual_reference(command):
           # This command likely requires visual grounding
           self.route_to_grounding(command)
       else:
           # This command likely requires general scene understanding
           self.route_to_perception(command)
   
   def contains_visual_reference(self, command: str) > bool:
       """Check if command contains visual references"""
       visual_keywords = [
           'the', 'that', 'there', 'left', 'right', 'front', 'behind', 
           'above', 'below', 'on', 'in', 'near', 'next to', 'beside',
           'red', 'blue', 'green', 'large', 'small', 'big', 'little'
       ]
       
       command_lower = command.lower()
       return any(keyword in command_lower for keyword in visual_keywords)
   
   def route_to_grounding(self, command: str):
       """Route command to visual grounding system"""
       # Publish to visual grounding system
       cmd_msg = String()
       cmd_msg.data = command
       self.grounding_command_pub.publish(cmd_msg)
   
   def route_to_perception(self, command: str):
       """Route command to perception system"""
       # For general perception, we'll use the command to focus attention
       # Publish to perception system along with current context
       context_msg = String()
       context_msg.data = json.dumps({
           "command": command,
           "focus_attention": self.extract_attention_targets(command)
       })
       self.perception_context_pub.publish(context_msg)
   
   def extract_attention_targets(self, command: str) > List[str]:
       """Extract potential attention targets from command"""
       # Simple keyword extraction  in practice, use more sophisticated NLP
       words = command.lower().split()
       targets = []
       
       # Common object words that might be targets
       potential_targets = [
           'object', 'item', 'thing', 'robot', 'person', 'table', 'chair',
           'bottle', 'cup', 'box', 'door', 'window', 'cabinet', 'shelf'
       ]
       
       for word in words:
           # Remove punctuation
           clean_word = word.strip('.,!?;:')
           if clean_word in potential_targets:
               targets.append(clean_word)
       
       # Also look for color + object combinations
       for i in range(len(words)1):
           if words[i].lower() in ['red', 'blue', 'green', 'yellow', 'black', 'white'] and words[i+1] in potential_targets:
               targets.append(f"{words[i]} {words[i+1]}")
       
       return targets
   
   def perception_result_callback(self, msg: String):
       """Handle perception results"""
       try:
           result = json.loads(msg.data)
           self.latest_perception = result
           
           # Integrate with other modalities
           integrated_result = self.integrate_perception_results()
           
           if integrated_result:
               self.publish_integrated_result(integrated_result)
               
       except json.JSONDecodeError:
           rospy.logerr("Error decoding perception result")
   
   def grounding_result_callback(self, msg: String):
       """Handle visual grounding results"""
       try:
           result = json.loads(msg.data)
           self.latest_grounding = result
           
           # Integrate with other modalities
           integrated_result = self.integrate_grounding_results()
           
           if integrated_result:
               self.publish_integrated_result(integrated_result)
               
       except json.JSONDecodeError:
           rospy.logerr("Error decoding grounding result")
   
   def integrate_perception_results(self) > Optional[Dict]:
       """Integrate perception results with system state"""
       if not self.latest_perception:
           return None
       
       integrated_result = {
           "timestamp": time.time(),
           "source": "perception",
           "data": self.latest_perception,
           "system_confidence": self.calculate_system_confidence(),
           "recommendations": self.generate_recommendations(self.latest_perception)
       }
       
       return integrated_result
   
   def integrate_grounding_results(self) > Optional[Dict]:
       """Integrate grounding results with system state"""
       if not self.latest_grounding:
           return None
       
       integrated_result = {
           "timestamp": time.time(),
           "source": "grounding",
           "data": self.latest_grounding,
           "system_confidence": self.calculate_system_confidence(),
           "actionables": self.extract_actionable_information(self.latest_grounding)
       }
       
       return integrated_result
   
   def calculate_system_confidence(self) > float:
       """Calculate overall system confidence"""
       # This is a simplified confidence calculation
       # In practice, use more sophisticated methods
       confidence_components = []
       
       if self.latest_perception:
           confidence_components.append(self.latest_perception.get("confidence", 0.5))
       
       if self.latest_grounding:
           # Grounding confidence might come from spatial relation confidence
           relations = self.latest_grounding.get("spatial_relations", [])
           if relations:
               avg_conf = sum(r.get("confidence", 0.5) for r in relations) / len(relations)
               confidence_components.append(avg_conf)
           else:
               confidence_components.append(0.3)  # Lower confidence if no spatial relations
       
       if confidence_components:
           avg_confidence = sum(confidence_components) / len(confidence_components)
           return min(avg_confidence, 1.0)
       else:
           return 0.5  # Default confidence
   
   def generate_recommendations(self, perception_data: Dict) > List[str]:
       """Generate recommendations based on perception data"""
       recommendations = []
       objects = perception_data.get("objects", [])
       
       if len(objects) > 5:
           recommendations.append("Scene is complex with many objects detected")
       
       if any(obj["confidence"] > 0.9 for obj in objects):
           recommendations.append("Highconfidence object detections available")
       
       if any("person" in obj["name"] for obj in objects):
           recommendations.append("Human detected  consider social interaction protocols")
       
       # Add other recommendation logic based on objects detected
       manipulable_objects = [
           obj for obj in objects 
           if obj["name"] in ["bottle", "cup", "box", "book", "phone", "laptop"]
       ]
       
       if manipulable_objects:
           recommendations.append(f"Found {len(manipulable_objects)} potentially manipulable objects")
       
       return recommendations
   
   def extract_actionable_information(self, grounding_data: Dict) > List[Dict]:
       """Extract actionable information from grounding results"""
       actionables = []
       
       # Extract target object for manipulation
       target_obj = grounding_data.get("target_object")
       if target_obj:
           actionables.append({
               "type": "manipulation_target",
               "object": target_obj["name"],
               "bbox": target_obj["bbox"],
               "confidence": target_obj["confidence"]
           })
       
       # Extract regions of interest
       roi = grounding_data.get("region_proposals", [])
       for region in roi:
           if region.get("type") == "primary":
               actionables.append({
                   "type": "region_of_interest",
                   "bbox": region["bbox"],
                   "object_name": region["object_name"],
                   "confidence": region["confidence"]
               })
       
       return actionables
   
   def publish_integrated_result(self, result: Dict):
       """Publish integrated VLM results"""
       result_msg = String()
       result_msg.data = json.dumps(result, indent=2)
       self.integrated_result_pub.publish(result_msg)
       
       # Update system status
       status_msg = String()
       status_msg.data = f"Integrated result published with {result.get('source', 'unknown')} data"
       self.system_status_pub.publish(status_msg)
   
   def run(self):
       """Run the complete VLM pipeline"""
       rospy.loginfo("Starting Complete VLM Robotics Pipeline...")
       
       try:
           rospy.spin()
       except KeyboardInterrupt:
           rospy.loginfo("Shutting down Complete VLM Robotics Pipeline...")

def main():
   pipeline = CompleteVLMPipeline()
   pipeline.run()

if __name__ == '__main__':
   main()
```

## Miniproject

Create a complete multimodal perception system that integrates:

1. Visionlanguage model for object recognition and scene understanding
2. Spatial reasoning for locationbased queries
3. Integration with robotic manipulation planning
4. Realtime processing capabilities
5. Confidence estimation for all perception outputs
6. Error handling and fallback mechanisms
7. Performance evaluation metrics
8. Humanrobot interaction through natural language

Your project should demonstrate:
 Complete VLM integration with robotics system
 Effective visual grounding for robotic tasks
 Multimodal data fusion for enhanced perception
 Realtime performance with appropriate optimization
 Robustness to different lighting and environmental conditions
 Natural language interaction with the robotic system

## Summary

This chapter covered VisionLanguage Models in robotics:

 **VLM Fundamentals**: Understanding contrastive learning and joint visuallinguistic embeddings
 **CLIP Integration**: Implementing zeroshot object recognition with VisionLanguage models
 **Spatial Reasoning**: Creating systems that understand spatial relationships between objects
 **Multimodal Fusion**: Combining visual and linguistic information for enhanced perception
 **Visual Grounding**: Connecting natural language expressions to visual content
 **Realtime Processing**: Optimizing VLMs for robotic applications
 **Confidence Estimation**: Assessing the reliability of perception outputs
 **Error Handling**: Creating robust systems that handle uncertain perception

VisionLanguage Models enable robots to understand their environment in more humanlike ways, connecting visual perception with natural language concepts. This capability is essential for creating robots that can interact naturally with humans and understand complex, languagedescribed tasks in diverse environments.