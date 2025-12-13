-----
title: Ch17  VoicetoAction with OpenAI Whisper
module: 4
chapter: 17
sidebar_label: Ch17: VoicetoAction with OpenAI Whisper
description: Implementing voice command processing with OpenAI Whisper for robotics applications
tags: [whisper, speechrecognition, voicecontrol, robotics, naturallanguage, audioprocessing]
difficulty: advanced
estimated_duration: 120
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# VoicetoAction with OpenAI Whisper

## Learning Outcomes
 Implement speech recognition using OpenAI Whisper in robotics applications
 Integrate voice commands with robot control systems
 Process realtime audio for continuous interaction
 Design voice command grammars for robotic tasks
 Implement audio preprocessing for robotics environment conditions
 Handle voice command ambiguities and context
 Create multimodal feedback systems for voice interactions
 Evaluate speech recognition performance in robotics contexts

## Theory

### OpenAI Whisper for Robotics

OpenAI Whisper is a robust speech recognition model that performs well in various conditions. For robotics applications, Whisper offers:

<MermaidDiagram chart={`
graph TD;
    A[Audio Input] > B[Preprocessing];
    B > C[Whisper ASR];
    C > D[SpeechtoText];
    D > E[NLP Processing];
    E > F[Command Interpretation];
    F > G[Robot Action];
    
    H[Robot Feedback] > I[Audio Feedback];
    H > J[Visual Feedback];
    I > A;
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style G fill:#2196F3,stroke:#0D47A1,color:#fff;
    style C fill:#FF9800,stroke:#E65100,color:#fff;
`} />

### Challenges in Robotics Audio Processing

**Acoustic Environment**: Robots operate in noisy environments where background sounds, motor noise, and other robots can interfere with speech recognition.

**RealTime Requirements**: Robot systems often require immediate responses to user commands.

**Vocabulary Constraints**: Robot commands are typically limited to specific actions and objects in the environment.

### Audio Preprocessing for Robotics

Preprocessing is critical for effective speech recognition in robotics environments:

 **Noise Reduction**: Filtering ambient noise
 **Audio Enhancement**: Improving signaltonoise ratio
 **Voice Activity Detection**: Identifying when speech is present
 **Echo Cancellation**: Managing audio feedback in robot systems

### Voice Command Context Processing

Robotics applications often require contextual understanding where commands depend on current robot state, environment, and previous interactions.

## StepbyStep Labs

### Lab 1: Setting up Whisper for Robotics

1. **Install Whisper and related dependencies**:
   ```bash
   pip install openaiwhisper torch torchaudio
   pip install pyaudio sounddevice librosa numpy
   pip install SpeechRecognition
   pip install ros2 rospy
   ```

2. **Create a basic Whisper interface for robotics** (`whisper_robot_interface.py`):
   ```python
   #!/usr/bin/env python3

   import whisper
   import torch
   import pyaudio
   import wave
   import numpy as np
   import rospy
   import threading
   import queue
   import time
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   from typing import Optional, Callable, Dict
   import tempfile
   import os

   class WhisperRobotInterface:
       def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
           # Initialize Whisper model
           print(f"Loading Whisper model '{model_size}' on {device}...")
           self.model = whisper.load_model(model_size).to(device)
           self.device = device
           
           # Audio parameters
           self.rate = 16000  # Sample rate (Whisper expects 16kHz)
           self.chunk = 1024  # Buffer size
           self.format = pyaudio.paInt16  # Audio format
           self.channels = 1  # Mono
           
           # Initialize PyAudio
           self.audio = pyaudio.PyAudio()
           
           # Command queue for processing
           self.command_queue = queue.Queue()
           
           # Robot control
           rospy.init_node('whisper_robot_interface', anonymous=True)
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           
           # Publishers for results
           self.transcript_pub = rospy.Publisher('/voice_transcript', String, queue_size=10)
           self.command_pub = rospy.Publisher('/voice_command', String, queue_size=10)
           
           # State variables
           self.is_listening = False
           self.recording_thread = None
           self.processing_thread = None
           self.voice_activity_threshold = 0.01
           self.min_silence_duration = 1.0  # seconds of silence to end recording
           
           print("Whisper Robot Interface initialized")
       
       def start_listening(self):
           """Start listening for voice commands"""
           if self.is_listening:
               print("Already listening")
               return
           
           self.is_listening = True
           
           # Start audio recording thread
           self.recording_thread = threading.Thread(target=self._record_audio_continuously)
           self.recording_thread.daemon = True
           self.recording_thread.start()
           
           # Start processing thread
           self.processing_thread = threading.Thread(target=self._process_commands)
           self.processing_thread.daemon = True
           self.processing_thread.start()
           
           rospy.loginfo("Started listening for voice commands")
       
       def stop_listening(self):
           """Stop listening for voice commands"""
           self.is_listening = False
           
           if self.recording_thread is not None:
               self.recording_thread.join(timeout=2.0)
           
           if self.processing_thread is not None:
               self.processing_thread.join(timeout=2.0)
           
           rospy.loginfo("Stopped listening for voice commands")
       
       def _is_voice_active(self, audio_data):
           """Detect if voice is active in audio chunk"""
           # Convert to numpy array
           audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
           
           # Calculate RMS amplitude
           rms = np.sqrt(np.mean(audio_array ** 2))
           
           return rms > self.voice_activity_threshold
       
       def _record_audio_chunk(self, seconds=1.0):
           """Record a single chunk of audio"""
           stream = self.audio.open(
               format=self.format,
               channels=self.channels,
               rate=self.rate,
               input=True,
               frames_per_buffer=self.chunk
           )
           
           frames = []
           chunks = int(self.rate / self.chunk * seconds)
           
           for _ in range(chunks):
               data = stream.read(self.chunk)
               frames.append(data)
           
           stream.stop_stream()
           stream.close()
           
           return b''.join(frames)
       
       def _record_audio_continuously(self):
           """Continuously record audio and detect voice activity"""
           stream = self.audio.open(
               format=self.format,
               channels=self.channels,
               rate=self.rate,
               input=True,
               frames_per_buffer=self.chunk
           )
           
           recording = False
           silence_counter = 0
           frames = []
           silence_chunks = int(self.rate / self.chunk * 0.1)  # 0.1s chunks for silence detection
           
           try:
               while self.is_listening:
                   data = stream.read(self.chunk, exception_on_overflow=False)
                   
                   if self._is_voice_active(data):
                       if not recording:
                           # Start recording
                           rospy.loginfo("Voice activity detected, starting recording...")
                           recording = True
                           frames = [data]
                           silence_counter = 0
                       else:
                           # Continue recording
                           frames.append(data)
                           silence_counter = 0
                   else:
                       if recording:
                           # Add to silence counter
                           silence_counter += 1
                           
                           # Append to frames anyway (might be trailing speech)
                           frames.append(data)
                           
                           # Check if silence duration exceeds threshold
                           if silence_counter > int(self.min_silence_duration * self.rate / self.chunk):
                               # End of speech detected
                               rospy.loginfo(f"End of speech detected, recorded {len(frames)} frames")
                               
                               # Save recorded audio to temporary file
                               temp_filename = self._save_audio_to_temp_file(frames)
                               
                               # Add to processing queue
                               self.command_queue.put(temp_filename)
                               
                               # Reset for next recording
                               recording = False
                               frames = []
                               silence_counter = 0
           except Exception as e:
               rospy.logerr(f"Error in audio recording: {e}")
           finally:
               stream.stop_stream()
               stream.close()
       
       def _save_audio_to_temp_file(self, frames):
           """Save audio frames to a temporary WAV file"""
           # Create temporary WAV file
           temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
           temp_file = wave.open(temp_path, 'wb')
           temp_file.setnchannels(self.channels)
           temp_file.setsampwidth(self.audio.get_sample_size(self.format))
           temp_file.setframerate(self.rate)
           temp_file.writeframes(b''.join(frames))
           temp_file.close()
           os.close(temp_fd)
           
           return temp_path
       
       def _process_commands(self):
           """Process audio files in the queue with Whisper"""
           while self.is_listening or not self.command_queue.empty():
               try:
                   # Get audio file from queue
                   audio_file = self.command_queue.get(timeout=1.0)
                   
                   # Process with Whisper
                   result = self._transcribe_audio(audio_file)
                   
                   if result and result.strip():
                       rospy.loginfo(f"Transcribed: {result}")
                       
                       # Publish transcript
                       transcript_msg = String()
                       transcript_msg.data = result
                       self.transcript_pub.publish(transcript_msg)
                       
                       # Process command
                       self._execute_command(result)
                   else:
                       rospy.loginfo("No speech detected or transcription failed")
                   
                   # Clean up temp file
                   if os.path.exists(audio_file):
                       os.remove(audio_file)
                   
                   self.command_queue.task_done()
                   
               except queue.Empty:
                   continue
               except Exception as e:
                   rospy.logerr(f"Error processing command: {e}")
       
       def _transcribe_audio(self, audio_file_path):
           """Transcribe audio using Whisper"""
           try:
               result = self.model.transcribe(
                   audio_file_path,
                   language="english",
                   fp16=False  # Set to True if using CUDA for faster inference
               )
               return result['text'].strip()
           except Exception as e:
               rospy.logerr(f"Transcription error: {e}")
               return ""
       
       def _execute_command(self, transcription):
           """Execute command based on transcription"""
           # Publish raw command
           cmd_msg = String()
           cmd_msg.data = transcription
           self.command_pub.publish(cmd_msg)
           
           # Simple command parsing (in real system, use NLP/RAG)
           command = self._parse_voice_command(transcription)
           
           if command:
               rospy.loginfo(f"Executing command: {command}")
               self._send_robot_command(command)
           else:
               rospy.logwarn(f"Unrecognized command: {transcription}")
       
       def _parse_voice_command(self, text):
           """Parse voice command into robot action"""
           text = text.lower()
           
           # Define simple command mappings
           commands = {
               "move forward": {"action": "move", "linear": 0.5, "angular": 0.0},
               "move backward": {"action": "move", "linear": 0.5, "angular": 0.0},
               "turn left": {"action": "turn", "linear": 0.0, "angular": 0.5},
               "turn right": {"action": "turn", "linear": 0.0, "angular": 0.5},
               "stop": {"action": "stop", "linear": 0.0, "angular": 0.0},
               "go straight": {"action": "move", "linear": 0.3, "angular": 0.0},
               "halt": {"action": "stop", "linear": 0.0, "angular": 0.0}
           }
           
           # Find best matching command
           for cmd_text, cmd_def in commands.items():
               if cmd_text in text:
                   return cmd_def
           
           # Check for other variants using simple fuzzy matching
           if "forward" in text or "ahead" in text:
               return {"action": "move", "linear": 0.3, "angular": 0.0}
           elif "back" in text or "reverse" in text:
               return {"action": "move", "linear": 0.3, "angular": 0.0}
           elif "left" in text:
               return {"action": "turn", "linear": 0.0, "angular": 0.3}
           elif "right" in text:
               return {"action": "turn", "linear": 0.0, "angular": 0.3}
           elif "stop" in text or "halt" in text:
               return {"action": "stop", "linear": 0.0, "angular": 0.0}
           
           return None
       
       def _send_robot_command(self, command):
           """Send command to robot"""
           twist = Twist()
           
           if command["action"] == "move" or command["action"] == "turn":
               twist.linear.x = command["linear"]
               twist.angular.z = command["angular"]
           elif command["action"] == "stop":
               twist.linear.x = 0.0
               twist.angular.z = 0.0
           
           self.cmd_vel_pub.publish(twist)
       
       def test_transcription(self, audio_file_path):
           """Test transcription on a specific audio file"""
           result = self._transcribe_audio(audio_file_path)
           rospy.loginfo(f"Test transcription: {result}")
           return result
       
       def __del__(self):
           """Cleanup audio resources"""
           if hasattr(self, 'audio'):
               self.audio.terminate()

   # Example usage
   if __name__ == "__main__":
       # Initialize interface (you'll need to provide your OpenAI API key for certain features)
       interface = WhisperRobotInterface(model_size="base")
       
       try:
           interface.start_listening()
           rospy.spin()  # Keep node alive
       except KeyboardInterrupt:
           interface.stop_listening()
           print("Shutting down...")
   ```

### Lab 2: Advanced Voice Command Processing

1. **Create a more sophisticated voice command processor** (`advanced_voice_processor.py`):
   ```python
   #!/usr/bin/env python3

   import whisper
   import torch
   import pyaudio
   import numpy as np
   import librosa
   import rospy
   import threading
   import queue
   import json
   import re
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist, Point
   from sensor_msgs.msg import LaserScan
   from typing import Dict, List, Optional, Tuple
   from dataclasses import dataclass
   import tempfile
   import os

   @dataclass
   class VoiceCommand:
       action: str
       parameters: Dict
       confidence: float
       original_text: str

   class AdvancedVoiceProcessor:
       def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
           # Initialize Whisper model
           self.model = whisper.load_model(model_size).to(device)
           self.device = device
           
           # Audio parameters
           self.rate = 16000
           self.chunk = 1024
           self.format = pyaudio.paInt16
           self.channels = 1
           
           # Initialize PyAudio
           self.audio = pyaudio.PyAudio()
           
           # ROS initialization
           rospy.init_node('advanced_voice_processor', anonymous=True)
           
           # Publishers and Subscribers
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           self.voice_cmd_pub = rospy.Publisher('/parsed_voice_command', String, queue_size=10)
           self.transcript_pub = rospy.Publisher('/voice_transcript', String, queue_size=10)
           self.status_pub = rospy.Publisher('/voice_control_status', String, queue_size=10)
           rospy.Subscriber('/scan', LaserScan, self.laser_callback)
           
           # State management
           self.laser_scan = None
           self.robot_position = Point(x=0.0, y=0.0, z=0.0)
           self.robot_heading = 0.0
           
           # Command queue
           self.command_queue = queue.Queue()
           
           # Voice activity detection
           self.voice_activity_threshold = 0.005
           self.min_voice_duration = 0.5  # seconds
           self.min_silence_duration = 1.0
           
           # Thread management
           self.is_listening = False
           self.recording_thread = None
           self.processing_thread = None
           
           # Command vocabularies
           self.navigation_commands = [
               "move forward", "move backward", "go forward", "go back",
               "go straight", "move straight", "turn left", "turn right",
               "pivot left", "pivot right", "rotate left", "rotate right", 
               "stop", "halt", "wait", "go to", "navigate to", "move to",
               "approach", "come here", "move closer", "go away"
           ]
           
           self.object_commands = [
               "pick up", "grasp", "grab", "take", "lift", "drop",
               "put down", "release", "place", "move", "bring", "fetch"
           ]
           
           # Contextaware command processor
           self.context_memory = []
           self.max_context_items = 50
           
           print("Advanced Voice Processor initialized")
       
       def laser_callback(self, msg):
           """Update laser scan data"""
           self.laser_scan = msg
       
       def update_robot_state(self, position: Point, heading: float):
           """Update robot state for contextual processing"""
           self.robot_position = position
           self.robot_heading = heading
       
       def start_listening(self):
           """Start voice command processing"""
           if self.is_listening:
               return
           
           self.is_listening = True
           
           # Start threads
           self.recording_thread = threading.Thread(target=self._continuous_recording)
           self.recording_thread.daemon = True
           self.recording_thread.start()
           
           self.processing_thread = threading.Thread(target=self._process_audio_queue)
           self.processing_thread.daemon = True
           self.processing_thread.start()
           
           self.status_pub.publish(String(data="Voice control active"))
           rospy.loginfo("Advanced voice processor started")
       
       def stop_listening(self):
           """Stop voice command processing"""
           self.is_listening = False
           self.status_pub.publish(String(data="Voice control inactive"))
       
       def _detect_voice_activity(self, audio_chunk):
           """Detect voice activity in audio chunk"""
           # Convert to numpy
           audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
           
           # Calculate RMS energy
           rms = np.sqrt(np.mean(audio_np ** 2))
           
           # Use librosa for more sophisticated features if needed
           # spectral_centroids = librosa.feature.spectral_centroid(y=audio_np, sr=self.rate)[0]
           
           return rms > self.voice_activity_threshold
       
       def _continuous_recording(self):
           """Continuously record and detect speech segments"""
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
           min_voice_frames = int(self.min_voice_duration * self.rate / self.chunk)
           min_silence_frames = int(self.min_silence_duration * self.rate / self.chunk)
           
           try:
               while self.is_listening:
                   chunk = stream.read(self.chunk, exception_on_overflow=False)
                   
                   if self._detect_voice_activity(chunk):
                       if not recording:
                           # Potential start of speech
                           voice_active_count += 1
                           if voice_active_count >= min_voice_frames:
                               # Confirmed start of speech
                               recording = True
                               frames = [chunk] * min_voice_frames  # Include preroll
                               voice_active_count = 0
                               silence_count = 0
                       else:
                           # Continue recording
                           frames.append(chunk)
                           silence_count = 0
                   else:
                       if recording:
                           # Accumulate silence
                           frames.append(chunk)  # Add to recording buffer
                           silence_count += 1
                           
                           if silence_count >= min_silence_frames:
                               # End of speech segment
                               if len(frames) >= min_voice_frames:  # Ensure minimum length
                                   # Save to temp file
                                   temp_file = self._save_recording(frames)
                                   self.command_queue.put(temp_file)
                                   rospy.loginfo(f"Recorded speech segment ({len(frames)} frames)")
                               
                               # Reset for next segment
                               recording = False
                               frames = []
                               silence_count = 0
                               voice_active_count = 0
                       else:
                           # Reset counters when not recording
                           voice_active_count = 0
           except Exception as e:
               rospy.logerr(f"Recording error: {e}")
           finally:
               stream.stop_stream()
               stream.close()
       
       def _save_recording(self, frames):
           """Save recorded frames to temporary WAV file"""
           import wave
           
           temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
           wf = wave.open(temp_path, 'wb')
           wf.setnchannels(self.channels)
           wf.setsampwidth(self.audio.get_sample_size(self.format))
           wf.setframerate(self.rate)
           wf.writeframes(b''.join(frames))
           wf.close()
           os.close(temp_fd)
           
           return temp_path
       
       def _process_audio_queue(self):
           """Process audio files from the queue"""
           while self.is_listening or not self.command_queue.empty():
               try:
                   audio_file = self.command_queue.get(timeout=1.0)
                   
                   # Transcribe audio
                   transcript = self._transcribe_with_retry(audio_file)
                   
                   if transcript and transcript.strip():
                       rospy.loginfo(f"Transcribed: {transcript}")
                       
                       # Publish transcript
                       self.transcript_pub.publish(String(data=transcript))
                       
                       # Parse and execute command
                       command = self._parse_command(transcript)
                       if command:
                           self._execute_parsed_command(command)
                           self._publish_parsed_command(command)
                           
                           # Update context
                           self._update_context(command)
                       
                   # Clean up temp file
                   if os.path.exists(audio_file):
                       os.remove(audio_file)
                   
                   self.command_queue.task_done()
                   
               except queue.Empty:
                   continue
               except Exception as e:
                   rospy.logerr(f"Processing error: {e}")
       
       def _transcribe_with_retry(self, audio_file, max_retries=3):
           """Transcribe audio with retry logic"""
           for attempt in range(max_retries):
               try:
                   result = self.model.transcribe(audio_file, language="english")
                   return result.get('text', '').strip()
               except Exception as e:
                   rospy.logwarn(f"Transcription attempt {attempt + 1} failed: {e}")
                   if attempt == max_retries  1:
                       return ""  # Return empty string if all retries failed
       
       def _parse_command(self, text: str) > Optional[VoiceCommand]:
           """Parse natural language command using regex and semantic analysis"""
           original_text = text
           text = text.lower().strip()
           
           # Clean text
           text = re.sub(r'[^\w\s]', ' ', text)
           text = ' '.join(text.split())  # Remove extra whitespace
           
           # Define command patterns
           patterns = [
               # Movement commands
               (r'.*\b(forward|ahead|straight)\b.*', self._parse_move_forward),
               (r'.*\b(backward|back|reverse|behind)\b.*', self._parse_move_backward),
               (r'.*\b(left|counterclockwise|port)\b.*', self._parse_turn_left),
               (r'.*\b(right|clockwise|starboard)\b.*', self._parse_turn_right),
               (r'.*\b(stop|halt|wait|pause)\b.*', self._parse_stop),
               
               # Distancebased movement
               (r'.*\b(move|go|travel)\b.*\b(\d+(?:\.\d+)?)\b.*\b(meter|meters|metre|metres)\b', 
                self._parse_move_distance),
               
               # Direction and distance
               (r'.*\b(turn|rotate|pivot)\b.*\b(\d+(?:\.\d+)?)\b.*\b(degrees|deg)\b', 
                self._parse_turn_degrees),
           ]
           
           # Try each pattern
           for pattern, parser_func in patterns:
               match = re.search(pattern, text)
               if match:
                   command = parser_func(match, text)
                   if command:
                       # Calculate confidence based on match strength and vocabulary
                       confidence = self._calculate_confidence(original_text, command.action)
                       return VoiceCommand(
                           action=command.action,
                           parameters=command.parameters,
                           confidence=confidence,
                           original_text=original_text
                       )
           
           # If no specific pattern matched, try general classification
           return self._classify_general_command(text, original_text)
       
       def _parse_move_forward(self, match, text):
           """Parse forward movement command"""
           return VoiceCommand(
               action="move_forward",
               parameters={"speed": 0.3, "duration": 1.0},
               confidence=0.8,
               original_text=text
           )
       
       def _parse_move_backward(self, match, text):
           """Parse backward movement command"""
           return VoiceCommand(
               action="move_backward", 
               parameters={"speed": 0.3, "duration": 1.0},
               confidence=0.8,
               original_text=text
           )
       
       def _parse_turn_left(self, match, text):
           """Parse left turn command"""
           return VoiceCommand(
               action="turn_left",
               parameters={"speed": 0.4, "angle": 90.0},
               confidence=0.8,
               original_text=text
           )
       
       def _parse_turn_right(self, match, text):
           """Parse right turn command"""
           return VoiceCommand(
               action="turn_right", 
               parameters={"speed": 0.4, "angle": 90.0},
               confidence=0.8,
               original_text=text
           )
       
       def _parse_stop(self, match, text):
           """Parse stop command"""
           return VoiceCommand(
               action="stop",
               parameters={"speed": 0.0, "duration": 0.0},
               confidence=0.9,
               original_text=text
           )
       
       def _parse_move_distance(self, match, text):
           """Parse distancebased movement command"""
           distance = float(match.group(2))
           return VoiceCommand(
               action="move_distance",
               parameters={"distance": distance, "speed": 0.3},
               confidence=0.7,
               original_text=text
           )
       
       def _parse_turn_degrees(self, match, text):
           """Parse degreebased turn command"""
           angle = float(match.group(2))
           return VoiceCommand(
               action="turn_degrees",
               parameters={"angle": angle, "speed": 0.3},
               confidence=0.7,
               original_text=text
           )
       
       def _classify_general_command(self, text, original_text):
           """Classify commands that don't match specific patterns"""
           # Check similarity with known commands
           if any(cmd in text for cmd in self.navigation_commands):
               return VoiceCommand(
                   action="generic_navigation",
                   parameters={"command": text},
                   confidence=0.5,
                   original_text=original_text
               )
           elif any(cmd in text for cmd in self.object_commands):
               return VoiceCommand(
                   action="manipulation_command",
                   parameters={"command": text},
                   confidence=0.5,
                   original_text=original_text
               )
           else:
               return None  # Unknown command
       
       def _calculate_confidence(self, original_text, action):
           """Calculate confidence based on various factors"""
           # Base confidence from pattern match
           base_conf = 0.7
           
           # Boost for clear directional words
           directional_boost = sum(1 for word in ['forward', 'backward', 'left', 'right', 'stop'] 
                                  if word in original_text.lower())
           
           # Penalize for ambiguous words
           ambiguity_penalty = sum(0.1 for word in ['maybe', 'perhaps', 'kind of', 'i think'] 
                                  if word in original_text.lower())
           
           confidence = min(0.95, base_conf + (directional_boost * 0.05)  ambiguity_penalty)
           return max(0.1, confidence)
       
       def _execute_parsed_command(self, command: VoiceCommand):
           """Execute a parsed voice command"""
           if command.confidence < 0.3:
               rospy.logwarn(f"Command confidence too low ({command.confidence}), ignoring: {command.original_text}")
               return
           
           rospy.loginfo(f"Executing command '{command.action}' with confidence {command.confidence:.2f}")
           
           # Handle different command types
           if command.action == "move_forward":
               self._execute_move(0.3, 0.0, command.parameters.get("duration", 2.0))
           elif command.action == "move_backward":
               self._execute_move(0.3, 0.0, command.parameters.get("duration", 2.0))
           elif command.action == "turn_left":
               self._execute_move(0.0, 0.5, command.parameters.get("duration", 1.5))
           elif command.action == "turn_right":
               self._execute_move(0.0, 0.5, command.parameters.get("duration", 1.5))
           elif command.action == "stop":
               self._execute_stop()
           elif command.action == "move_distance":
               # For distancebased movement, we'd need to track odometry
               # For simplicity, use timebased approximation
               distance = command.parameters["distance"]
               speed = command.parameters["speed"]
               duration = distance / speed
               self._execute_move(speed, 0.0, duration)
           elif command.action == "turn_degrees":
               # For degreebased turns, we'd need to track robot heading
               # For simplicity, use timebased approximation
               angle = command.parameters["angle"]
               duration = abs(angle) / 90.0 * 1.5  # Assuming 90 degree turn takes 1.5 seconds
               angular_speed = np.sign(angle) * 0.5
               self._execute_move(0.0, angular_speed, duration)
           else:
               rospy.logwarn(f"Unknown command action: {command.action}")
       
       def _execute_move(self, linear_x, angular_z, duration):
           """Execute movement command"""
           twist = Twist()
           twist.linear.x = linear_x
           twist.angular.z = angular_z
           
           # Publish for the duration
           rate = rospy.Rate(10)  # 10 Hz
           start_time = rospy.Time.now()
           
           while (rospy.Time.now()  start_time).to_sec() < duration and not rospy.is_shutdown():
               self.cmd_vel_pub.publish(twist)
               rate.sleep()
           
           # Stop robot after movement
           self._execute_stop()
       
       def _execute_stop(self):
           """Stop the robot"""
           twist = Twist()
           twist.linear.x = 0.0
           twist.angular.z = 0.0
           self.cmd_vel_pub.publish(twist)
       
       def _publish_parsed_command(self, command: VoiceCommand):
           """Publish parsed command as JSON"""
           cmd_dict = {
               "action": command.action,
               "parameters": command.parameters,
               "confidence": command.confidence,
               "original_text": command.original_text
           }
           
           self.voice_cmd_pub.publish(String(data=json.dumps(cmd_dict)))
       
       def _update_context(self, command: VoiceCommand):
           """Update context memory with new command"""
           self.context_memory.append({
               "timestamp": rospy.Time.now().to_sec(),
               "command": command
           })
           
           # Keep only recent items
           if len(self.context_memory) > self.max_context_items:
               self.context_memory = self.context_memory[self.max_context_items:]
       
       def get_context_summary(self):
           """Get a summary of recent context"""
           if not self.context_memory:
               return "No recent commands in context"
           
           recent_commands = [item["command"].original_text for item in self.context_memory[5:]]
           return f"Recent commands: {', '.join(recent_commands)}"
   ```

### Lab 3: Integrating with ROS and Robot Control

1. **Create ROS integration for voice control** (`ros_voice_integration.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import LaserScan
   import json
   import threading
   from typing import Dict, Any
   from advanced_voice_processor import AdvancedVoiceProcessor

   class ROSVoiceController:
       def __init__(self):
           rospy.init_node('ros_voice_controller', anonymous=True)
           
           # Initialize voice processor
           self.voice_processor = AdvancedVoiceProcessor(model_size="base")
           
           # Publishers
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           self.status_pub = rospy.Publisher('/voice_control_status', String, queue_size=10)
           self.response_pub = rospy.Publisher('/voice_response', String, queue_size=10)
           
           # Subscribers
           rospy.Subscriber('/voice_transcript', String, self.transcript_callback)
           rospy.Subscriber('/parsed_voice_command', String, self.parsed_command_callback)
           rospy.Subscriber('/scan', LaserScan, self.laser_callback)
           rospy.Subscriber('/toggle_voice_control', Bool, self.toggle_callback)
           
           # Parameters
           self.voice_control_enabled = rospy.get_param('~voice_control_enabled', True)
           self.safe_distance = rospy.get_param('~safe_distance', 0.5)  # meters
           
           # State
           self.laser_scan = None
           self.obstacle_detected = False
           self.voice_control_active = False
           
           rospy.loginfo("ROS Voice Controller initialized")
       
       def laser_callback(self, msg):
           """Update laser scan and check for obstacles"""
           self.laser_scan = msg
           
           # Check for frontfacing obstacles
           if self.laser_scan:
               # Get frontfacing ranges (±30 degrees)
               front_ranges = self.laser_scan.ranges[
                   len(self.laser_scan.ranges)//2  30 :
                   len(self.laser_scan.ranges)//2 + 30
               ]
               
               # Filter valid ranges
               valid_ranges = [r for r in front_ranges if not (r == float('inf') or r == float('NaN'))]
               
               if valid_ranges:
                   min_range = min(valid_ranges)
                   self.obstacle_detected = min_range < self.safe_distance
                   
                   if self.obstacle_detected and self.voice_control_active:
                       # Issue warning if robot is trying to move forward into obstacle
                       self.announce_obstacle(min_range)
       
       def transcript_callback(self, msg):
           """Handle new transcript from voice recognition"""
           rospy.loginfo(f"New transcript: {msg.data}")
           
           # Check if command is related to movement when obstacle is detected
           if self.obstacle_detected:
               text = msg.data.lower()
               if any(word in text for word in ['forward', 'ahead', 'straight', 'go']):
                   self.warn_about_obstacle(msg.data)
       
       def parsed_command_callback(self, msg):
           """Handle parsed voice command"""
           try:
               cmd_data = json.loads(msg.data)
               action = cmd_data["action"]
               confidence = cmd_data["confidence"]
               original_text = cmd_data["original_text"]
               
               rospy.loginfo(f"Parsed command: {action} (conf: {confidence:.2f})")
               
               if confidence > 0.3:
                   if self.voice_control_enabled and self._is_command_safe(action, cmd_data):
                       self.execute_command(action, cmd_data)
                       self.acknowledge_command(original_text)
                   else:
                       self.reject_command(original_text, "Command deemed unsafe")
               else:
                   self.reject_command(original_text, "Low confidence")
           
           except json.JSONDecodeError as e:
               rospy.logerr(f"JSON decode error: {e}")
       
       def toggle_callback(self, msg):
           """Toggle voice control on/off"""
           if msg.data:
               if not self.voice_control_active:
                   self.voice_control_active = True
                   self.voice_processor.start_listening()
                   self.status_pub.publish(String(data="Voice control activated"))
                   self.announce("Voice control activated")
                   rospy.loginfo("Voice control activated")
           else:
               if self.voice_control_active:
                   self.voice_control_active = False
                   self.voice_processor.stop_listening()
                   self.status_pub.publish(String(data="Voice control deactivated"))
                   self.announce("Voice control deactivated")
                   rospy.loginfo("Voice control deactivated")
       
       def _is_command_safe(self, action: str, cmd_data: Dict[str, Any]) > bool:
           """Check if command is safe to execute given current state"""
           # Check for obstacles if command involves forward movement
           if action in ["move_forward", "move_distance"] and self.obstacle_detected:
               rospy.logwarn("Obstacle detected, cancelling forward movement")
               return False
           
           # Check other safety conditions
           # (Add more safety checks as needed)
           
           return True
       
       def execute_command(self, action: str, cmd_data: Dict[str, Any]):
           """Execute the parsed command"""
           twist = Twist()
           
           if action == "move_forward":
               twist.linear.x = 0.3
           elif action == "move_backward":
               twist.linear.x = 0.3
           elif action == "turn_left":
               twist.angular.z = 0.4
           elif action == "turn_right":
               twist.angular.z = 0.4
           elif action == "stop":
               pass  # Already zero
           elif action == "move_distance":
               # This would need odometry feedback for precision
               # For now, just move forward
               twist.linear.x = 0.3
           elif action == "turn_degrees":
               # This would need orientation feedback for precision
               # For now, just turn
               angle = cmd_data["parameters"].get("angle", 90.0)
               twist.angular.z = 0.4 if angle > 0 else 0.4
           
           # Publish command
           self.cmd_vel_pub.publish(twist)
           
           # For commands that need to stop after execution, add a timer
           if action in ["move_distance", "turn_degrees"]:
               duration = cmd_data["parameters"].get("duration", 1.0)
               threading.Timer(duration, self._stop_after_movement).start()
       
       def _stop_after_movement(self):
           """Stop robot after timed movement command"""
           twist = Twist()
           self.cmd_vel_pub.publish(twist)
       
       def announce(self, message: str):
           """Announce a message (placeholder  would use texttospeech in real implementation)"""
           rospy.loginfo(f"Announcement: {message}")
           # In a real implementation, this would use a texttospeech system
       
       def acknowledge_command(self, command_text: str):
           """Acknowledge successful command execution"""
           self.announce(f"Executing: {command_text}")
       
       def reject_command(self, command_text: str, reason: str):
           """Reject command with reason"""
           response = f"Unable to execute '{command_text}'. Reason: {reason}."
           self.response_pub.publish(String(data=response))
           self.announce(response)
       
       def announce_obstacle(self, distance: float):
           """Announce obstacle detection"""
           if distance < 0.2:
               alert = "Critical obstacle very close!"
           elif distance < 0.5:
               alert = f"Obstacle detected at {distance:.2f} meters."
           else:
               alert = f"Obstacle ahead at {distance:.2f} meters."
           
           self.announce(alert)
       
       def warn_about_obstacle(self, command_text: str):
           """Warn about obstacle when forward command is issued"""
           warning = f"Obstacle detected. Cannot execute '{command_text}' safely."
           self.response_pub.publish(String(data=warning))
           self.announce(warning)
       
       def run(self):
           """Run the voice controller"""
           rospy.loginfo("Starting ROS Voice Controller")
           
           # Start voice processor if initially enabled
           if self.voice_control_enabled:
               self.voice_control_active = True
               self.voice_processor.start_listening()
               self.announce("Voice control system online")
           
           try:
               rospy.spin()
           except KeyboardInterrupt:
               rospy.loginfo("Shutting down...")
               self.voice_processor.stop_listening()

   def main():
       controller = ROSVoiceController()
       controller.run()

   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's a complete example that demonstrates the voicetoaction system:

```python
#!/usr/bin/env python3
# complete_voice_to_action_system.py

import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from complete_whisper_integration import WhisperRobotInterface
import threading
import time

class CompleteVoiceToActionSystem:
    """Complete voicetoaction system using Whisper for robotics"""
    
    def __init__(self):
        rospy.init_node('complete_voice_to_action_system', anonymous=True)
        
        # Parameters
        self.whisper_model_size = rospy.get_param('~whisper_model', 'base')
        self.voice_control_enabled = rospy.get_param('~voice_control_enabled', True)
        self.safe_distance = rospy.get_param('~safe_distance', 0.5)
        
        # Initialize Whisper interface
        self.whisper_interface = WhisperRobotInterface(
            model_size=self.whisper_model_size
        )
        
        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/voice_system_status', String, queue_size=10)
        
        rospy.Subscriber('/voice_transcript', String, self.transcript_callback)
        rospy.Subscriber('/voice_command', String, self.command_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # System state
        self.laser_scan = None
        self.obstacle_detected = False
        self.system_active = False
        self.last_command_time = 0
        
        rospy.loginfo("Complete VoicetoAction System initialized")
    
    def scan_callback(self, msg):
        """Update laser scan data"""
        self.laser_scan = msg
        
        # Check for obstacles in front
        if self.laser_scan:
            # Check front arc (±30 degrees)
            front_idx = len(self.laser_scan.ranges) // 2
            front_start = max(0, front_idx  30)
            front_end = min(len(self.laser_scan.ranges), front_idx + 30)
            front_ranges = self.laser_scan.ranges[front_start:front_end]
            
            # Filter valid ranges
            valid_ranges = [r for r in front_ranges if 0.1 < r < 10.0]
            
            if valid_ranges:
                min_range = min(valid_ranges)
                self.obstacle_detected = min_range < self.safe_distance
            else:
                self.obstacle_detected = False
    
    def transcript_callback(self, msg):
        """Handle new voice transcript"""
        transcript = msg.data
        rospy.loginfo(f"Heard: {transcript}")
        
        # Check if transcript contains a command
        if self._contains_command(transcript):
            if self.obstacle_detected and self._is_movement_command(transcript):
                self._warn_about_obstacle()
            else:
                # Process command
                command = self._interpret_command(transcript)
                if command:
                    self._execute_command(command)
    
    def command_callback(self, msg):
        """Handle processed command"""
        command = msg.data
        rospy.loginfo(f"Processed command: {command}")
        # Additional command handling can go here
    
    def _contains_command(self, text):
        """Check if text contains actionable commands"""
        command_keywords = [
            'move', 'go', 'stop', 'turn', 'left', 'right', 
            'forward', 'backward', 'straight', 'halt'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in command_keywords)
    
    def _is_movement_command(self, text):
        """Check if command involves movement"""
        movement_keywords = ['forward', 'backward', 'go', 'move', 'straight']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in movement_keywords)
    
    def _interpret_command(self, text):
        """Interpret voice command"""
        text_lower = text.lower()
        
        # Simple command mapping
        if 'forward' in text_lower or 'ahead' in text_lower or 'straight' in text_lower:
            return {'action': 'move', 'direction': 'forward', 'speed': 0.3}
        elif 'backward' in text_lower or 'back' in text_lower:
            return {'action': 'move', 'direction': 'backward', 'speed': 0.3}
        elif 'left' in text_lower:
            return {'action': 'turn', 'direction': 'left', 'speed': 0.4}
        elif 'right' in text_lower:
            return {'action': 'turn', 'direction': 'right', 'speed': 0.4}
        elif 'stop' in text_lower or 'halt' in text_lower:
            return {'action': 'stop'}
        else:
            return None
    
    def _execute_command(self, command):
        """Execute parsed command"""
        twist = Twist()
        
        if command['action'] == 'move':
            if command['direction'] == 'forward':
                twist.linear.x = command['speed']
            elif command['direction'] == 'backward':
                twist.linear.x = command['speed']
        elif command['action'] == 'turn':
            if command['direction'] == 'left':
                twist.angular.z = command['speed']
            elif command['direction'] == 'right':
                twist.angular.z = command['speed']
        elif command['action'] == 'stop':
            # Twist is already zeroed
            pass
        
        # Publish command
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo(f"Executed command: {command}")
        
        # Update last command time
        self.last_command_time = time.time()
    
    def _warn_about_obstacle(self):
        """Warn about obstacle when movement command is issued"""
        rospy.logwarn("Obstacle detected in front! Movement command aborted.")
        status_msg = String()
        status_msg.data = "WARNING: Obstacle detected, movement halted"
        self.status_pub.publish(status_msg)
    
    def start_system(self):
        """Start the voicetoaction system"""
        rospy.loginfo("Starting voicetoaction system...")
        
        if self.voice_control_enabled:
            self.whisper_interface.start_listening()
            self.system_active = True
            rospy.loginfo("Voice control system activated")
        else:
            rospy.loginfo("Voice control disabled by parameter")
    
    def stop_system(self):
        """Stop the system"""
        self.whisper_interface.stop_listening()
        self.system_active = False
        
        # Stop robot
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        
        rospy.loginfo("Voicetoaction system stopped")
    
    def run(self):
        """Main run loop"""
        self.start_system()
        
        try:
            # Keep the node running
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Interrupted, shutting down...")
        finally:
            self.stop_system()

def main():
    """Main function to run the complete system"""
    system = CompleteVoiceToActionSystem()
    
    try:
        system.run()
    except Exception as e:
        rospy.logerr(f"Error running voicetoaction system: {e}")
        system.stop_system()

if __name__ == '__main__':
    main()
```

### Launch file for the system (`voice_control_system.launch`):

```xml
<launch>
  <! VoicetoAction System >
  <node name="complete_voice_to_action_system" pkg="robot_voice_control" type="complete_voice_to_action_system.py" output="screen">
    <param name="whisper_model" value="base"/>
    <param name="voice_control_enabled" value="true"/>
    <param name="safe_distance" value="0.5"/>
  </node>
  
  <! Optional: Add voice activity detection node >
  <node name="voice_activity_detector" pkg="audio_common" type="voice_activity_detector.py" output="screen">
    <param name="threshold" value="0.01"/>
  </node>
  
  <! Optional: Add texttospeech for feedback >
  <node name="text_to_speech" pkg="sound_play" type="say.py" output="screen"/>
</launch>
```

## Miniproject

Create a complete voicecontrolled robot system that:

1. Implements Whisperbased speech recognition for robot commands
2. Integrates with ROS for seamless robot control
3. Handles noise reduction and voice activity detection
4. Implements contextaware command processing
5. Creates a multimodal feedback system (audio + visual)
6. Evaluates performance with various accents and environments
7. Implements safety mechanisms for voice command execution
8. Provides error recovery for misunderstood commands

Your project should include:
 Complete Whisper integration with audio preprocessing
 ROS node for voice command processing
 Contextaware command interpretation
 Safety validation for voice commands
 Performance evaluation metrics
 Multimodal feedback system
 Error recovery mechanisms

## Summary

This chapter covered voicetoaction processing using OpenAI Whisper for robotics:

 **Whisper Integration**: Using OpenAI's Whisper model for speech recognition in robotics
 **Audio Preprocessing**: Techniques for handling roboticsspecific audio challenges
 **Command Interpretation**: Mapping voice commands to robot actions
 **RealTime Processing**: Continuous audio processing for immediate response
 **Safety Mechanisms**: Validation and safety checks for voicecontrolled robots
 **Context Awareness**: Using environmental context to disambiguate commands
 **ROS Integration**: Seamless integration with the Robot Operating System

Voice control provides a natural interface for humanrobot interaction, allowing users to command robots using everyday language. However, special attention must be paid to the unique challenges of audio processing in robotic environments and ensuring safe execution of voice commands.