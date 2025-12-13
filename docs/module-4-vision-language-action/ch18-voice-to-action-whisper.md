-----
title: Ch18  VoicetoAction with OpenAI Whisper
module: 4
chapter: 18
sidebar_label: Ch18: VoicetoAction with OpenAI Whisper
description: Implementing speech recognition and voice command processing for robotics using OpenAI Whisper
tags: [whisper, speechrecognition, voicecontrol, robotics, naturallanguage, audioprocessing]
difficulty: advanced
estimated_duration: 120
-----

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# VoicetoAction with OpenAI Whisper

## Learning Outcomes
 Understand speech recognition systems and OpenAI Whisper architecture
 Implement voice command processing for robotic systems
 Process realtime audio for continuous robot interaction
 Design voice command grammars for robotic tasks
 Integrate speech recognition with robot control systems
 Handle voice command ambiguities and context
 Create multimodal feedback systems for voice interactions
 Implement safety checks and validation for voice commands
 Evaluate speech recognition performance in robotic contexts

## Theory

### OpenAI Whisper Architecture

OpenAI Whisper is a robust automatic speech recognition (ASR) system that has revolutionized speech processing. Unlike traditional ASR systems, Whisper was trained on a vast dataset of audio and text from the internet, making it highly versatile and accurate across different domains.

<MermaidDiagram chart={`
graph TD;
    A[Audio Input] > B[Audio Preprocessing];
    B > C[Mel Spectrogram];
    C > D[Whisper Encoder];
    D > E[Whisper Decoder];
    E > F[Text Output];
    
    G[Robot Control] > H[Command Parser];
    H > I[NLP Processor];
    I > J[Action Generator];
    J > K[Robot Execution];
    
    F > I;
    K > L[Safety Validator];
    L > M[Execution Confirmation];
    M > K;
    
    N[Voice Command] > A;
    O[Robot Action] > P[Confirmation];
    
    style A fill:#4CAF50,stroke:#388E3C,color:#fff;
    style D fill:#2196F3,stroke:#0D47A1,color:#fff;
    style K fill:#FF9800,stroke:#E65100,color:#fff;
    style N fill:#E91E63,stroke:#AD1457,color:#fff;
`} />

### ASR in Robotics Context

Automatic Speech Recognition in robotics differs from traditional applications in several key ways:

 **Environmental Noise**: Robots operate in noisy environments with motor noise, fan noise, and other acoustic interference
 **Realtime Requirements**: Robot systems often require immediate responses to user commands
 **Limited Vocabulary**: Robot commands typically come from a predefined set of actions
 **Context Dependency**: Commands often depend on robot state and environment
 **Safety Considerations**: Voice commands might lead to physical actions that need validation

### Audio Preprocessing Pipeline

For robotic applications, audio preprocessing becomes critical due to the challenging acoustic environment:

 **Noise Reduction**: Filtering environmental noise to improve speech clarity
 **Voice Activity Detection**: Identifying when speech is actually occurring
 **Audio Enhancement**: Improving signaltonoise ratio
 **Echo Cancellation**: Removing selfgenerated robot sounds from audio input

### Voice Command Understanding

The transformation from speech to robot action involves several processing steps:

1. **Speech Recognition**: Converting audio to text
2. **Natural Language Understanding**: Interpreting the meaning of text
3. **Command Mapping**: Converting natural language to robotspecific commands
4. **Context Integration**: Considering robot state and environment
5. **Safety Validation**: Ensuring commands are safe to execute
6. **Action Execution**: Sending validated commands to robot

## StepbyStep Labs

### Lab 1: Setting up Whisper for Robotics

1. **Install Whisper and dependencies**:
   ```bash
   pip install openaiwhisper
   pip install torch torchaudio
   pip install pyaudio sounddevice
   pip install SpeechRecognition
   pip install rospy sensor_msgs std_msgs
   ```

2. **Create a basic Whisper interface** (`whisper_robot_interface.py`):
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
           self.rate = 16000  # Whisper expects 16kHz
           self.chunk = 1024
           self.format = pyaudio.paInt16
           self.channels = 1
           
           # Initialize PyAudio
           self.audio = pyaudio.PyAudio()
           
           # Initialize ROS
           rospy.init_node('whisper_robot_interface', anonymous=True)
           
           # Publishers
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           self.transcript_pub = rospy.Publisher('/voice_transcript', String, queue_size=10)
           self.command_pub = rospy.Publisher('/voice_command', String, queue_size=10)
           
           # Internal state
           self.transcript_queue = queue.Queue()
           self.is_listening = False
           self.recording_thread = None
           self.processing_thread = None
           
           # Voice activity threshold
           self.voice_activity_threshold = 0.01
           self.min_voice_duration = 0.5  # seconds
           self.max_silence_duration = 1.0  # seconds before stopping recording
           
           print("Whisper Robot Interface initialized")
       
       def start_listening(self):
           """Start continuous listening for voice commands"""
           if self.is_listening:
               print("Already listening")
               return
           
           self.is_listening = True
           
           # Start audio recording thread
           self.recording_thread = threading.Thread(target=self._record_audio_continuously)
           self.recording_thread.daemon = True
           self.recording_thread.start()
           
           # Start processing thread
           self.processing_thread = threading.Thread(target=self._process_recordings)
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
           audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
           
           # Calculate RMS amplitude
           rms = np.sqrt(np.mean(audio_array ** 2))
           
           return rms > self.voice_activity_threshold
       
       def _record_audio_continuously(self):
           """Continuously record audio with voice activity detection"""
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
                           if silence_counter > int(self.max_silence_duration * self.rate / self.chunk):
                               # End of speech detected
                               rospy.loginfo(f"End of speech detected, recorded {len(frames)} frames")
                               
                               # Save recorded audio to temporary file
                               temp_filename = self._save_audio_to_temp_file(frames)
                               
                               # Add to processing queue
                               self.transcript_queue.put(temp_filename)
                               
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
       
       def _process_recordings(self):
           """Process recorded audio files with Whisper"""
           while self.is_listening or not self.transcript_queue.empty():
               try:
                   # Get audio file from queue
                   audio_file = self.transcript_queue.get(timeout=1.0)
                   
                   # Transcribe audio
                   result = self._transcribe_audio(audio_file)
                   
                   if result and result.strip():
                       rospy.loginfo(f"Transcribed: {result}")
                       
                       # Publish transcript
                       transcript_msg = String()
                       transcript_msg.data = result
                       self.transcript_pub.publish(transcript_msg)
                       
                       # Process command
                       self._process_command(result)
                   else:
                       rospy.loginfo("No speech detected or transcription failed")
                   
                   # Clean up temp file
                   if os.path.exists(audio_file):
                       os.remove(audio_file)
                   
                   self.transcript_queue.task_done()
                   
               except queue.Empty:
                   continue
               except Exception as e:
                   rospy.logerr(f"Error processing audio: {e}")
       
       def _transcribe_audio(self, audio_file_path):
           """Transcribe audio file using Whisper"""
           try:
               result = self.model.transcribe(
                   audio_file_path,
                   language="english",  # Specify language for better accuracy
                   fp16=torch.cuda.is_available()  # Use fp16 if CUDA available
               )
               return result['text'].strip()
           except Exception as e:
               rospy.logerr(f"Transcription error: {e}")
               return ""
       
       def _process_command(self, transcription):
           """Process transcribed command and convert to robot action"""
           # Simple command parsing (in a real system, use more sophisticated NLU)
           command = self._parse_voice_command(transcription)
           
           if command:
               rospy.loginfo(f"Executed command: {command}")
               self.command_pub.publish(String(data=command))
               self._execute_robot_action(command)
           else:
               rospy.logwarn(f"Unrecognized command: {transcription}")
       
       def _parse_voice_command(self, text):
           """Simple command parser  in real system, use NLP/LLM"""
           text_lower = text.lower().strip()
           
           # Navigation commands
           if "move forward" in text_lower or "go forward" in text_lower or "forward" in text_lower:
               return "MOVE_FORWARD"
           elif "move backward" in text_lower or "go backward" in text_lower or "backward" in text_lower:
               return "MOVE_BACKWARD"
           elif "turn left" in text_lower or "left" in text_lower:
               return "TURN_LEFT"
           elif "turn right" in text_lower or "right" in text_lower:
               return "TURN_RIGHT"
           elif "stop" in text_lower or "halt" in text_lower:
               return "STOP"
           elif "come here" in text_lower or "come to me" in text_lower:
               return "COME_HERE"
           elif "follow me" in text_lower:
               return "FOLLOW_ME"
           elif "pick up" in text_lower or "grasp" in text_lower or "grab" in text_lower:
               return "GRASP_OBJECT"
           elif "put down" in text_lower or "release" in text_lower:
               return "RELEASE_OBJECT"
           
           return None
       
       def _execute_robot_action(self, command):
           """Execute robot action based on command"""
           twist = Twist()
           
           if command == "MOVE_FORWARD":
               twist.linear.x = 0.3  # m/s
           elif command == "MOVE_BACKWARD":
               twist.linear.x = 0.3
           elif command == "TURN_LEFT":
               twist.angular.z = 0.5  # rad/s
           elif command == "TURN_RIGHT":
               twist.angular.z = 0.5
           elif command == "STOP":
               twist.linear.x = 0.0
               twist.angular.z = 0.0
           elif command == "COME_HERE":
               # For this example, just turn in place
               twist.angular.z = 0.2
           elif command == "FOLLOW_ME":
               # This would require person tracking
               pass
           elif command == "GRASP_OBJECT":
               # This would require gripper control
               pass
           elif command == "RELEASE_OBJECT":
               # This would require gripper control
               pass
           else:
               rospy.logwarn(f"Unknown command: {command}")
               return
           
           # Publish command
           self.cmd_vel_pub.publish(twist)
       
       def run(self):
           """Run the main loop"""
           rospy.loginfo("Whisper Robot Interface running...")
           self.start_listening()
           
           try:
               rospy.spin()
           except KeyboardInterrupt:
               rospy.loginfo("Shutting down...")
           finally:
               self.stop_listening()

   def main():
       interface = WhisperRobotInterface(model_size="base")
       interface.run()

   if __name__ == '__main__':
       main()
   ```

### Lab 2: Advanced Voice Command Processing

1. **Create an advanced voice command processor** (`advanced_voice_processor.py`):
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
           
           # Initialize ROS
           rospy.init_node('advanced_voice_processor', anonymous=True)
           
           # Publishers and Subscribers
           self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
           self.voice_cmd_pub = rospy.Publisher('/parsed_voice_command', String, queue_size=10)
           self.status_pub = rospy.Publisher('/voice_control_status', String, queue_size=10)
           self.feedback_pub = rospy.Publisher('/voice_feedback', String, queue_size=10)
           
           rospy.Subscriber('/scan', LaserScan, self.laser_callback)
           rospy.Subscriber('/voice_command_raw', String, self.voice_command_callback)
           
           # State management
           self.laser_data = None
           self.robot_position = Point(x=0.0, y=0.0, z=0.0)
           
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
           
           # Context memory for disambiguation
           self.context_memory = []
           self.max_context_items = 50
           
           rospy.loginfo("Advanced Voice Processor initialized")
       
       def laser_callback(self, msg: LaserScan):
           """Update laser scan data for contextaware processing"""
           self.laser_data = msg
       
       def voice_command_callback(self, msg: String):
           """Handle voice command from another source (if needed)"""
           # This could be used for preprocessed audio or simulation
           pass
       
       def start_listening(self):
           """Start voice command processing"""
           if self.is_listening:
               return
           
           self.is_listening = True
           
           # Start threads
           self.recording_thread = threading.Thread(target=self._continuous_recording)
           self.recording_thread.daemon = True
           self.recording_thread.start()
           
           self.processing_thread = threading.Thread(target=self._process_commands)
           self.processing_thread.daemon = True
           self.processing_thread.start()
           
           self.status_pub.publish(String(data="Voice control active"))
           rospy.loginfo("Advanced voice processor started")
       
       def stop_listening(self):
           """Stop voice command processing"""
           self.is_listening = False
           self.status_pub.publish(String(data="Voice control inactive"))
       
       def _detect_voice_activity(self, audio_chunk):
           """Enhanced voice activity detection with multiple features"""
           # Convert to numpy
           audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
           
           # Calculate multiple features for robust VAD
           rms = np.sqrt(np.mean(audio_np ** 2))
           
           # Use librosa for more sophisticated features
           try:
               # Spectral features
               stft = librosa.stft(audio_np)
               spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
               spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft))[0]
               
               # Calculate average spectral centroid and rolloff
               avg_centroid = np.mean(spectral_centroids)
               avg_rolloff = np.mean(spectral_rolloff)
               
               # Combined voice activity score
               va_score = (
                   0.4 * (rms / 0.01) +  # Normalize RMS
                   0.3 * (avg_centroid / 2000) +  # Normalize spectral centroid
                   0.3 * (avg_rolloff / 5000)   # Normalize rolloff
               ) / 3.0
               
               return va_score > self.voice_activity_threshold
               
           except:
               # Fallback to simple RMS if librosa fails
               return rms > self.voice_activity_threshold
       
       def _continuous_recording(self):
           """Continuously record audio with enhanced voice activity detection"""
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
           silence_chunks = int(self.min_silence_duration * self.rate / self.chunk)
           min_voice_chunks = int(self.min_voice_duration * self.rate / self.chunk)
           
           try:
               while self.is_listening:
                   data = stream.read(self.chunk, exception_on_overflow=False)
                   
                   if self._detect_voice_activity(data):
                       if not recording:
                           # Potential start of speech
                           rospy.loginfo("Potential speech start detected")
                           recording = True
                           frames = [data]
                           silence_counter = 0
                       else:
                           # Continue recording
                           frames.append(data)
                           silence_counter = 0
                   else:
                       if recording:
                           # Accumulate silence
                           frames.append(data)  # Add to frames for trailing audio
                           silence_counter += 1
                           
                           # End recording if silence exceeds threshold
                           if silence_counter >= silence_chunks:
                               if len(frames) >= min_voice_chunks:  # Ensure minimum speech duration
                                   # Create temp file and add to queue
                                   temp_file = self._save_audio_chunk(frames)
                                   self.command_queue.put(temp_file)
                                   rospy.loginfo(f"Recorded speech segment: {len(frames)} frames")
                               
                               # Reset for next segment
                               recording = False
                               frames = []
                               silence_counter = 0
           except Exception as e:
               rospy.logerr(f"Error in continuous recording: {e}")
           finally:
               stream.stop_stream()
               stream.close()
       
       def _save_audio_chunk(self, frames):
           """Save audio chunk to temporary file"""
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
           """Process audio files in the queue with Whisper"""
           while self.is_listening or not self.command_queue.empty():
               try:
                   audio_file = self.command_queue.get(timeout=1.0)
                   
                   # Transcribe with Whisper
                   transcription = self._transcribe_audio(audio_file)
                   
                   if transcription and transcription.strip():
                       rospy.loginfo(f"Transcribed: {transcription}")
                       
                       # Process with advanced command parsing
                       command = self._parse_advanced_command(transcription)
                       
                       if command:
                           # Validate command with context
                           if self._validate_command(command):
                               self._execute_command(command)
                               self._publish_command(command)
                           else:
                               rospy.logwarn(f"Command validation failed: {command}")
                               self._publish_feedback("Command unsafe or invalid", level="warning")
                       else:
                           rospy.logwarn(f"Could not parse command: {transcription}")
                           self._publish_feedback("Could not understand command", level="error")
                   else:
                       rospy.loginfo("No speech detected")
                   
                   # Clean up temp file
                   if os.path.exists(audio_file):
                       os.remove(audio_file)
                   
                   self.command_queue.task_done()
                   
               except queue.Empty:
                   continue
               except Exception as e:
                   rospy.logerr(f"Error processing command: {e}")
       
       def _transcribe_audio(self, audio_path):
           """Transcribe audio with error handling and retry"""
           max_retries = 3
           for attempt in range(max_retries):
               try:
                   result = self.model.transcribe(audio_path, language="english")
                   return result['text'].strip()
               except Exception as e:
                   rospy.logwarn(f"Transcription attempt {attempt+1} failed: {e}")
                   if attempt == max_retries  1:
                       return ""  # Return empty string after all retries
       
       def _parse_advanced_command(self, text: str) > Optional[VoiceCommand]:
           """Advanced command parsing with context awareness"""
           original_text = text
           
           # Clean text
           text = re.sub(r'[^\w\s]', ' ', text.lower())
           text = ' '.join(text.split())  # Remove extra whitespace
           
           # Define command patterns with parameters
           command_patterns = [
               # Navigation with parameters
               (r'.*\b(move|go|navigate)\b.*\b(forward|ahead|straight)\b.*\b(\d+(?:\.\d+)?)\b.*\b(meter|meters|metre|metres)\b', 
                self._parse_move_distance),
               (r'.*\b(turn|rotate|pivot)\b.*\b(left|right)\b.*\b(\d+(?:\.\d+)?)\b.*\b(degrees|deg)\b', 
                self._parse_turn_degrees),
               (r'.*\b(approach|move to|go to)\b.*\b(\w+)\b', self._parse_approach_object),
               
               # Simple navigation
               (r'.*\b(forward|ahead|straight)\b.*', lambda m, t: VoiceCommand("move_forward", {"distance": 1.0}, 0.8, t)),
               (r'.*\b(backward|reverse|back)\b.*', lambda m, t: VoiceCommand("move_backward", {"distance": 1.0}, 0.8, t)),
               (r'.*\b(left|port)\b.*', lambda m, t: VoiceCommand("turn_left", {"angle": 90.0}, 0.8, t)),
               (r'.*\b(right|starboard)\b.*', lambda m, t: VoiceCommand("turn_right", {"angle": 90.0}, 0.8, t)),
               
               # Object manipulation
               (r'.*\b(pick up|grasp|grab|take|lift)\b.*\b(\w+)\b', self._parse_pick_object),
               (r'.*\b(place|put down|drop|release)\b.*\b(\w+)\b', self._parse_place_object),
               
               # Stop/abort
               (r'.*\b(stop|halt|park|abort|cancel)\b.*', lambda m, t: VoiceCommand("stop", {}, 0.9, t)),
           ]
           
           # Try each pattern
           for pattern, handler in command_patterns:
               match = re.search(pattern, text)
               if match:
                   try:
                       command = handler(match, original_text)
                       if command:
                           return command
                   except Exception as e:
                       rospy.logerr(f"Error in command handler: {e}")
           
           # If no specific pattern matched, try general classification
           return self._classify_general_command(text, original_text)
       
       def _parse_move_distance(self, match, text):
           """Parse distancebased movement command"""
           distance = float(match.group(3))
           return VoiceCommand(
               "move_distance",
               {"direction": "forward", "distance": distance},  # Would need more context to determine direction
               0.7,
               text
           )
       
       def _parse_turn_degrees(self, match, text):
           """Parse degreebased turn command"""
           direction = match.group(2)
           angle = float(match.group(3))
           return VoiceCommand(
               "turn_degrees",
               {"direction": direction, "angle": angle},
               0.7,
               text
           )
       
       def _parse_approach_object(self, match, text):
           """Parse approach object command"""
           object_name = match.group(2)
           return VoiceCommand(
               "approach_object",
               {"object_name": object_name},
               0.6,
               text
           )
       
       def _parse_pick_object(self, match, text):
           """Parse pick up object command"""
           object_name = match.group(2)
           return VoiceCommand(
               "pick_object",
               {"object_name": object_name},
               0.6,
               text
           )
       
       def _parse_place_object(self, match, text):
           """Parse place object command"""
           object_name = match.group(2)
           return VoiceCommand(
               "place_object", 
               {"object_name": object_name, "location": "current_position"},  # Would need more context
               0.6,
               text
           )
       
       def _classify_general_command(self, clean_text: str, original_text: str):
           """Classify commands that don't match specific patterns"""
           # Use keyword matching for general classification
           text_lower = clean_text.lower()
           
           # Check for navigationrelated keywords
           if any(keyword in text_lower for keyword in ['move', 'go', 'walk', 'drive', 'navigate', 'forward', 'backward', 'left', 'right']):
               return VoiceCommand("navigate", {}, 0.5, original_text)
           elif any(keyword in text_lower for keyword in ['turn', 'rotate', 'pivot', 'spin', 'around']):
               return VoiceCommand("rotate", {}, 0.5, original_text)
           elif any(keyword in text_lower for keyword in ['stop', 'halt', 'pause', 'wait', 'freeze']):
               return VoiceCommand("stop", {}, 0.8, original_text)
           elif any(keyword in text_lower for keyword in ['pick', 'grasp', 'grab', 'take', 'lift', 'hold']):
               return VoiceCommand("manipulate", {"action": "grasp"}, 0.5, original_text)
           elif any(keyword in text_lower for keyword in ['place', 'put', 'set', 'drop', 'release']):
               return VoiceCommand("manipulate", {"action": "release"}, 0.5, original_text)
           else:
               return None  # Unknown command
       
       def _validate_command(self, command: VoiceCommand) > bool:
           """Validate command considering current state and environment"""
           # Check if environment is safe for this command
           if command.action == "move_forward" and self.laser_data:
               # Check for obstacles ahead (simplified  check middle third of scan)
               middle_start = len(self.laser_data.ranges) // 3
               middle_end = 2 * len(self.laser_data.ranges) // 3
               middle_ranges = self.laser_data.ranges[middle_start:middle_end]
               
               # Filter invalid ranges
               valid_ranges = [r for r in middle_ranges if not (np.isinf(r) or np.isnan(r))]
               
               if valid_ranges and min(valid_ranges) < 0.5:  # 0.5m safety distance
                   rospy.logwarn("Obstacle detected ahead, aborting forward movement")
                   return False
           
           # Additional validation checks could go here
           #  Check robot state (battery level, joint limits, etc.)
           #  Check environmental constraints
           #  Check safety zones
           
           return True
       
       def _execute_command(self, command: VoiceCommand):
           """Execute validated command"""
           rospy.loginfo(f"Executing: {command.action} with params: {command.parameters}")
           
           if command.action in ["move_forward", "move_distance"]:
               self._execute_movement_command(command)
           elif command.action in ["turn_left", "turn_right", "turn_degrees"]:
               self._execute_rotation_command(command)
           elif command.action in ["stop", "halt"]:
               self._execute_stop_command()
           elif command.action in ["approach_object", "navigate_to"]:
               self._execute_navigation_command(command)
           elif command.action in ["pick_object", "place_object"]:
               self._execute_manipulation_command(command)
           else:
               rospy.logwarn(f"Unknown command action: {command.action}")
       
       def _execute_movement_command(self, command: VoiceCommand):
           """Execute movement command"""
           twist = Twist()
           distance = command.parameters.get("distance", 1.0)
           speed = 0.3  # m/s
           
           # Calculate time needed: distance / speed
           duration = distance / speed
           
           # Move for calculated duration
           twist.linear.x = speed
           rate = rospy.Rate(10)  # 10 Hz
           start_time = rospy.Time.now()
           
           while (rospy.Time.now()  start_time).to_sec() < duration and not rospy.is_shutdown():
               self.cmd_vel_pub.publish(twist)
               rate.sleep()
           
           # Stop robot
           self._send_stop_command()
       
       def _execute_rotation_command(self, command: VoiceCommand):
           """Execute rotation command"""
           twist = Twist()
           angle = command.parameters.get("angle", 90.0)
           direction = command.parameters.get("direction", "left")
           
           # Convert angle to time (assuming 0.5 rad/s angular velocity)
           angular_vel = 0.5  # rad/s
           duration = np.radians(angle) / angular_vel
           
           # Set rotation direction
           twist.angular.z = angular_vel if direction == "left" else angular_vel
           
           rate = rospy.Rate(10)
           start_time = rospy.Time.now()
           
           while (rospy.Time.now()  start_time).to_sec() < duration and not rospy.is_shutdown():
               self.cmd_vel_pub.publish(twist)
               rate.sleep()
           
           # Stop rotation
           self._send_stop_command()
       
       def _execute_stop_command(self):
           """Execute stop command"""
           self._send_stop_command()
       
       def _send_stop_command(self):
           """Send stop command to robot"""
           twist = Twist()
           self.cmd_vel_pub.publish(twist)
       
       def _publish_command(self, command: VoiceCommand):
           """Publish parsed command"""
           cmd_dict = {
               "action": command.action,
               "parameters": command.parameters,
               "confidence": command.confidence,
               "original_text": command.original_text,
               "timestamp": rospy.Time.now().to_sec()
           }
           
           cmd_msg = String()
           cmd_msg.data = json.dumps(cmd_dict)
           self.voice_cmd_pub.publish(cmd_msg)
       
       def _publish_feedback(self, message: str, level: str = "info"):
           """Publish voice processing feedback"""
           feedback_msg = String()
           feedback_msg.data = json.dumps({
               "message": message,
               "level": level,
               "timestamp": rospy.Time.now().to_sec()
           })
           self.feedback_pub.publish(feedback_msg)
           
           # Log message with appropriate level
           if level == "error":
               rospy.logerr(message)
           elif level == "warning":
               rospy.logwarn(message)
           else:
               rospy.loginfo(message)
       
       def run(self):
           """Run the main processing loop"""
           rospy.loginfo("Starting advanced voice processor...")
           self.start_listening()
           
           try:
               rospy.spin()
           except KeyboardInterrupt:
               rospy.loginfo("Shutting down voice processor...")
           finally:
               self.stop_listening()

   def main():
       processor = AdvancedVoiceProcessor(model_size="base")
       processor.run()

   if __name__ == '__main__':
       main()
   ```

### Lab 3: Creating a Voice Control Safety System

1. **Create a safetyaware voice control system** (`voice_control_safety.py`):
   ```python
   #!/usr/bin/env python3

   import rospy
   import pyaudio
   import numpy as np
   import threading
   import queue
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import LaserScan, Imu
   from voice_recognition_module import WhisperRobotInterface
   import time
   import json

   class VoiceControlSafetySystem:
       def __init__(self):
           rospy.init_node('voice_control_safety_system', anonymous=True)
           
           # Initialize whisper interface
           self.whisper_interface = WhisperRobotInterface(model_size="base")
           
           # Publishers
           self.safed_cmd_pub = rospy.Publisher('/cmd_vel_safe', Twist, queue_size=10)
           self.safety_status_pub = rospy.Publisher('/voice_control_safety_status', String, queue_size=10)
           self.emergency_stop_pub = rospy.Publisher('/emergency_stop', Bool, queue_size=10)
           
           # Subscribers
           self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
           self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
           self.voice_cmd_sub = rospy.Subscriber('/voice_command', String, self.voice_command_callback)
           
           # Safety parameters
           self.safety_distance = rospy.get_param('~safety_distance', 0.5)  # meters
           self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.5)  # m/s
           self.max_angular_speed = rospy.get_param('~max_angular_speed', 0.8)  # rad/s
           self.emergency_stop_enabled = True
           
           # State variables
           self.laser_data = None
           self.imu_data = None
           self.current_command = None
           self.is_safe_to_proceed = True
           self.last_safe_check = 0
           self.safe_check_interval = 0.1  # seconds
           
           # Emergency stop state
           self.emergency_stop_active = False
           self.last_emergency_time = 0
           self.emergency_reset_time = 5.0  # seconds to wait before reset
           
           # Command queue for processing
           self.command_queue = queue.Queue()
           
           rospy.loginfo("Voice Control Safety System initialized")
       
       def laser_callback(self, msg: LaserScan):
           """Update laser scan data"""
           self.laser_data = msg
       
       def imu_callback(self, msg: Imu):
           """Update IMU data"""
           self.imu_data = msg
       
       def voice_command_callback(self, msg: String):
           """Process voice command with safety validation"""
           if self.emergency_stop_active:
               rospy.logwarn("Emergency stop active, ignoring voice command")
               self.publish_safety_status("EMERGENCY_STOP_ACTIVE")
               return
           
           try:
               command_data = json.loads(msg.data)
               command_type = command_data.get('action', 'unknown')
               command_params = command_data.get('parameters', {})
               
               # Validate command for safety
               if self.is_command_safe(command_type, command_params):
                   # Add to processing queue
                   self.command_queue.put({
                       'type': command_type,
                       'params': command_params,
                       'original_message': msg.data,
                       'timestamp': rospy.Time.now().to_sec()
                   })
                   rospy.loginfo(f"Safe command queued: {command_type}")
               else:
                   rospy.logwarn(f"Unsafe command blocked: {command_type}")
                   self.publish_feedback("Command blocked for safety reasons", "warning")
                   
           except json.JSONDecodeError:
               # Handle as simple command string
               command_str = msg.data
               
               # Simple validation based on command content
               if self.is_simple_command_safe(command_str):
                   self.command_queue.put({
                       'type': 'simple_command',
                       'params': {'command': command_str},
                       'original_message': msg.data,
                       'timestamp': rospy.Time.now().to_sec()
                   })
                   rospy.loginfo(f"Safe command queued: {command_str}")
               else:
                   rospy.logwarn(f"Unsafe simple command blocked: {command_str}")
                   self.publish_feedback("Command blocked for safety reasons", "warning")
       
       def is_command_safe(self, command_type: str, parameters: dict) > bool:
           """Validate if command is safe to execute"""
           # Check for emergency stop keywords
           if any(keyword in command_type.lower() for keyword in ['emergency', 'stop', 'halt', 'danger']):
               return True  # These are valid safety commands
           
           # Check environmental safety for movement commands
           if command_type in ['move_forward', 'move_backward', 'move_distance']:
               return self.is_movement_safe(parameters)
           
           # Check for rotation commands
           if command_type in ['turn_left', 'turn_right', 'turn_degrees']:
               # Rotations are generally safer than translations
               return True
           
           # Check for manipulation commands in safe areas
           if command_type in ['pick_object', 'place_object']:
               return self.is_manipulation_safe(parameters)
           
           return True  # Default to safe for other commands
       
       def is_simple_command_safe(self, command_str: str) > bool:
           """Validate simple string command for safety"""
           command_lower = command_str.lower()
           
           # Check for dangerous words that should trigger safety checks
           dangerous_indicators = [
               'danger', 'emergency', 'crash', 'break', 'smash', 
               'damage', 'destroy', 'hurt', 'injure', 'kill'
           ]
           
           # Check if command has dangerous indicators but no safety context
           has_danger = any(indicator in command_lower for indicator in dangerous_indicators)
           has_request = any(phrase in command_lower for phrase in ['please', 'could you', 'can you'])
           has_stop_phrase = any(phrase in command_lower for phrase in ['stop', 'emergency', 'help'])
           
           if has_danger and not has_request and not has_stop_phrase:
               rospy.logwarn(f"Potentially dangerous command detected: {command_str}")
               return False
           
           # Allow commands that include safety words with context
           if has_stop_phrase:
               return True  # Stop requests are always safe
           
           # Check if movement command is safe
           if any(phrase in command_lower for phrase in ['forward', 'ahead', 'move', 'go']):
               return self.is_movement_safe({'direction': 'forward', 'distance': 1.0})
           
           return True
       
       def is_movement_safe(self, params: dict) > bool:
           """Check if movement command is safe"""
           if not self.laser_data:
               rospy.logwarn("No laser data available, assuming unsafe for movement")
               return False
           
           # Check for obstacles in the movement direction
           if params.get('direction') == 'forward' or params.get('action') == 'move_forward':
               # Check front sector (±30 degrees)
               front_sector = self.laser_data.ranges[
                   len(self.laser_data.ranges)//2  30 :
                   len(self.laser_data.ranges)//2 + 30
               ]
               
               # Filter valid ranges
               valid_ranges = [r for r in front_sector if not (np.isinf(r) or np.isnan(r))]
               
               if valid_ranges:
                   min_distance = min(valid_ranges)
                   if min_distance < self.safety_distance:
                       rospy.logwarn(f"Obstacle detected: {min_distance:.2f}m ahead, unsafe to move forward")
                       return False
           
           # Check for other directions based on command
           return True
       
       def is_manipulation_safe(self, params: dict) > bool:
           """Check if manipulation command is safe"""
           # For manipulation, we'd need to check joint limits, collisions, etc.
           # This is a simplified check  in practice, would use MoveIt! collision checking
           return True
       
       def process_command_queue(self):
           """Process commands from the safetyvalidated queue"""
           rate = rospy.Rate(10)  # 10 Hz
           
           while not rospy.is_shutdown():
               try:
                   # Check for commands to process
                   try:
                       cmd = self.command_queue.get_nowait()
                       
                       # Execute the validated command
                       self.execute_validated_command(cmd)
                       
                   except queue.Empty:
                       pass  # No commands to process
                   
                   # Perform periodic safety checks
                   current_time = rospy.Time.now().to_sec()
                   if (current_time  self.last_safe_check) > self.safe_check_interval:
                       self.perform_periodic_safety_check()
                       self.last_safe_check = current_time
                   
                   rate.sleep()
                   
               except KeyboardInterrupt:
                   rospy.loginfo("Voice Control Safety System interrupted")
                   break
       
       def execute_validated_command(self, cmd: dict):
           """Execute a safetyvalidated command"""
           command_type = cmd['type']
           
           # Create Twist message based on command type
           twist_cmd = Twist()
           
           if command_type == "move_forward":
               twist_cmd.linear.x = min(self.max_linear_speed, cmd['params'].get('speed', 0.3))
           elif command_type == "move_backward":
               twist_cmd.linear.x = min(self.max_linear_speed, cmd['params'].get('speed', 0.3))
           elif command_type == "turn_left":
               twist_cmd.angular.z = min(self.max_angular_speed, cmd['params'].get('speed', 0.5))
           elif command_type == "turn_right":
               twist_cmd.angular.z = min(self.max_angular_speed, cmd['params'].get('speed', 0.5))
           elif command_type == "stop":
               twist_cmd.linear.x = 0.0
               twist_cmd.angular.z = 0.0
           
           # Apply safety limits
           twist_cmd.linear.x = max(self.max_linear_speed, min(self.max_linear_speed, twist_cmd.linear.x))
           twist_cmd.angular.z = max(self.max_angular_speed, min(self.max_angular_speed, twist_cmd.angular.z))
           
           # Publish the command
           self.safed_cmd_pub.publish(twist_cmd)
           rospy.loginfo(f"Executed safe command: {command_type}")
       
       def perform_periodic_safety_check(self):
           """Perform periodic safety checks on the environment"""
           if not self.laser_data:
               return
           
           # Check for emergency conditions
           emergency_detected = self.check_for_emergency_conditions()
           
           if emergency_detected:
               self.trigger_emergency_stop()
           elif self.emergency_stop_active:
               # Check if it's time to reset emergency stop
               if rospy.Time.now().to_sec()  self.last_emergency_time > self.emergency_reset_time:
                   self.reset_emergency_stop()
       
       def check_for_emergency_conditions(self) > bool:
           """Check if emergency conditions exist"""
           if not self.laser_data:
               return False
           
           # Check if obstacles are very close (within safety distance)
           close_ranges = [r for r in self.laser_data.ranges 
                          if not (np.isinf(r) or np.isnan(r)) and r < self.safety_distance * 0.5]
           
           return len(close_ranges) > 3  # More than 3 close obstacles is an emergency
       
       def trigger_emergency_stop(self):
           """Trigger emergency stop"""
           if not self.emergency_stop_active:
               self.emergency_stop_active = True
               self.last_emergency_time = rospy.Time.now().to_sec()
               
               # Publish emergency stop message
               emergency_msg = Bool()
               emergency_msg.data = True
               self.emergency_stop_pub.publish(emergency_msg)
               
               # Stop the robot immediately
               stop_cmd = Twist()
               self.safed_cmd_pub.publish(stop_cmd)
               
               rospy.logerr("EMERGENCY STOP ACTIVATED")
               self.publish_safety_status("EMERGENCY_STOP_ACTIVATED")
       
       def reset_emergency_stop(self):
           """Reset emergency stop"""
           if self.emergency_stop_active:
               self.emergency_stop_active = False
               
               # Publish reset message
               reset_msg = Bool()
               reset_msg.data = False
               self.emergency_stop_pub.publish(reset_msg)
               
               rospy.loginfo("Emergency stop reset")
               self.publish_safety_status("EMERGENCY_STOP_RESET")
       
       def publish_safety_status(self, status: str):
           """Publish safety system status"""
           status_msg = String()
           status_msg.data = json.dumps({
               "status": status,
               "timestamp": rospy.Time.now().to_sec(),
               "emergency_active": self.emergency_stop_active,
               "safety_distance": self.safety_distance
           })
           self.safety_status_pub.publish(status_msg)
       
       def publish_feedback(self, message: str, level: str = "info"):
           """Publish feedback message"""
           feedback_msg = String()
           feedback_msg.data = json.dumps({
               "message": message,
               "level": level,
               "timestamp": rospy.Time.now().to_sec()
           })
           # Publish to a generic feedback topic (would need to be defined elsewhere)
           # For this example, we'll just log it
           if level == "error":
               rospy.logerr(message)
           elif level == "warning":
               rospy.logwarn(message)
           else:
               rospy.loginfo(message)
       
       def run(self):
           """Run the safety system"""
           rospy.loginfo("Voice Control Safety System starting...")
           self.publish_safety_status("SYSTEM_STARTED")
           
           try:
               self.process_command_queue()
           except KeyboardInterrupt:
               rospy.loginfo("Shutting down Voice Control Safety System...")

   def main():
       safety_system = VoiceControlSafetySystem()
       safety_system.run()

   if __name__ == '__main__':
       main()
   ```

## Runnable Code Example

Here's a complete working example of the voicetoaction system with safety validation:

```python
#!/usr/bin/env python3
# complete_voice_control_system.py

import rospy
import openai
import whisper
import torch
import pyaudio
import numpy as np
import json
import threading
import queue
import time
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan, Imu
from cv_bridge import CvBridge

class CompleteVoiceControlSystem:
    """Complete voice control system with safety validation"""
    
    def __init__(self, api_key: str):
        rospy.init_node('complete_voice_control_system', anonymous=True)
        
        # Initialize Whisper model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model = whisper.load_model("base").to(self.device)
        
        # Initialize OpenAI client if API key provided
        self.openai_client = None
        if api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize OpenCV bridge for potential visual integration
        self.cv_bridge = CvBridge()
        
        # Audio parameters
        self.audio = pyaudio.PyAudio()
        self.rate = 16000
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.voice_transcript_pub = rospy.Publisher('/voice_transcript', String, queue_size=10)
        self.parsed_command_pub = rospy.Publisher('/parsed_voice_command', String, queue_size=10)
        self.system_status_pub = rospy.Publisher('/voice_system_status', String, queue_size=10)
        self.safety_status_pub = rospy.Publisher('/voice_safety_status', String, queue_size=10)
        
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        
        # System state
        self.laser_data = None
        self.imu_data = None
        self.is_listening = False
        self.recording_thread = None
        self.processing_thread = None
        
        # Safety parameters
        self.safety_distance = rospy.get_param('~safety_distance', 0.5)
        self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.4)
        self.max_angular_speed = rospy.get_param('~max_angular_speed', 0.6)
        
        # Voice activity detection
        self.voice_threshold = 0.005
        self.min_voice_duration = 0.3
        self.min_silence_duration = 0.8
        
        # Command and safety queues
        self.command_queue = queue.Queue()
        self.safety_queue = queue.Queue()
        
        # Internal state
        self.current_command = None
        self.emergency_stop_active = False
        
        rospy.loginfo("Complete Voice Control System initialized")
    
    def laser_callback(self, msg: LaserScan):
        """Update laser scan data"""
        self.laser_data = msg
    
    def imu_callback(self, msg: Imu):
        """Update IMU data"""
        self.imu_data = msg
    
    def start_system(self):
        """Start the complete voice control system"""
        self.is_listening = True
        
        # Start audio recording thread
        self.recording_thread = threading.Thread(target=self._audio_recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._command_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start safety monitoring thread
        self.safety_thread = threading.Thread(target=self._safety_monitoring_loop)
        self.safety_thread.daemon = True
        self.safety_thread.start()
        
        rospy.loginfo("Complete Voice Control System started")
        self._publish_status("System active and listening")
    
    def stop_system(self):
        """Stop the complete voice control system"""
        self.is_listening = False
        self._publish_status("System stopped")
        rospy.loginfo("Complete Voice Control System stopped")
    
    def _audio_recording_loop(self):
        """Audio recording loop with voice activity detection"""
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
                data = stream.read(self.chunk, exception_on_overflow=False)
                
                if self._is_voice_active(data):
                    if not recording:
                        # Start recording when voice is detected
                        voice_active_count += 1
                        if voice_active_count >= min_voice_frames:
                            recording = True
                            frames = [data] * min_voice_frames  # Include preroll
                            voice_active_count = 0
                            silence_count = 0
                            rospy.loginfo("Voice activity confirmed, recording started")
                    else:
                        # Continue recording
                        frames.append(data)
                        silence_count = 0
                        voice_active_count = 0
                else:
                    if recording:
                        # Add to silence counter
                        frames.append(data)
                        silence_count += 1
                        
                        if silence_count >= min_silence_frames:
                            # End of speech detected
                            if len(frames) >= min_voice_frames:
                                # Save to temp file and add to processing queue
                                temp_file = self._save_audio_frames(frames)
                                self.command_queue.put(temp_file)
                                rospy.loginfo(f"Recorded audio segment with {len(frames)} frames")
                            
                            # Reset for next recording
                            recording = False
                            frames = []
                            silence_count = 0
                            voice_active_count = 0
                    else:
                        # Reset voice counter when not recording
                        voice_active_count = 0
        except Exception as e:
            rospy.logerr(f"Error in audio recording loop: {e}")
        finally:
            stream.stop_stream()
            stream.close()
    
    def _is_voice_active(self, audio_data):
        """Detect voice activity in audio chunk"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_np ** 2))
        return rms > self.voice_threshold
    
    def _save_audio_frames(self, frames):
        """Save audio frames to temporary file"""
        import tempfile
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
    
    def _command_processing_loop(self):
        """Process commands from the queue"""
        while self.is_listening or not self.command_queue.empty():
            try:
                audio_file = self.command_queue.get(timeout=1.0)
                
                # Transcribe audio
                transcription = self._transcribe_audio(audio_file)
                
                if transcription and transcription.strip():
                    rospy.loginfo(f"Transcribed: {transcription}")
                    
                    # Publish transcription
                    transcript_msg = String()
                    transcript_msg.data = transcription
                    self.voice_transcript_pub.publish(transcript_msg)
                    
                    # Parse and validate command
                    parsed_command = self._parse_and_validate_command(transcription)
                    
                    if parsed_command:
                        # Add to safety queue for validation
                        self.safety_queue.put(parsed_command)
                        
                        # Publish parsed command
                        cmd_msg = String()
                        cmd_msg.data = json.dumps(parsed_command)
                        self.parsed_command_pub.publish(cmd_msg)
                
                # Clean up temp file
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Error processing command: {e}")
    
    def _safety_monitoring_loop(self):
        """Monitor safety and execute validated commands"""
        rate = rospy.Rate(20)  # 20 Hz
        
        while self.is_listening or not self.safety_queue.empty():
            try:
                # Check for commands to validate and execute
                try:
                    cmd = self.safety_queue.get_nowait()
                    
                    # Validate command for safety
                    if self._is_command_safe(cmd):
                        # Execute command
                        self._execute_command(cmd)
                        rospy.loginfo(f"Executed safe command: {cmd['action']}")
                    else:
                        rospy.logwarn(f"Unsafe command blocked: {cmd['action']}")
                        self._publish_safety_status(f"Command blocked: {cmd['action']}")
                        
                except queue.Empty:
                    pass  # No commands to process
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in safety monitoring: {e}")
    
    def _transcribe_audio(self, audio_file_path) > str:
        """Transcribe audio using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_file_path, language="english")
            return result['text'].strip()
        except Exception as e:
            rospy.logerr(f"Transcription error: {e}")
            return ""
    
    def _parse_and_validate_command(self, transcription: str) > Optional[Dict]:
        """Parse transcription into command with context"""
        if not transcription:
            return None
        
        # Get environmental context
        context = self._get_environmental_context()
        
        # Use OpenAI for advanced command parsing if available
        if self.openai_client:
            return self._parse_with_openai(transcription, context)
        else:
            # Use basic parsing
            return self._parse_basic_command(transcription)
    
    def _get_environmental_context(self) > Dict:
        """Get environmental context for command validation"""
        context = {
            "robot_state": {
                "has_laser_data": self.laser_data is not None,
                "has_imu_data": self.imu_data is not None
            }
        }
        
        # Add laserbased context if available
        if self.laser_data:
            # Check front, left, right sectors
            n_ranges = len(self.laser_data.ranges)
            front_ranges = self.laser_data.ranges[n_ranges//2  30:n_ranges//2 + 30]
            left_ranges = self.laser_data.ranges[:n_ranges//4]
            right_ranges = self.laser_data.ranges[3*n_ranges//4:]
            
            # Filter valid ranges
            valid_front = [r for r in front_ranges if not (np.isinf(r) or np.isnan(r))]
            valid_left = [r for r in left_ranges if not (np.isinf(r) or np.isnan(r))]
            valid_right = [r for r in right_ranges if not (np.isinf(r) or np.isnan(r))]
            
            context["surroundings"] = {
                "front_clear_m": min(valid_front) if valid_front else float('inf'),
                "left_clear_m": min(valid_left) if valid_left else float('inf'),
                "right_clear_m": min(valid_right) if valid_right else float('inf')
            }
        
        return context
    
    def _parse_with_openai(self, transcription: str, context: Dict) > Optional[Dict]:
        """Use OpenAI to parse command with context"""
        try:
            prompt = f"""
            Transcription: "{transcription}"
            
            Environment Context: {json.dumps(context, indent=2)}
            
            Parse this natural language command into a robot action.
            Consider the environmental context when interpreting the command.
            If the command seems unsafe given the environmental context, mark it as unsafe.
            
            Return only a JSON object in this format:
            {{
              "action": "move_forward|move_backward|turn_left|turn_right|stop|pick_object|place_object|other",
              "parameters": {{
                "linear_speed": float,
                "angular_speed": float,
                "distance": float,
                "angle": float,
                "object_name": str
              }},
              "confidence": float,
              "is_safe": boolean,
              "reasoning": "brief explanation"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a robotics command parser. Parse natural language commands into robot actions. Consider environmental context for safety."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response if wrapped in code blocks
            if response_text.startswith('```'):
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != 1 and end_idx != 1:
                    response_text = response_text[start_idx:end_idx]
            
            command_data = json.loads(response_text)
            return command_data
            
        except Exception as e:
            rospy.logerr(f"Error with OpenAI command parsing: {e}")
            # Fall back to basic parsing
            return self._parse_basic_command(transcription)
    
    def _parse_basic_command(self, transcription: str) > Optional[Dict]:
        """Basic regularexpressionbased command parsing"""
        text_lower = transcription.lower()
        
        # Navigation commands
        if any(word in text_lower for word in ['forward', 'ahead', 'straight', 'go']):
            return {
                "action": "move_forward",
                "parameters": {"linear_speed": 0.3, "distance": 1.0},
                "confidence": 0.7,
                "is_safe": True,
                "reasoning": "Basic forward movement command"
            }
        elif any(word in text_lower for word in ['backward', 'back', 'reverse']):
            return {
                "action": "move_backward", 
                "parameters": {"linear_speed": 0.3, "distance": 1.0},
                "confidence": 0.7,
                "is_safe": True,
                "reasoning": "Basic backward movement command"
            }
        elif any(word in text_lower for word in ['left', 'port']):
            return {
                "action": "turn_left",
                "parameters": {"angular_speed": 0.5, "angle": 90.0},
                "confidence": 0.7,
                "is_safe": True,
                "reasoning": "Basic left turn command"
            }
        elif any(word in text_lower for word in ['right', 'starboard']):
            return {
                "action": "turn_right",
                "parameters": {"angular_speed": 0.5, "angle": 90.0},
                "confidence": 0.7,
                "is_safe": True,
                "reasoning": "Basic right turn command"
            }
        elif any(word in text_lower for word in ['stop', 'halt', 'pause']):
            return {
                "action": "stop",
                "parameters": {},
                "confidence": 0.9,
                "is_safe": True,
                "reasoning": "Stop command is always safe"
            }
        else:
            return {
                "action": "unknown",
                "parameters": {},
                "confidence": 0.3,
                "is_safe": False,
                "reasoning": "Unknown command"
            }
    
    def _is_command_safe(self, command: Dict) > bool:
        """Validate if command is safe to execute in current environment"""
        if not self.laser_data:
            # If no sensor data, be conservative
            action = command.get('action', 'unknown')
            if action in ['move_forward', 'move_backward', 'move_distance']:
                return command.get('is_safe', False)  # Trust parsed safety flag
            else:
                return True  # Other commands are generally safe
        
        # Check for movement commands with obstacle consideration
        action = command.get('action')
        if action in ['move_forward', 'move_distance'] and command.get('parameters', {}).get('linear_speed', 0) > 0:
            # Check front for obstacles
            n_ranges = len(self.laser_data.ranges)
            front_sector = self.laser_data.ranges[n_ranges//2  30 : n_ranges//2 + 30]
            valid_ranges = [r for r in front_sector if not (np.isinf(r) or np.isnan(r))]
            
            if valid_ranges and min(valid_ranges) < self.safety_distance:
                rospy.logwarn(f"Obstacle detected ahead ({min(valid_ranges):.2f}m), command unsafe")
                return False
        
        elif action == 'move_backward' and command.get('parameters', {}).get('linear_speed', 0) < 0:
            # Check rear for obstacles (simplified  check back 30 degrees)
            n_ranges = len(self.laser_data.ranges)
            back_sector = self.laser_data.ranges[3*n_ranges//4  15 : n_ranges] + self.laser_data.ranges[0 : n_ranges//4 + 15]
            valid_ranges = [r for r in back_sector if not (np.isinf(r) or np.isnan(r))]
            
            if valid_ranges and min(valid_ranges) < self.safety_distance:
                rospy.logwarn(f"Obstacle detected behind ({min(valid_ranges):.2f}m), command unsafe")
                return False
        
        # Check if command has been marked unsafe by parser
        if not command.get('is_safe', True):
            return False
        
        return True
    
    def _execute_command(self, command: Dict):
        """Execute validated command"""
        action = command.get('action', 'unknown')
        params = command.get('parameters', {})
        
        twist_cmd = Twist()
        
        if action == 'move_forward':
            linear_speed = params.get('linear_speed', 0.3)
            twist_cmd.linear.x = min(linear_speed, self.max_linear_speed)
        elif action == 'move_backward':
            linear_speed = params.get('linear_speed', 0.3)
            twist_cmd.linear.x = min(linear_speed, self.max_linear_speed)
        elif action == 'turn_left':
            angular_speed = params.get('angular_speed', 0.5)
            twist_cmd.angular.z = min(angular_speed, self.max_angular_speed)
        elif action == 'turn_right':
            angular_speed = params.get('angular_speed', 0.5)
            twist_cmd.angular.z = min(angular_speed, self.max_angular_speed)
        elif action == 'stop':
            # Twist is already zeroed
            pass
        else:
            # For unknown actions, just stop
            pass
        
        # Apply safety limits
        twist_cmd.linear.x = max(self.max_linear_speed, min(self.max_linear_speed, twist_cmd.linear.x))
        twist_cmd.angular.z = max(self.max_angular_speed, min(self.max_angular_speed, twist_cmd.angular.z))
        
        # Publish command
        self.cmd_vel_pub.publish(twist_cmd)
    
    def _publish_status(self, status: str):
        """Publish system status"""
        status_msg = String()
        status_msg.data = json.dumps({
            "status": status,
            "timestamp": rospy.Time.now().to_sec()
        })
        self.system_status_pub.publish(status_msg)
    
    def _publish_safety_status(self, status: str):
        """Publish safety status"""
        safety_msg = String()
        safety_msg.data = json.dumps({
            "status": status,
            "timestamp": rospy.Time.now().to_sec(),
            "emergency_active": self.emergency_stop_active
        })
        self.safety_status_pub.publish(safety_msg)
    
    def run(self):
        """Run the complete system"""
        self.start_system()
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down Complete Voice Control System...")
        finally:
            self.stop_system()

def main():
    # Get OpenAI API key from parameter or user input
    api_key = rospy.get_param('~openai_api_key', '')
    if not api_key:
        api_key = input("Enter OpenAI API key (or press Enter to skip): ").strip()
    
    system = CompleteVoiceControlSystem(api_key if api_key else None)
    system.run()

if __name__ == '__main__':
    main()
```

### Launch file for the complete system:

```xml
<launch>
  <! Complete Voice Control System >
  <node name="complete_voice_control_system" pkg="robot_voice_control" type="complete_voice_control_system.py" output="screen">
    <param name="openai_api_key" value="" />
    <param name="safety_distance" value="0.5" />
    <param name="max_linear_speed" value="0.4" />
    <param name="max_angular_speed" value="0.6" />
  </node>
  
  <! Example sensor nodes (needed for safety validation) >
  <node name="fake_laser_scan" pkg="topic_tools" type="relay" args="/scan /fake_scan" />
  <node name="fake_imu" pkg="topic_tools" type="relay" args="/imu/data /fake_imu" />
  
  <! TF broadcasters for coordinate systems >
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0 0 0 0 0 0 1 base_link laser_frame" />
  <node name="static_transform_publisher" pkg="tf2_ros" type="static_transform_publisher" 
        args="0 0 0 0 0 0 1 base_link imu_link" />
</launch>
```

## Miniproject

Create a complete voiceactivated robot system that:

1. Implements Whisperbased speech recognition for robot commands
2. Integrates with ROS navigation stack for voiceguided navigation
3. Creates a natural language interface for robot manipulation tasks
4. Implements safety validation for all voice commands
5. Develops contextaware command interpretation using LLMs
6. Creates multimodal feedback system (voice + visual)
7. Evaluates system performance with various speakers and accents
8. Implements fallback mechanisms for misunderstood commands

Your project should include:
 Complete Whisper integration with audio preprocessing
 Natural language command parsing and validation
 Safety system to prevent dangerous robot movements
 Contextaware interpretation using environmental sensors
 Voice and visual feedback mechanisms
 Performance evaluation metrics
 Demo scenarios with various natural language commands

## Summary

This chapter covered voicetoaction systems using OpenAI Whisper for robotics:

 **Whisper Integration**: Using OpenAI's Whisper model for speech recognition in robotics
 **Audio Processing**: Techniques for realtime audio processing and voice activity detection  
 **Command Parsing**: Converting natural language to robot commands
 **Safety Validation**: Ensuring voice commands are safe before execution
 **Environmental Context**: Using sensor data to validate command appropriateness
 **Multimodal Integration**: Combining voice with other sensory inputs for robust operation
 **Realtime Processing**: Optimizing for realtime robotic interaction

Voice command systems enable more natural humanrobot interaction, allowing users to command robots using everyday language rather than specialized interfaces. However, special attention must be paid to safety, validation, and environmental context to ensure reliable operation in dynamic environments.