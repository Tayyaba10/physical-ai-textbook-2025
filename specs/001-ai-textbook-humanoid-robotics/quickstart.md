# Quickstart Guide: AI-Native Textbook Platform

## Overview
This guide will help you set up and run the AI-Native Textbook platform for Physical AI & Humanoid Robotics locally. The platform consists of a Docusaurus frontend for the textbook content and a FastAPI backend for services like the RAG chatbot and authentication.

## Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.10+
- ROS 2 Humble Hawksbill (for running code examples)
- Isaac Sim 2024.x (for simulation examples)
- Access to OpenAI API (for RAG chatbot)
- Access to Qdrant Cloud (for vector storage)

## Frontend Setup (Docusaurus)

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Development Server
```bash
npm start
```

This will start the Docusaurus development server at http://localhost:3000

### 3. Build for Production
```bash
npm run build
```

## Backend Setup (FastAPI)

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create a `.env` file in the backend directory:
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_database_url
SECRET_KEY=your_secret_key_for_auth
```

### 4. Run the Backend Server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

The backend will be available at http://localhost:8000

## Initial Content Setup

### 1. Prepare Textbook Content
The textbook content is organized in the `docs/` directory with 20 chapters across 4 modules:

```
docs/
├── module-1-robotic-nervous-system/
│   ├── ch1-why-ros2.md
│   ├── ch2-nodes-topics-actions.md
│   ├── ch3-bridging-ai-agents.md
│   ├── ch4-urdf-xacro-modeling.md
│   └── ch5-building-launching-packages.md
├── module-2-digital-twin/
│   ├── ch6-physics-simulation.md
│   ├── ch7-gazebo-setup-building.md
│   ├── ch8-simulating-sensors.md
│   ├── ch9-unity-hdrp-visualization.md
│   └── ch10-debugging-simulations.md
├── module-3-ai-robot-brain/
│   ├── ch11-nvidia-isaac-overview.md
│   ├── ch12-isaac-sim-photorealistic.md
│   ├── ch13-isaac-ros-accelerated-vslam.md
│   ├── ch14-bipedal-locomotion-balance.md
│   └── ch15-reinforcement-learning-sim2real.md
├── module-4-vision-language-action/
│   ├── ch16-llms-meet-robotics.md
│   ├── ch17-voice-to-action-whisper.md
│   ├── ch18-cognitive-task-planning-gpt4o.md
│   ├── ch19-multi-modal-perception.md
│   └── ch20-capstone-autonomous-humanoid.md
└── intro.md
```

### 2. Content Format
Each chapter follows this format:

```markdown
---
title: Chapter Title
module: 1
chapter: 1
sidebar_label: Ch1: Chapter Title
description: Brief description of the chapter
tags: [ros2, middleware, introduction]
difficulty: beginner
estimated_duration: 30
---

import MermaidDiagram from '@site/src/components/MermaidDiagram';

# Chapter Title

## Learning Outcomes
- Outcome 1
- Outcome 2
- Outcome 3

## Theory
Content explaining concepts...

<MermaidDiagram chart={`
graph TD;
    A[Robot] --> B[ROS 2];
    B --> C{Middleware};
`} />

## Step-by-Step Labs
1. Step 1
2. Step 2
3. Step 3

## Runnable Code Example
```python
# Python code example
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
```

## Mini-project
Challenge for students to apply concepts...

## Summary
Key points recap...

```

## Backend Services Setup

### 1. Initialize Database
```bash
cd backend
python -m scripts.init_db
```

### 2. Load Textbook Content into Vector Store
```bash
cd backend
python -m scripts.load_content_to_qdrant
```

This will process all textbook content and store vector embeddings in Qdrant for the RAG chatbot.

### 3. Run Backend Tests
```bash
cd backend
python -m pytest tests/
```

## Running the Full Application

### 1. Terminal 1 - Start Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 2. Terminal 2 - Start Frontend
```bash
npm start
```

### 3. Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Backend docs: http://localhost:8000/docs

## Key Features

### RAG Chatbot
Ask questions about the textbook content:
```javascript
// Example API call to chatbot
fetch('http://localhost:8000/api/chat/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer session_token'
  },
  body: JSON.stringify({
    query: 'Explain the difference between ROS 1 and ROS 2',
    context: { module: 1, chapter: 1 },
    user_id: 'user_uuid'
  })
})
```

### User Authentication
Register and login users using the auth API:
```javascript
// Register a new user
fetch('http://localhost:8000/api/auth/register', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    email: 'student@example.com',
    password: 'securePassword123',
    name: 'Student Name',
    preferred_language: 'en',
    learning_difficulty: 'intermediate'
  })
})
```

### Content Personalization
Get personalized content based on user preferences:
```javascript
// Get personalized content
fetch('http://localhost:8000/api/personalize/content', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer session_token'
  },
  body: JSON.stringify({
    user_id: 'user_uuid',
    content_id: 'content_uuid',
    requested_difficulty: 'intermediate'
  })
})
```

## Deployment to GitHub Pages

### 1. Configure GitHub Actions
The repository includes a GitHub Actions workflow file that will deploy the Docusaurus site to GitHub Pages on every push to the main branch.

### 2. Set GitHub Pages Variables
In your GitHub repository settings:
- Set the source to GitHub Actions
- The workflow will handle the rest

### 3. Build and Deploy Process
The workflow automatically:
1. Builds the Docusaurus application
2. Deploys it to GitHub Pages
3. Runs backend tests
4. (Optionally) deploys the backend to a cloud provider

## Troubleshooting

### Common Issues

1. **Backend not connecting to Qdrant**
   - Verify your QDRANT_URL and QDRANT_API_KEY are correct
   - Check that your Qdrant Cloud instance is running

2. **OpenAI API errors**
   - Confirm your OPENAI_API_KEY is valid
   - Check that you have sufficient credits

3. **ROS 2 examples not running**
   - Ensure ROS 2 Humble is properly installed and sourced
   - Verify your ROS 2 environment is activated

4. **Content not displaying properly**
   - Check that all frontmatter is correctly formatted
   - Verify that the sidebars.js file includes all new content