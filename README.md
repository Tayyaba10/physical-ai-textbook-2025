# AI-Native Robotics Textbook: Physical AI & Humanoid Robotics

## 🤖 Complete Educational Resource for Modern Robotics

This repository contains the complete AI-Native Robotics textbook covering the full spectrum from ROS 2 foundations to advanced vision-language-action integration for humanoid robotics.

### 📚 **Table of Contents**
- **Module 1**: Robotic Nervous System (ROS 2) - Chapters 1-5
- **Module 2**: Digital Twin & Simulation - Chapters 6-10
- **Module 3**: AI-Robot Brain (NVIDIA Isaac) - Chapters 11-15
- **Module 4**: Vision-Language-Action Integration - Chapters 16-20

### 🎯 **Learning Outcomes**
- Master ROS 2 concepts from foundational to advanced levels
- Implement physics simulation with Isaac Sim and Gazebo
- Integrate AI models (GPT-4o, Whisper) with robotic systems
- Design multimodal perception and control systems
- Build end-to-end robotic applications with natural language interfaces
- Create humanoid robot control systems with balance and locomotion

### 🚀 **Quick Start**

#### Prerequisites
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- NVIDIA GPU with Isaac Sim 2024.x
- Node.js 18+ and npm
- OpenAI API key for GPT-4o/Whisper integration

#### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/physical-ai-textbook-2025.git
cd physical-ai-textbook-2025

# Install dependencies
npm install

# Build the documentation
npm run build

# Serve locally for development
npm start
```

### 🏗 **Project Structure**
```
├── docs/                     # Textbook content (20 chapters)
│   ├── module-1-robotic-nervous-system/    # ROS 2 fundamentals
│   ├── module-2-digital-twin/              # Simulation and digital twins  
│   ├── module-3-ai-robot-brain/            # Isaac Platform integration
│   └── module-4-vision-language-action/    # AI integration and capstone
├── src/                      # Docusaurus source components
├── backend/                  # FastAPI services for RAG chatbot
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
├── docusaurus.config.js      # Site configuration
├── sidebars.js               # Navigation structure
└── package.json              # Dependencies
```

### 🧠 **Technology Stack**
- **Framework**: Docusaurus v3 for documentation
- **Robotics**: ROS 2 Humble/Iron with Isaac Sim 2024.x
- **AI Integration**: OpenAI GPT-4o/Whisper APIs
- **Backend**: FastAPI + Neon PostgreSQL + Qdrant Cloud
- **Authentication**: BetterAuth
- **Deployment**: GitHub Pages with automated CI/CD

### 📖 **About the Content**

This textbook provides a comprehensive learning journey from basic ROS 2 concepts to advanced AI-integrated robotic systems. Each chapter includes:

- **Theoretical foundations** with mathematical explanations
- **Step-by-step labs** with runnable code examples
- **Integration concepts** connecting different modules
- **Practical implementation** guides for real systems
- **Safety considerations** for responsible AI development

### 🏁 **Capstone Project**
The culmination of the textbook is a complete humanoid robot system that can:
- Understand voice commands through Whisper integration
- Plan complex tasks using GPT-4o cognitive reasoning
- Navigate to specified locations using Isaac-based navigation
- Manipulate objects using vision-language-action pipelines
- Adapt to environmental conditions with multimodal perception

### 🤝 **Contributing**
We welcome contributions to improve the textbook:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request with clear description

### 📄 **License**
This textbook is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

### 💬 **Support**
For questions about the content, please open an issue in the GitHub repository.

---

**Ready to start your journey into AI-Native Robotics? Begin with [Module 1, Chapter 1](./docs/module-1-robotic-nervous-system/ch1-why-ros2.md)!**