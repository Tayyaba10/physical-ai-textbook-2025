# Implementation Plan: AI-Native Textbook: Physical AI & Humanoid Robotics

**Branch**: `001-ai-textbook-humanoid-robotics` | **Date**: 2025-12-12 | **Spec**: [spec link](spec.md)
**Input**: Feature specification from `/specs/001-ai-textbook-humanoid-robotics/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The AI-Native Textbook project delivers a comprehensive educational platform for Physical AI & Humanoid Robotics. It includes 20 chapters across 4 modules (ROS 2, Digital Twin, NVIDIA Isaac, Vision-Language-Action) with hands-on labs, runnable code, and a capstone project where students build and control a simulated humanoid robot. The platform is built with Docusaurus v3, deployed on GitHub Pages, and includes an integrated RAG chatbot using FastAPI backend with Neon PostgreSQL, Qdrant Cloud, and OpenAI ChatKit. It features content personalization, multilingual support (Urdu), and user authentication via BetterAuth.

## Technical Context

**Language/Version**: Markdown for Docusaurus documentation, Python 3.10+ for ROS 2/Humble/Iron code examples, JavaScript/TypeScript for frontend functionality
**Primary Dependencies**: Docusaurus v3+ with @docusaurus/preset-classic, ROS 2 Humble/Iron, Isaac Sim 2024.x, FastAPI, OpenAI SDK, BetterAuth, Qdrant Cloud, Neon PostgreSQL
**Storage**: GitHub Pages for static content, Neon Serverless PostgreSQL for user data and chatbot embeddings, Qdrant Cloud for vector storage of textbook content
**Testing**: pytest for backend services, automated GitHub Actions for deployment testing, manual validation of ROS 2 code examples on Ubuntu 22.04
**Target Platform**: Web-based (GitHub Pages), with code examples validated on Ubuntu 22.04 + ROS 2 Humble/Iron + Isaac Sim 2024.x
**Project Type**: Web application with static documentation and backend services
**Performance Goals**: Pages load in under 2 seconds, RAG chatbot responds to queries in under 3 seconds, 99% uptime during academic periods
**Constraints**: Total content ~35,000 words (~3–4k per chapter), compatibility with Ubuntu 22.04 + ROS 2 Humble/Iron + Isaac Sim 2024.x, deployment before Nov 30, 2025 6 PM
**Scale/Scope**: Target audience of undergraduate/graduate students, robotics practitioners, and educators; supports multilingual content and personalized learning paths

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Phase 0 Status:
- Technical Accuracy: All robotics, AI, and hardware claims must rely on reputable sources (ROS, NVIDIA, Unity, Gazebo docs) ✅
- Clarity and Accessibility: Content must be beginner-friendly (Flesch-Kincaid Grade 8–12) with clear explanations ✅
- Engineering-Focused: All content must emphasize practical workflows matching modern robotics standards ✅
- Reproducibility: All code, diagrams, and steps must be reproducible and runnable ✅
- Industry Alignment: Practices, tools, and methodologies must align with current industry standards ✅

### Post-Phase 1 Status:
- Technical Accuracy: Verified through research of official documentation and best practices for Docusaurus, ROS 2, Isaac Sim, FastAPI, and related technologies ✅
- Clarity and Accessibility: Confirmed through architectural decisions that prioritize user experience and learning paths ✅
- Engineering-Focused: Confirmed through practical architecture decisions that align with modern web and robotics development practices ✅
- Reproducibility: Confirmed through well-defined project structure and deployment processes ✅
- Industry Alignment: Verified through use of current industry standard technologies (Docusaurus, ROS 2 Humble, Isaac Sim 2024.x, FastAPI, etc.) ✅

## Project Structure

### Documentation (this feature)

```text
specs/001-ai-textbook-humanoid-robotics/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
.
├── docs/                           # Docusaurus documentation content
│   ├── module-1-robotic-nervous-system/    # Module 1 chapters
│   │   ├── ch1-why-ros2.md
│   │   ├── ch2-nodes-topics-actions.md
│   │   ├── ch3-bridging-ai-agents.md
│   │   ├── ch4-urdf-xacro-modeling.md
│   │   └── ch5-building-launching-packages.md
│   ├── module-2-digital-twin/              # Module 2 chapters
│   │   ├── ch6-physics-simulation.md
│   │   ├── ch7-gazebo-setup-building.md
│   │   ├── ch8-simulating-sensors.md
│   │   ├── ch9-unity-hdrp-visualization.md
│   │   └── ch10-debugging-simulations.md
│   ├── module-3-ai-robot-brain/            # Module 3 chapters
│   │   ├── ch11-nvidia-isaac-overview.md
│   │   ├── ch12-isaac-sim-photorealistic.md
│   │   ├── ch13-isaac-ros-accelerated-vslam.md
│   │   ├── ch14-bipedal-locomotion-balance.md
│   │   └── ch15-reinforcement-learning-sim2real.md
│   ├── module-4-vision-language-action/    # Module 4 chapters
│   │   ├── ch16-llms-meet-robotics.md
│   │   ├── ch17-voice-to-action-whisper.md
│   │   ├── ch18-cognitive-task-planning-gpt4o.md
│   │   ├── ch19-multi-modal-perception.md
│   │   └── ch20-capstone-autonomous-humanoid.md
│   └── intro.md
├── src/                            # Docusaurus source files
│   ├── components/                 # Custom React components
│   │   ├── Chatbot.js              # RAG chatbot component
│   │   ├── Personalization.js      # Content personalization
│   │   ├── UrduToggle.js           # Urdu language toggle
│   │   └── ...
│   ├── pages/                      # Additional pages if needed
│   └── theme/                      # Custom theme components
├── backend/                        # FastAPI backend for RAG and auth
│   ├── main.py                     # Main application
│   ├── models/                     # Data models
│   │   ├── user.py
│   │   └── chat.py
│   ├── routes/                     # API routes
│   │   ├── auth.py
│   │   ├── chat.py
│   │   └── rag.py
│   ├── services/                   # Business logic
│   │   ├── rag_service.py
│   │   ├── chat_service.py
│   │   └── auth_service.py
│   ├── config/                     # Configuration
│   │   └── settings.py
│   └── tests/                      # Backend tests
├── docusaurus.config.js            # Docusaurus configuration
├── sidebars.js                     # Sidebar configuration
├── package.json                    # Node dependencies
├── babel.config.js                 # Babel configuration
└── README.md                       # Project documentation
```

**Structure Decision**: Web application with static documentation frontend (Docusaurus) and backend services (FastAPI) for RAG chatbot and user authentication. This structure separates static educational content from dynamic services while maintaining good performance and SEO.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multi-component architecture | Need to separate static content (Docusaurus) from dynamic services (RAG chatbot, auth) | Single static site insufficient for AI chatbot functionality |
| Multiple external services | Need to leverage specialized services (Qdrant Cloud, Neon, OpenAI) for optimal performance | Self-hosting all services would increase complexity and maintenance |

## Summary of Phase 0 and Phase 1 Deliverables

### Phase 0: Research & Outline
- [x] `research.md` - Comprehensive research document covering technology stack decisions, chapter content validation, and implementation requirements
- Identified key technologies: Docusaurus v3, ROS 2 Humble, Isaac Sim 2024.x, FastAPI, Qdrant Cloud, Neon PostgreSQL, OpenAI
- Validated all architectural assumptions against official documentation
- Resolved all "NEEDS CLARIFICATION" points from technical context

### Phase 1: Design & Contracts
- [x] `data-model.md` - Complete data model outlining all key entities and their relationships
- [x] `contracts/` directory - API contracts for all backend services (RAG chatbot, authentication, personalization)
- [x] `quickstart.md` - Complete setup and usage guide for the entire platform
- [x] Agent context updated - Technology stack integrated into AI agent context for future development
- Validated constitution compliance post-design
- Created complete project structure with 20 chapters across 4 modules
- Designed backend architecture for RAG chatbot, authentication, and personalization services
