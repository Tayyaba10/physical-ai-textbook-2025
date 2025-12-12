---
id: 1
title: Create Project Constitution
stage: constitution
date: 2025-12-12
surface: agent
model: gpt-4o
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution, textbook, ai, robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - D:\textbook\.specify\memory\constitution.md
 - D:\textbook\.specify\templates\plan-template.md
 - D:\textbook\.specify\templates\spec-template.md
 - D:\textbook\.specify\templates\tasks-template.md
tests:
 - null
---

## Prompt

Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync

## Response snapshot

# AI-Native Textbook for "Physical AI & Humanoid Robotics" Constitution

## Core Principles

### Technical Accuracy
All robotics, AI, and hardware claims must rely on reputable sources (academic or vendor docs: ROS, NVIDIA, Unity, Gazebo). Every statement about technology, APIs, or capabilities must be verified against official documentation or academic papers. This ensures students receive accurate information that reflects real-world implementations.

### Clarity and Accessibility
Content must be professional but beginner-friendly (Flesch-Kincaid Grade 8–12) with clear explanations for students with basic AI/programming background. Diagrams should be clean, labeled, and implementation-focused. Writing should prioritize understanding over complexity.

### Engineering-Focused Explanations
All content must emphasize practical workflows and real-world engineering practices that match modern robotics engineering standards. Theoretical concepts should always be grounded with practical applications and implementation guidance. Content must support hands-on learning experiences.

### Reproducibility
All code, diagrams, and steps must be reproducible and runnable. Every example, tutorial, and lab exercise must be tested to ensure it works as described. Students should be able to follow along and achieve identical results with the provided instructions and code.

### Modular Structure
Content must follow a modular design supporting AI personalization and RAG-based learning. Each module should be self-contained yet connected to the broader curriculum. This enables flexible learning paths and personalized education experiences.

### Industry Alignment
All practices, tools, and methodologies must align with current industry standards in robotics and AI. Content should reflect real-world workflows used in professional environments. Technologies covered should represent current best practices in the field.

## Additional Requirements

### Book Structure
The textbook must include at least 12 chapters mapping directly to weekly curriculum requirements. Each chapter must include: learning outcomes, practical steps, examples, mini-projects, and assessment checkpoints. Content should flow logically from foundational concepts to advanced applications.

### Code Standards
All code must be runnable, tested, annotated, and cross-validated. Code examples must support Ubuntu 22.04, ROS 2 Humble/Iron, Python 3.10+. All robotics, AI, and hardware claims must rely on reputable sources. Code should follow consistent formatting and include appropriate comments for educational purposes.

### Hardware Guidance
Content must include comprehensive hardware guidance covering: high-performance RTX "Digital Twin" workstation, Jetson Edge AI "Physical AI" kit, and optional robotics platforms (Unitree, Hiwonder, Go2). Hardware recommendations should be current, practical, and clearly explained.

### Simulation Workflows
Simulation workflows must be reproducible on both local RTX workstation and cloud workstations (AWS g5/g6e instances). All simulation examples should work consistently across these platforms with minimal configuration changes.

### Content Formatting
Use clean hierarchical structure: Modules → Chapters → Sections → Examples → Labs → Checkpoints. Include Mermaid diagrams for systems and architecture, tables for hardware comparison, and clear step-by-step lab instructions. Include at least one mini-project per module.

## Development Workflow

### Content Validation
All content must undergo technical validation to ensure accuracy against ROS 2, Gazebo, Isaac Sim, and VLA standards. Robotics accuracy must be verified, code execution correctness confirmed, and all architectures validated against official documentation. All undefined terms must be clarified and chapter coherence maintained.

### Prohibitions
Do not include hallucinated robotics APIs or false hardware specifications. Avoid overly theoretical content without application, unsupported claims about performance, SLAM accuracy, or simulation fidelity. Do not include outdated robotics content (ROS1, Python2, improper Gazebo versions).

### Deliverables
The project must deliver: a fully functional book built using Docusaurus and deployed to GitHub Pages or Vercel, an integrated RAG chatbot using OpenAI ChatCompletions/Agents, ChatKit SDK, FastAPI backend, Qdrant Cloud (Free Tier), and Neon Serverless PostgreSQL. The chatbot must answer full-book questions and section-specific questions based on user-highlighted text.

## Governance

This constitution governs all development, content creation, and implementation decisions for the AI-Native Textbook project. All contributions must comply with these principles. Amendments require documentation of changes and approval from the project leadership team. All pull requests and reviews must verify compliance with these principles before merging.

**Version**: 1.0.0 | **Ratified**: 2025-01-01 | **Last Amended**: 2025-12-12

## Outcome

- ✅ Impact: Created new constitution for AI-Native Textbook project
- 🧪 Tests: Updated templates validated for consistency
- 📁 Files: Updated constitution and related templates
- 🔁 Next prompts: Continue with specification and planning
- 🧠 Reflection: Constitution successfully aligned with project requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Not applicable