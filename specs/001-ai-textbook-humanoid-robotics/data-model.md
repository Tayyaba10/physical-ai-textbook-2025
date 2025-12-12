# Data Model: AI-Native Textbook: Physical AI & Humanoid Robotics

## Entity Relationships Overview
The system contains several key entities that represent the educational content, user interactions, and platform functionality for the AI-Native Textbook project.

## Core Entities

### 1. User
- **Description**: Represents a registered user of the textbook platform
- **Fields**:
  - `id` (UUID): Unique identifier for the user
  - `email` (string): User's email address for authentication
  - `name` (string): User's display name
  - `created_at` (timestamp): Account creation date
  - `updated_at` (timestamp): Last update to user profile
  - `preferred_language` (string): Default language preference (e.g., "en", "ur")
  - `learning_difficulty` (enum): Preferred content difficulty (beginner, intermediate, advanced)
  - `learning_path` (string): Current learning path or module
  - `progress_data` (json): Track progress across chapters and modules
  - `questionnaire_response` (json): Responses from background questionnaire
- **Relationships**:
  - One-to-many with `UserSession`
  - One-to-many with `ChatHistory`
  - One-to-many with `Bookmarks`

### 2. TextbookContent
- **Description**: Represents the educational material organized in 4 modules with 20 chapters
- **Fields**:
  - `id` (UUID): Unique identifier for the content
  - `title` (string): Title of the chapter/module
  - `content_type` (enum): Type of content (module, chapter, section)
  - `module_number` (integer): Module position (1-4)
  - `chapter_number` (integer): Chapter position within module (1-5)
  - `content` (text): Main content in Markdown format
  - `learning_outcomes` (text): Learning outcomes for the chapter
  - `prerequisites` (array): List of prerequisite concepts
  - `lab_exercises` (json): Step-by-step lab exercises
  - `code_examples` (json): Runnable code examples
  - `diagrams` (json): Reference to Mermaid and other diagrams
  - `mini_project` (text): Mini-project description
  - `summary` (text): Chapter summary
  - `created_at` (timestamp): Creation date
  - `updated_at` (timestamp): Last update date
  - `word_count` (integer): Number of words in the content
  - `estimated_reading_time` (integer): Estimated reading time in minutes
  - `difficulty_level` (enum): Intrinsic difficulty (beginner, intermediate, advanced)
  - `tags` (array): List of tags for content discovery
- **Relationships**:
  - One-to-many with `ContentVersion`
  - Many-to-many with `UserProgress` (via junction table)
  - One-to-many with `ChatContext` (for RAG)

### 3. UserProgress
- **Description**: Tracks user progress through the textbook content
- **Fields**:
  - `id` (UUID): Unique identifier for the progress record
  - `user_id` (UUID): Reference to the user
  - `content_id` (UUID): Reference to the textbook content
  - `status` (enum): Progress status (not_started, in_progress, completed)
  - `last_accessed_at` (timestamp): When the content was last accessed
  - `completion_percentage` (float): Percentage of content completed
  - `time_spent` (integer): Time spent on content in seconds
  - `notes` (text): User's personal notes on the content
  - `difficulty_feedback` (enum): How difficult user found the content
  - `rating` (integer): User rating (1-5 stars)
- **Relationships**:
  - Many-to-one with `User`
  - Many-to-one with `TextbookContent`

### 4. ChatHistory
- **Description**: Stores conversation history between users and the RAG chatbot
- **Fields**:
  - `id` (UUID): Unique identifier for the chat history entry
  - `user_id` (UUID): Reference to the user
  - `session_id` (UUID): Reference to the chat session
  - `query` (text): The user's question/query
  - `response` (text): The AI's response
  - `context_used` (json): Context from textbook used to generate response
  - `timestamp` (timestamp): When the interaction occurred
  - `feedback` (enum): Quality feedback (positive, negative, neutral)
  - `feedback_notes` (text): Additional feedback notes
- **Relationships**:
  - Many-to-one with `User`
  - One-to-many with `ChatContext` (references to textbook content used)

### 5. ChatContext
- **Description**: Represents the textbook content used as context for RAG responses
- **Fields**:
  - `id` (UUID): Unique identifier for the context record
  - `chat_history_id` (UUID): Reference to the chat history entry
  - `content_id` (UUID): Reference to the textbook content used
  - `relevance_score` (float): How relevant the content was to the query
  - `text_snippet` (text): The specific text snippet used from the content
  - `section_title` (string): Title of the section where snippet appears
- **Relationships**:
  - Many-to-one with `ChatHistory`
  - Many-to-one with `TextbookContent`

### 6. UserSession
- **Description**: Represents an authenticated session for a user
- **Fields**:
  - `id` (UUID): Unique identifier for the session
  - `user_id` (UUID): Reference to the user
  - `session_token` (string): The session token
  - `created_at` (timestamp): When the session was created
  - `expires_at` (timestamp): When the session expires
  - `ip_address` (string): IP address of the user
  - `user_agent` (string): User agent string of the browser/device
- **Relationships**:
  - Many-to-one with `User`

### 7. ContentVersion
- **Description**: Tracks different versions of textbook content
- **Fields**:
  - `id` (UUID): Unique identifier for the version
  - `content_id` (UUID): Reference to the textbook content
  - `version_number` (string): Version identifier (e.g., "1.0.0")
  - `content` (text): The actual content in Markdown format
  - `created_at` (timestamp): When this version was created
  - `created_by` (string): Who created this version
  - `change_summary` (text): Summary of changes made in this version
- **Relationships**:
  - Many-to-one with `TextbookContent`

## Validation Rules

### User Entity
- Email must be a valid email format
- Name must be between 2 and 100 characters
- Preferred language must be one of the supported languages
- Learning difficulty must be one of the defined enum values

### TextbookContent Entity
- Title must not be empty
- Content must be in valid Markdown format
- Module number must be between 1 and 4
- Chapter number must be between 1 and 5 (within each module)
- Word count must be positive
- Estimated reading time must be positive
- Difficulty level must be one of the defined enum values

### UserProgress Entity
- Status must be one of the defined enum values
- Completion percentage must be between 0 and 100
- Difficulty feedback must be one of the defined enum values
- Rating must be between 1 and 5

### ChatHistory Entity
- Query must not be empty
- Response must not be empty
- Feedback must be one of the defined enum values

## State Transitions

### UserProgress States
- `not_started` → `in_progress`: When user starts reading a chapter
- `in_progress` → `completed`: When user completes a chapter
- `completed` → `in_progress`: If user returns to review content

### ContentVersion States
- New content starts with version "1.0.0"
- Versions increment based on content changes
- Each new version is linked to the previous version