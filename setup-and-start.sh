#!/bin/bash
# setup-and-start.sh - Script to set up and start the documentation site

echo "AI-Native Robotics Textbook Development Setup"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js before proceeding."
    echo "Visit https://nodejs.org/ to download and install."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2)
NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)

if [ "$NODE_MAJOR" -lt 18 ]; then
    echo "Node.js version 18 or higher is required. Current version: $NODE_VERSION"
    exit 1
fi

echo "Node.js version: $NODE_VERSION ✓"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install npm along with Node.js."
    exit 1
fi

echo "npm is installed ✓"

# Install Docusaurus globally if not already installed
if ! command -v docusaurus &> /dev/null; then
    echo "Installing Docusaurus CLI..."
    npm install -g @docusaurus/cli
else
    echo "Docusaurus CLI is already installed ✓"
fi

# Install local dependencies
echo "Installing project dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully ✓"
else
    echo "Error installing dependencies"
    exit 1
fi

# Build the site
echo "Building the documentation site..."
npm run build

if [ $? -eq 0 ]; then
    echo "Site built successfully ✓"
else
    echo "Error building site"
    exit 1
fi

echo ""
echo "Setup complete! To start the development server:"
echo "1. Run: npm start"
echo "2. Open your browser to http://localhost:3000"
echo ""
echo "The AI-Native Robotics Textbook documentation is ready for development!"