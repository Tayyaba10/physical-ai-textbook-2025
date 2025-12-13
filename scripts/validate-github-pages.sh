#!/bin/bash
# GitHub Pages configuration validation script

echo "Validating GitHub Pages configuration..."

# Check if docusaurus.config.js exists and has required settings
if [ ! -f "docusaurus.config.js" ]; then
    echo "❌ ERROR: docusaurus.config.js not found"
    exit 1
else
    echo "✅ docusaurus.config.js exists"
fi

# Verify required GitHub Pages settings exist in config
if grep -q "organizationName" docusaurus.config.js && grep -q "projectName" docusaurus.config.js; then
    echo "✅ GitHub Pages organizationName and projectName found"
else
    echo "❌ ERROR: Missing organizationName or projectName in docusaurus.config.js"
    exit 1
fi

if grep -q "baseUrl" docusaurus.config.js; then
    echo "✅ baseUrl setting found"
else
    echo "❌ ERROR: Missing baseUrl in docusaurus.config.js"
    exit 1
fi

# Check if .nojekyll file exists in static directory
if [ -f "static/.nojekyll" ]; then
    echo "✅ .nojekyll file exists in static directory"
else
    echo "❌ WARNING: .nojekyll file not found in static directory"
fi

# Check if .nojekyll file exists in root
if [ -f ".nojekyll" ]; then
    echo "✅ .nojekyll file exists in root directory"
else
    echo "❌ WARNING: .nojekyll file not found in root directory"
fi

# Check if deploy workflow exists
if [ -f ".github/workflows/deploy.yml" ]; then
    echo "✅ GitHub Actions deploy workflow exists"
else
    echo "❌ ERROR: .github/workflows/deploy.yml not found"
    exit 1
fi

echo "✅ GitHub Pages configuration validation completed"
echo "⚠️  Note: Some documentation files may have formatting errors that prevent successful builds"