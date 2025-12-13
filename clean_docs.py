#!/usr/bin/env python3
"""
Script to clean all documentation files of null bytes and other potential issues
"""

import os
import glob

def clean_file(filepath):
    """Remove null bytes and other potential issues from a file"""
    print(f"Cleaning {filepath}...")
    
    # Read the file in binary mode to properly handle all characters
    with open(filepath, 'rb') as f:
        content = f.read()
    
    original_size = len(content)
    
    # Remove null bytes which cause YAML parsing issues
    cleaned_content = content.replace(b'\x00', b'')
    
    # Remove any other problematic characters if needed
    # (Currently focusing on null bytes which are known to cause the issue)
    
    new_size = len(cleaned_content)
    
    if original_size != new_size:
        # Write the cleaned content back
        with open(filepath, 'wb') as f:
            f.write(cleaned_content)
        print(f"  Fixed {original_size - new_size} null bytes in {filepath}")
    else:
        print(f"  No null bytes found in {filepath}")

def main():
    # Get all markdown files in the docs directory
    doc_files = glob.glob("docs/**/*.md", recursive=True)
    
    print(f"Found {len(doc_files)} documentation files to clean")
    
    for file_path in doc_files:
        clean_file(file_path)
    
    print("All documentation files have been cleaned!")

if __name__ == "__main__":
    main()