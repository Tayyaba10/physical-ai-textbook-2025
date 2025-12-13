#!/usr/bin/env python3
"""
Comprehensive cleaning of all documentation files
"""

import glob
import os

def clean_file(filepath):
    print(f"Cleaning {os.path.basename(filepath)}")
    
    # Read the file with error handling
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    original_size = len(content)
    
    # Remove null bytes and other problematic characters
    cleaned_content = content.replace('\x00', '')
    
    # Replace problematic dash characters
    cleaned_content = cleaned_content.replace('–', '-')  # en dash
    cleaned_content = cleaned_content.replace('—', '-')  # em dash
    cleaned_content = cleaned_content.replace('−', '-')  # minus sign
    
    new_size = len(cleaned_content)
    
    if original_size != new_size:
        print(f"  Removed {original_size - new_size} problematic characters from {os.path.basename(filepath)}")
    
    # Write back the cleaned content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"  Successfully cleaned {os.path.basename(filepath)}")

def main():
    # Get all markdown files in the docs directory
    doc_files = glob.glob("docs/**/*.md", recursive=True)
    
    print(f"Found {len(doc_files)} documentation files to clean")
    
    for file_path in doc_files:
        try:
            clean_file(file_path)
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print("All documentation files have been cleaned!")

if __name__ == "__main__":
    main()