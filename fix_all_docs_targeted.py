#!/usr/bin/env python3
"""
Fix encoding issues in all documentation files
"""
import sys
import glob
import re
import os

def read_file_with_fallback_encoding(filepath):
    """Try to read a file with various encodings"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"  Successfully read {os.path.basename(filepath)} with {encoding} encoding")
            return content, encoding
        except UnicodeDecodeError:
            continue
    
    raise Exception(f"Could not decode {filepath} with any common encoding")

def fix_file_encoding(filepath):
    print(f"Fixing encoding in {os.path.basename(filepath)}")
    
    # Read with fallback encodings
    original_content, used_encoding = read_file_with_fallback_encoding(filepath)
    
    # Fix common encoding issues without over-processing the content
    fixed_content = original_content
    
    # Replace various problematic dash characters with regular hyphen
    dash_replacements = {
        '–': '-',  # en dash to hyphen
        '—': '-',  # em dash to hyphen
        '−': '-',  # minus sign to hyphen
    }
    
    for bad_dash, good_dash in dash_replacements.items():
        fixed_content = fixed_content.replace(bad_dash, good_dash)
    
    # Remove any null bytes that might be present
    fixed_content = fixed_content.replace('\x00', '')
    
    # Write back the fixed content in UTF-8
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"  Fixed encoding issues in {os.path.basename(filepath)}")

def main():
    # Get all markdown files in the docs directory
    doc_files = glob.glob("docs/**/*.md", recursive=True)
    
    print(f"Found {len(doc_files)} documentation files to fix")
    
    for file_path in doc_files:
        try:
            fix_file_encoding(file_path)
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print("All documentation files have been fixed for encoding!")

if __name__ == "__main__":
    main()