#!/usr/bin/env python3
"""
Fix encoding issues in a specific file
"""
import sys
import re

def read_file_with_fallback_encoding(filepath):
    """Try to read a file with various encodings"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"Successfully read with {encoding} encoding")
            return content, encoding
        except UnicodeDecodeError:
            continue
    
    raise Exception(f"Could not decode {filepath} with any common encoding")

def fix_file_encoding(filepath):
    print(f"Fixing encoding in {filepath}")
    
    # Read with fallback encodings
    original_content, used_encoding = read_file_with_fallback_encoding(filepath)
    
    # Fix common encoding issues
    fixed_content = original_content
    
    # Replace various problematic dash characters with regular hyphen or proper en-dash
    # Different types of dash characters that might be causing issues
    dash_replacements = {
        '–': '-',  # en dash to hyphen
        '—': '-',  # em dash to hyphen
        '−': '-',  # minus sign to hyphen
    }
    
    for bad_dash, good_dash in dash_replacements.items():
        fixed_content = fixed_content.replace(bad_dash, good_dash)
    
    # Also fix the specific problem with special characters in headings
    # that might be interpreted as JSX
    fixed_content = re.sub(r'[^\x00-\x7F]+', lambda m: m.group(0).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'), fixed_content)
    
    # Remove any remaining null bytes that might have been missed
    fixed_content = fixed_content.replace('\x00', '')
    
    # Write back the fixed content in UTF-8
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed encoding issues in {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_encoding_targeted.py <file_path>")
    else:
        fix_file_encoding(sys.argv[1])