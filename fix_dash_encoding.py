#!/usr/bin/env python3
"""
Script to fix files that have characters separated by dashes
e.g. 'title:' becomes '-t-i-t-l-e-:'
"""

import os
import glob

def fix_dash_separated_content(filepath):
    """Fix content where characters are separated by dashes"""
    print(f"Processing {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Method 1: If content is mostly dashes between characters, 
    # join every sequence of character-dash
    original_content = content
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # If more than 60% of characters in the line are dashes,
        # assume it's dash-separated content
        if len(line) > 0 and line.count('-') / len(line) > 0.4:
            # Remove leading dash if present
            if line.startswith('-'):
                line = line[1:]
            
            # Split by dash and join the single characters
            parts = line.split('-')
            # Only consider it dash-separated if most parts are single characters
            single_char_parts = [p for p in parts if len(p) == 1]
            if len(single_char_parts) > len(parts) / 2:
                # Join the single characters
                fixed_line = ''.join(single_char_parts)
            else:
                # If not mostly single chars, keep original
                fixed_line = line
        else:
            # Line doesn't appear dash-separated, keep as is
            fixed_line = line
        
        fixed_lines.append(fixed_line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    # Write back the fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed {filepath} - Original length: {len(original_content)}, New length: {len(fixed_content)}")

def main():
    # Get all markdown files in the docs directory
    doc_files = glob.glob("docs/**/*.md", recursive=True)
    
    print(f"Found {len(doc_files)} documentation files to fix")
    
    for file_path in doc_files:
        try:
            fix_dash_separated_content(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("All documentation files have been processed!")

if __name__ == "__main__":
    main()