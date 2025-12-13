#!/usr/bin/env python3
"""
Fix encoding issues in documentation files
"""

def fix_file_encoding(filepath):
    print(f"Fixing encoding in {filepath}")
    
    # Read as raw bytes first to identify problematic sequences
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    
    # Decode with error handling to replace problematic characters
    text_content = raw_bytes.decode('utf-8', errors='replace')
    
    # Fix common problematic characters
    # Replace replacement character () with proper en dash if context suggests it
    # Note: This is a general approach, but for this specific case we know it's the en dash
    fixed_content = text_content.replace('', '–')  # Replace replacement char with en dash
    
    # Write back the fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed encoding issues in {filepath}")

def main():
    import glob
    
    # Get all markdown files in the docs directory
    doc_files = glob.glob("docs/**/*.md", recursive=True)
    
    print(f"Found {len(doc_files)} documentation files to fix")
    
    for file_path in doc_files:
        fix_file_encoding(file_path)
    
    print("All documentation files have been fixed for encoding!")

if __name__ == "__main__":
    main()