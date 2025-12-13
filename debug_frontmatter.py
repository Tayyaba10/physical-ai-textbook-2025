#!/usr/bin/env python3
"""
Debug script to check for frontmatter issues in documentation files
"""

def debug_frontmatter(filepath):
    print(f"Debugging file: {filepath}")
    
    with open(filepath, 'rb') as f:
        raw_content = f.read()
    
    # Check for BOM
    if raw_content.startswith(b'\xef\xbb\xbf'):
        print("  - BOM found at beginning")
        raw_content = raw_content[3:]
    else:
        print("  - No BOM found")
    
    # Decode content
    try:
        content = raw_content.decode('utf-8')
    except UnicodeDecodeError as e:
        print(f"  - Unicode decode error: {e}")
        return
    
    # Show first few characters with their codes
    print("  - First 10 characters (with ASCII codes):")
    for i, char in enumerate(content[:10]):
        print(f"    {i}: '{char}' ({ord(char)})")
    
    # Split content into lines
    lines = content.splitlines(keepends=False)  # Don't keep newlines
    
    print(f"  - First 5 lines: {lines[:5]}")
    
    # Check for --- pattern
    if lines and lines[0] == '---':
        print("  - First line is correctly '---'")
    else:
        print(f"  - First line is '{lines[0] if lines else 'EMPTY'}'")
    
    # Count occurrences of '---'
    count = sum(1 for line in lines if line.strip() == '---')
    print(f"  - Total count of lines with just '---': {count}")
    
    # Find positions of '---'
    positions = [i for i, line in enumerate(lines) if line.strip() == '---']
    print(f"  - Positions of '---': {positions}")
    
    if len(positions) >= 2:
        print(f"  - Frontmatter ends at line {positions[1]}")
    
    # Check if there are consecutive '---' lines at the start
    start_count = 0
    for i, line in enumerate(lines):
        if line.strip() == '---':
            start_count += 1
        else:
            break
            
    if start_count > 1:
        print(f"  - ERROR: Found {start_count} consecutive '---' lines at start!")
    else:
        print(f"  - OK: Only {start_count} '---' line at start")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python debug_frontmatter.py <file_path>")
    else:
        debug_frontmatter(sys.argv[1])