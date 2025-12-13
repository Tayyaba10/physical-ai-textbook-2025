#!/usr/bin/env python3
"""
Script to fix frontmatter issues in Docusaurus documentation files
"""

def fix_frontmatter(filepath):
    with open(filepath, 'rb') as f:
        raw_content = f.read()
    
    # Check for BOM (Byte Order Mark) and remove if present
    if raw_content.startswith(b'\xef\xbb\xbf'):
        print(f"Removing BOM from {filepath}")
        raw_content = raw_content[3:]  # Remove BOM
    
    # Decode content
    content = raw_content.decode('utf-8')
    
    # Split content into lines
    lines = content.splitlines(keepends=True)
    
    # Check if the file starts correctly with ---
    if lines[0].strip() == '---' and lines[1].strip() != '---':
        # The format seems correct, just need to check if there are extra ---
        # Find the end of frontmatter (second occurrence of '---')
        frontmatter_end_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                frontmatter_end_idx = i
                break
        
        if frontmatter_end_idx != -1:
            # The frontmatter looks correct as is
            print(f"No issues found in {filepath}")
        else:
            print(f"Missing end frontmatter delimiter in {filepath}")
    elif lines[0].strip() == '' and lines[1].strip() == '---':
        # There's an extra blank line at the beginning
        print(f"Removing extra blank line from {filepath}")
        lines = lines[1:]
        new_content = ''.join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        # Check if there are multiple --- at the beginning
        # This might happen if there are extra --- lines
        print(f"Checking {filepath} for extra ---")
        
        # Look for extra --- at the beginning
        i = 0
        extra_lines_at_start = 0
        while i < len(lines) and lines[i].strip() == '---':
            extra_lines_at_start += 1
            i += 1
        
        if extra_lines_at_start > 1:
            print(f"Removing {extra_lines_at_start - 1} extra --- lines from {filepath}")
            # Keep only the first ---
            corrected_lines = ['---\n'] + lines[extra_lines_at_start-1:]
            new_content = ''.join(corrected_lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print(f"No specific issues found in {filepath}, checking content...")
            print(f"First line: '{lines[0]}'")
            print(f"Second line: '{lines[1]}'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python temp_fix.py <file_path>")
    else:
        fix_frontmatter(sys.argv[1])