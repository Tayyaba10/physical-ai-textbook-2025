#!/usr/bin/env python3
"""
More robust script to fix dash-separated characters in a file
"""

def fix_content_dashes(content):
    """
    Fix content where characters are separated by dashes.
    This handles the specific pattern where each character was followed by a dash.
    """
    # Split content into lines for processing
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if the line appears to be dash-separated (more than 30% are dashes)
        if len(line) > 0 and line.count('-') / len(line) > 0.3:
            # Process this line to remove alternating character-dash pattern
            new_line = ""
            i = 0
            while i < len(line):
                # Add the current character if it's not a dash
                if line[i] != '-':
                    new_line += line[i]
                    # If the next character is a dash, skip it
                    if i + 1 < len(line) and line[i + 1] == '-':
                        i += 2  # Skip both the character and the dash
                    else:
                        i += 1
                else:
                    # Current character is a dash, add it (for actual dashes in content)
                    new_line += line[i]
                    i += 1
            fixed_lines.append(new_line)
        else:
            # Line doesn't appear dash-encoded, keep as is
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_file_dashes(filepath):
    """Fix dash-separated characters in a specific file"""
    print(f"Processing {filepath}")
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Fix the content
    fixed_content = fix_content_dashes(original_content)
    
    # Write the fixed content back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed {filepath} - Original length: {len(original_content)}, New length: {len(fixed_content)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_file_encoding.py <filepath>")
    else:
        fix_file_dashes(sys.argv[1])