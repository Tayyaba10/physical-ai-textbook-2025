#!/usr/bin/env python3
"""
Fix YAML frontmatter in documentation files to properly quote values containing colons
"""

import glob
import re

def fix_frontmatter(filepath):
    print(f"Checking {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match the frontmatter section (between --- and ---)
    frontmatter_pattern = r'(---\n)(.*?)(\n---\n)'
    
    def replace_frontmatter(match):
        start, frontmatter_content, end = match.groups()
        
        # Find lines with sidebar_label that contain a colon
        lines = frontmatter_content.split('\n')
        fixed_lines = []
        for line in lines:
            # Check for sidebar_label containing a colon
            if line.startswith('sidebar_label:') and ':' in line[14:]:  # After 'sidebar_label:'
                # Extract the value part
                parts = line.split(':', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    # Add quotes around the value if not already quoted
                    if not (value.startswith('"') and value.endswith('"')) and not (value.startswith("'") and value.endswith("'")):
                        quoted_value = f'"{value}"'
                        fixed_line = f"sidebar_label: {quoted_value}"
                        print(f"  Fixed sidebar_label in {filepath}")
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        fixed_frontmatter = '\n'.join(fixed_lines)
        return start + fixed_frontmatter + end
    
    # Apply the fix to the content
    fixed_content = re.sub(frontmatter_pattern, replace_frontmatter, content, flags=re.DOTALL)
    
    # Write the fixed content back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"  Processed {filepath}")

def main():
    # Get all markdown files in the docs directory
    doc_files = glob.glob("docs/**/*.md", recursive=True)
    
    print(f"Found {len(doc_files)} documentation files to check")
    
    for file_path in doc_files:
        try:
            fix_frontmatter(file_path)
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print("All documentation files have been checked and fixed!")

if __name__ == "__main__":
    main()