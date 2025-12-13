#!/bin/bash
# Script to identify YAML frontmatter issues in documentation files

echo "Validating YAML frontmatter in documentation files..."

# Find all .md files in docs directory and subdirectories
find docs -name "*.md" -type f | while read -r file; do
    echo "Checking: $file"
    
    # Check if the file starts with exactly one '---' line
    first_line=$(head -n 1 "$file" | tr -d '\r')
    
    if [ "$first_line" != "---" ]; then
        echo "  ERROR: File doesn't start with '---': $file"
    else
        # Count the number of '---' lines at the beginning
        line_count=$(head -n 10 "$file" | grep -c '^---$' || echo 0)
        
        if [ "$line_count" -gt 1 ]; then
            echo "  ERROR: Multiple '---' lines at start of file: $file"
        else
            # Verify if the frontmatter closes properly
            frontmatter_end=$(grep -n '^---$' "$file" | sed -n 2p | cut -d: -f1)
            if [ -z "$frontmatter_end" ]; then
                echo "  ERROR: Frontmatter not properly closed in: $file"
            else
                echo "  OK: Frontmatter appears valid for $file"
            fi
        fi
    fi
    echo ""
done

echo "Validation complete."