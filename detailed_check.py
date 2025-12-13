#!/usr/bin/env python3
"""
Check for various encoding/character issues in a file
"""

def detailed_check(filepath):
    print(f"Detailed check of {filepath}")
    
    # Read as raw bytes first
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    
    print(f"Total bytes: {len(raw_bytes)}")
    
    # Check for various problematic sequences
    problems = []
    
    # Look for null bytes
    if b'\x00' in raw_bytes:
        problems.append("Null bytes (\\x00)")
        null_positions = []
        start = 0
        while True:
            pos = raw_bytes.find(b'\x00', start)
            if pos == -1:
                break
            null_positions.append(pos)
            start = pos + 1
        print(f"  Null byte positions: {null_positions}")
    
    # Look for other potential problem chars
    if b'\x01' in raw_bytes:  # Start of heading
        problems.append("Start of heading (\\x01)")
    if b'\x02' in raw_bytes:  # Start of text
        problems.append("Start of text (\\x02)")
    if b'\x1a' in raw_bytes:  # DOS EOF / Windows Ctrl+Z
        problems.append("DOS EOF (\\x1a)")
    
    # Check for BOM
    if raw_bytes.startswith(b'\xef\xbb\xbf'):
        problems.append("UTF-8 BOM present")
    
    # Check for the sequence that might be causing the YAML issue
    # The issue might be double dashes or some other character
    print(f"Problems found: {problems if problems else 'None'}")
    
    # Let's also check the text content
    try:
        text_content = raw_bytes.decode('utf-8')
        print("Successfully decoded as UTF-8")
        
        # Check for problematic characters in text
        problematic_chars = []
        for i, char in enumerate(text_content):
            if ord(char) < 32 and ord(char) not in (9, 10, 13):  # tab, newline, carriage return are OK
                problematic_chars.append((i, ord(char), repr(char)))
        
        if problematic_chars:
            print(f"Control characters found: {problematic_chars[:10]}")  # show first 10
        else:
            print("No problematic control characters found")
            
    except UnicodeDecodeError as e:
        print(f"UTF-8 decode error: {e}")
        return
    
    # Check the structure around the frontmatter
    lines = text_content.split('\n')
    print(f"Total lines: {len(lines)}")
    
    # Show first 15 lines
    print("First 15 lines:")
    for i, line in enumerate(lines[:15]):
        print(f"{i:2d}: {repr(line)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detailed_check.py <file_path>")
    else:
        detailed_check(sys.argv[1])