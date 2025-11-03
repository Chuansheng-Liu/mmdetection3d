#!/usr/bin/env python3
"""Check for hardcoded absolute paths in the codebase.

This script checks for common patterns of hardcoded absolute paths that
should not be committed to the repository.
"""
import argparse
import re
import sys
from pathlib import Path


def check_file_for_hardcoded_paths(filepath):
    """Check a single file for hardcoded absolute paths.
    
    Args:
        filepath: Path to the file to check.
        
    Returns:
        List of tuples (line_number, line_content) with issues found.
    """
    # Patterns to detect hardcoded absolute paths
    # Exclude common false positives like URLs, comments about paths, etc.
    patterns = [
        r'["\']/(home|usr/local|opt)/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+',  # /home/user/...
        r'["\'][A-Z]:\\Users\\[a-zA-Z0-9_-]+',  # Windows paths C:\Users\...
    ]
    
    # Files to skip
    skip_extensions = {'.pkl', '.pth', '.ckpt', '.bin', '.so', '.pyc'}
    
    if Path(filepath).suffix in skip_extensions:
        return []
    
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Skip lines that are clearly comments about paths
                if line.strip().startswith('#'):
                    continue
                if '"""' in line or "'''" in line:
                    continue
                    
                for pattern in patterns:
                    if re.search(pattern, line):
                        # Additional validation to reduce false positives
                        # Skip if it's in a URL
                        if 'http://' in line or 'https://' in line:
                            continue
                        # Skip if it's in a comment
                        if '//' in line and line.index('//') < line.index(pattern):
                            continue
                            
                        issues.append((line_num, line.strip()))
                        break
    except (UnicodeDecodeError, FileNotFoundError, PermissionError):
        pass
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Check for hardcoded absolute paths in files')
    parser.add_argument('files', nargs='*', help='Files to check')
    parser.add_argument('--all', action='store_true',
                        help='Check all Python and config files')
    args = parser.parse_args()
    
    files_to_check = []
    if args.all:
        repo_root = Path(__file__).parent.parent
        patterns = ['**/*.py', '**/*.yaml', '**/*.yml', '**/*.json']
        for pattern in patterns:
            files_to_check.extend(repo_root.glob(pattern))
        # Exclude some directories
        files_to_check = [
            f for f in files_to_check 
            if '.git' not in str(f) and 'tests/data' not in str(f)
        ]
    else:
        files_to_check = args.files
    
    all_issues = []
    for filepath in files_to_check:
        issues = check_file_for_hardcoded_paths(filepath)
        if issues:
            all_issues.append((filepath, issues))
    
    if all_issues:
        print('ERROR: Found hardcoded absolute paths:')
        for filepath, issues in all_issues:
            print(f'\n{filepath}:')
            for line_num, line in issues:
                print(f'  Line {line_num}: {line}')
        print('\nPlease use relative paths or configuration variables instead.')
        return 1
    else:
        print('No hardcoded absolute paths found.')
        return 0


if __name__ == '__main__':
    sys.exit(main())
