#!/usr/bin/env python3
"""
Script to automatically fix common linting issues in the CardioAI Pro backend.
"""

import os
import re
import subprocess
from pathlib import Path


def fix_whitespace_issues(file_path):
    """Fix common whitespace issues in a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

        content = re.sub(r"^[ \t]+$", "", content, flags=re.MULTILINE)

        content = re.sub(r"\n\n\n+", "\n\n", content)

        content = content.rstrip() + "\n"

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed whitespace issues in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def fix_line_length_issues(file_path):
    """Fix some common line length issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for line in lines:
            if len(line.rstrip()) > 88:
                if '"""' in line or "'''" in line:
                    new_lines.append(line)
                    continue

                if line.strip().startswith("from ") and " import " in line:
                    if len(line.rstrip()) > 88:
                        parts = line.split(" import ")
                        if len(parts) == 2:
                            imports = parts[1].strip()
                            if "," in imports:
                                new_line = f"{parts[0]} import (\n"
                                import_items = [
                                    item.strip() for item in imports.split(",")
                                ]
                                for i, item in enumerate(import_items):
                                    if i == len(import_items) - 1:
                                        new_line += f"    {item}\n)\n"
                                    else:
                                        new_line += f"    {item},\n"
                                new_lines.append(new_line)
                                modified = True
                                continue

                if " + " in line and '"' in line:
                    new_lines.append(line)
                    continue

            new_lines.append(line)

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Fixed line length issues in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing line length in {file_path}: {e}")
        return False


def main():
    """Main function to fix linting issues."""
    backend_dir = Path("/home/ubuntu/repos/cardio.ai.pro/backend")
    app_dir = backend_dir / "app"

    python_files = []
    for root, dirs, files in os.walk(app_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    print(f"Found {len(python_files)} Python files to process")

    fixed_files = 0
    for file_path in python_files:
        print(f"Processing {file_path}")
        if fix_whitespace_issues(file_path):
            fixed_files += 1

    print(f"Fixed whitespace issues in {fixed_files} files")


if __name__ == "__main__":
    main()
