#!/bin/bash

# Script to push all changes to git repository
# This script stages, commits, and pushes changes including:
# - Updated .gitignore (allows CSV files and archive folder)
# - Updated README.md (comprehensive folder documentation)
# - New/renamed scripts in autointerp_full/
# - prompts_finance.yaml
# - Archive folder contents

cd /home/nvidia/Documents/Hariom/autointerp

echo "=== Checking git status ==="
git status

echo ""
echo "=== Adding all changes ==="
git add .

echo ""
echo "=== Checking what will be committed ==="
git status

echo ""
read -p "Do you want to proceed with commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "=== Committing changes ==="
git commit -m "Update repository structure and documentation

- Updated .gitignore to allow CSV files and archive folder
- Added comprehensive README documentation for all folders
- Renamed scripts to shorter names (max 3 words)
- Added prompts_finance.yaml for finance-specific prompts
- Documented when to use each folder and their features
- Added archive folder documentation
- Included CSV files in repository"

echo ""
read -p "Do you want to push to remote? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Commit completed. Push manually when ready."
    exit 0
fi

echo ""
echo "=== Pushing to remote ==="
# Try to detect default branch
DEFAULT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
echo "Pushing to branch: $DEFAULT_BRANCH"
git push origin "$DEFAULT_BRANCH"

echo ""
echo "=== Done ==="


