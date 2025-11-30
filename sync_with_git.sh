#!/bin/bash
# Script to sync local folder structure exactly with git
# Removes files/folders from git that don't exist locally
# Adds all local files/folders to git

set -e

cd /home/nvidia/Documents/Hariom/autointerp

echo "=== Syncing repository with local folder structure ==="
echo ""

# Step 1: Get list of all files currently tracked in git
echo "Step 1: Checking files tracked in git..."
GIT_FILES=$(git ls-files)

# Step 2: Remove files from git that don't exist locally
echo "Step 2: Removing files from git that don't exist locally..."
for file in $GIT_FILES; do
    if [ ! -e "$file" ]; then
        echo "  Removing from git: $file"
        git rm --cached "$file" 2>/dev/null || true
    fi
done

# Step 3: Remove directories from git that don't exist locally
echo "Step 3: Checking for directories in git that don't exist locally..."
GIT_DIRS=$(git ls-files | sed 's|/[^/]*$||' | sort -u)
for dir in $GIT_DIRS; do
    if [ -n "$dir" ] && [ ! -d "$dir" ]; then
        echo "  Directory in git but not local: $dir (will be removed when files are removed)"
    fi
done

# Step 4: Add all local files (respecting .gitignore)
echo "Step 4: Adding all local files..."
git add .

# Step 5: Show status
echo ""
echo "=== Git Status ==="
git status

echo ""
echo "=== Files to be committed ==="
git diff --cached --name-only | head -20
echo "..."

echo ""
read -p "Do you want to commit these changes? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted. Changes are staged but not committed."
    exit 0
fi

# Step 6: Commit
echo ""
echo "=== Committing changes ==="
git commit -m "Sync repository with local folder structure

- Removed files/folders from git that don't exist locally
- Added all local files/folders including autointerp_advanced
- Matched repository exactly with local structure"

echo ""
read -p "Do you want to push to remote? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Commit completed. Push manually when ready."
    exit 0
fi

# Step 7: Push
echo ""
echo "=== Pushing to remote ==="
DEFAULT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
echo "Pushing to branch: $DEFAULT_BRANCH"
git push origin "$DEFAULT_BRANCH"

echo ""
echo "=== Done ==="



