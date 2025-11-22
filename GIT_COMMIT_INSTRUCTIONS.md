# Git Commit Instructions

## Changes Made

1. **Updated .gitignore:**
   - Removed `archive/` from ignore list (now tracked)
   - Added exception for CSV files (`!*.csv`)
   - Kept log files ignored

2. **Updated README.md:**
   - Added comprehensive documentation for all folders
   - Documented when to use each folder
   - Added repository structure section
   - Documented archive folders and their purposes

3. **New/Updated Scripts in autointerp_full/:**
   - `run_finbert.sh` (renamed from longer name)
   - `run_test_cache.sh` (renamed)
   - `run_nemotron.sh` (renamed)
   - `run_nemotron_system.sh` (renamed)
   - `run_llama_all.sh` (new)
   - `run_llama_features.sh` (renamed)
   - `prompts_finance.yaml` (new - finance-specific prompts)

## Git Commands to Run

```bash
cd /home/nvidia/Documents/Hariom/autointerp

# Check status
git status

# Add all changes (including archive folder and CSV files)
git add .

# Remove files that shouldn't be tracked (if any)
# Check what's staged
git status

# Commit changes
git commit -m "Update repository structure and documentation

- Updated .gitignore to allow CSV files and archive folder
- Added comprehensive README documentation for all folders
- Renamed scripts to shorter names (max 3 words)
- Added prompts_finance.yaml for finance-specific prompts
- Documented when to use each folder and their features
- Added archive folder documentation"

# Push to remote
git push origin main
# or
git push origin master
# (depending on your default branch name)
```

## Files to Verify

Before committing, verify:
- ✅ CSV files are included (not ignored)
- ✅ Archive folder is included
- ✅ New scripts are included
- ✅ prompts_finance.yaml is included
- ✅ README.md updates are included
- ✅ .gitignore changes are included

## Files That Should NOT Be Committed

- Large log files (*.log)
- Results folders (already in .gitignore)
- Cache folders (.embedding_cache/)
- __pycache__/ folders
- *.pyc files

