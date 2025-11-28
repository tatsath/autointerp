#!/usr/bin/env python3
"""
Unified entry point for feature search.
Wrapper around main/run_feature_search.py with simplified interface.
"""

import os
import sys

# Add main directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.join(script_dir, "main")
sys.path.insert(0, main_dir)

# Import and run
from run_feature_search import run_feature_search
import fire

if __name__ == "__main__":
    fire.Fire(run_feature_search)

