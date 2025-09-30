#!/usr/bin/env python3
"""
AutoInterp Lite - Main Runner Script
Flexible command-line interface for feature activation analysis
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.run_lite_analysis import main

if __name__ == "__main__":
    main()
