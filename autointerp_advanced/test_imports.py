#!/usr/bin/env python3
"""Quick test to verify imports work correctly"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from main.run_labeling import run_labeling
    from main.extract_examples import extract_examples
    from main.generate_labels import generate_labels
    print("✅ All imports successful!")
    print(f"✅ run_labeling function: {run_labeling}")
    print(f"✅ extract_examples function: {extract_examples}")
    print(f"✅ generate_labels function: {generate_labels}")
    print("\n✅ Package is ready to use!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

