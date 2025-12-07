"""Launcher for the Drone Detection GUI."""

import os
import sys

# Add src directory to path for cross-platform compatibility
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from drone_detector_gui import main


if __name__ == "__main__":
    main()
