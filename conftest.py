"""Ensure the project root is importable so tests can `import config` / `he_benchmark`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
