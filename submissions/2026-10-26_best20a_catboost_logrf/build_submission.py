"""Portable entry point; original workspace builder is archived separately."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("reproduce.py")), run_name="__main__")
