"""Backward-compatible script; use the integrated default postprocessor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from room_size_pipeline.blender_scene import build

if __name__ == "__main__":
    build(Path(sys.argv[sys.argv.index("--")+1]).resolve())
