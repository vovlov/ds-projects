"""Root conftest.py — adds each project dir to sys.path for test discovery."""

import sys
from pathlib import Path

root = Path(__file__).parent
for project_dir in sorted(root.glob("[0-9]*-*")):
    if project_dir.is_dir() and (project_dir / "src").is_dir():
        project_str = str(project_dir)
        if project_str not in sys.path:
            sys.path.insert(0, project_str)
