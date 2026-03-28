import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
print(f"CONFTEST: Adding {_project_root} to sys.path")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
