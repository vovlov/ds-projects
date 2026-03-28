import sys
from pathlib import Path

# Add project root to sys.path so 'from anomaly.X import Y' works
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
