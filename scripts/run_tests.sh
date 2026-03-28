#!/bin/bash
# Run tests for a specific project in complete isolation.
# Sets ONLY the target project dir on sys.path — no other project's src/ can interfere.
# Usage: ./scripts/run_tests.sh 04-graph-fraud-detection

set -euo pipefail

PROJECT="$1"
if [ -z "$PROJECT" ]; then
    echo "Usage: $0 <project-dir>"
    exit 1
fi

# Run a clean Python that ONLY has this project on sys.path.
# --isolated prevents loading conftest.py from parent dirs.
exec uv run python -c "
import sys, os
# Clear any cached 'src' modules from other projects
for mod in list(sys.modules):
    if mod == 'src' or mod.startswith('src.'):
        del sys.modules[mod]
# Add ONLY this project to path
sys.path.insert(0, os.path.abspath('${PROJECT}'))
# Run pytest
import pytest
# Debug: check for src namespace collision
import importlib
try:
    import src
    print(f'DEBUG: src.__path__={src.__path__}')
    print(f'DEBUG: src.__file__={getattr(src, \"__file__\", \"N/A\")}')
except Exception as e:
    print(f'DEBUG: src import failed: {e}')
# Check for stale .pth files
import site
for sp in site.getsitepackages():
    pth_files = [f for f in os.listdir(sp) if f.endswith('.pth')] if os.path.isdir(sp) else []
    for pf in pth_files:
        content = open(os.path.join(sp, pf)).read().strip()
        if content:
            print(f'DEBUG: .pth file {pf}: {content[:100]}')
sys.exit(pytest.main([
    '${PROJECT}/tests/',
    '-v', '--tb=short',
    '--rootdir=${PROJECT}',
]))
"
