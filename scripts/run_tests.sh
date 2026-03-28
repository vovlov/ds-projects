#!/bin/bash
set -euo pipefail
exec uv run python -c "
import sys, os

# 1. Remove ALL other project src/ dirs from path
project = os.path.abspath('$1')
sys.path = [p for p in sys.path if '/src' not in p or project in p]
sys.path.insert(0, project)

# 2. Clear any cached src module
for mod in list(sys.modules):
    if mod == 'src' or mod.startswith('src.'):
        del sys.modules[mod]

# 3. Verify src is importable from the right place
os.chdir(project)
import src
assert project in str(src.__path__), f'src resolves to {src.__path__}, expected {project}'

# 4. Run tests
import pytest
raise SystemExit(pytest.main(['tests/', '-v', '--tb=short']))
"
