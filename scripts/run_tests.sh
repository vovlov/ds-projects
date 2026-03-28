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
print(f'sys.path[0] = {sys.path[0]}')
print(f'Project dir exists: {os.path.isdir(os.path.abspath(\"${PROJECT}\"))}')
print(f'src dir exists: {os.path.isdir(os.path.join(os.path.abspath(\"${PROJECT}\"), \"src\"))}')
sys.exit(pytest.main([
    '${PROJECT}/tests/',
    '-v', '--tb=short',
    '--rootdir=${PROJECT}',
]))
"
