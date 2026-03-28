#!/bin/bash
set -euo pipefail
exec uv run python -c "
import sys, os
sys.path.insert(0, os.path.abspath('$1'))
os.chdir('$1')
import pytest
raise SystemExit(pytest.main(['tests/', '-v', '--tb=short']))
"
