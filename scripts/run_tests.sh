#!/bin/bash
# Run tests for a specific project with correct sys.path
# Usage: ./scripts/run_tests.sh 04-graph-fraud-detection

set -euo pipefail

PROJECT="$1"
if [ -z "$PROJECT" ]; then
    echo "Usage: $0 <project-dir>"
    exit 1
fi

# Use Python to set sys.path and run pytest — guaranteed to work everywhere
exec uv run python -c "
import sys, os
sys.path.insert(0, os.path.abspath('${PROJECT}'))
import pytest
sys.exit(pytest.main(['${PROJECT}/tests/', '-v', '--tb=short']))
"
