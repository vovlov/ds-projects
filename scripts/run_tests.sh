#!/bin/bash
# Run tests for a specific project with correct PYTHONPATH
# Usage: ./scripts/run_tests.sh 04-graph-fraud-detection

PROJECT="$1"
if [ -z "$PROJECT" ]; then
    echo "Usage: $0 <project-dir>"
    exit 1
fi

export PYTHONPATH="$PROJECT:$PYTHONPATH"
exec uv run python -m pytest "$PROJECT/tests/" -v --tb=short
