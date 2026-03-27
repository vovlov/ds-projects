#!/bin/bash
# Run tests for a specific project with correct PYTHONPATH
# Usage: ./scripts/run_tests.sh 04-graph-fraud-detection

set -euo pipefail

PROJECT="$1"
if [ -z "$PROJECT" ]; then
    echo "Usage: $0 <project-dir>"
    exit 1
fi

# Use --env-file trick: uv run passes env vars from parent process
# but some CI environments strip them. Force it via uv run --env.
exec uv run --env-file /dev/null -- env "PYTHONPATH=${PROJECT}:${PYTHONPATH:-}" python -m pytest "$PROJECT/tests/" -v --tb=short
