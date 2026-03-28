#!/bin/bash
# Run tests for a specific project.
# Usage: ./scripts/run_tests.sh 04-graph-fraud-detection
set -euo pipefail
PROJECT="$1"
exec uv run python -m pytest "$PROJECT/tests/" -v --tb=short --rootdir="$PROJECT"
