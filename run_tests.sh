#!/usr/bin/env bash
set -e

# ComfyUI-Inpaint-CropAndStitch Test Runner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Running ComfyUI-Inpaint-CropAndStitch Unit & Integration Tests "
echo "============================================================"

# Ensure python3 is available
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

# Run test discovery
if python3 -m unittest discover -s tests -p "test_*.py" -v "$@"; then
    echo "============================================================"
    echo " [SUCCESS] All tests passed successfully!"
    echo "============================================================"
    exit 0
else
    EXIT_CODE=$?
    echo "============================================================"
    echo " [FAILURE] Some tests failed. Exit code: $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
