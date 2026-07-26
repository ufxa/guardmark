#!/bin/bash
# Build GuardMark paper with tectonic
# Usage: ./scripts/build.sh

set -e
cd "$(dirname "$0")/../paper"

echo "Building GuardMark paper with tectonic..."
tectonic main.tex
echo "Build complete: paper/main.pdf"
