#!/bin/bash
# Script to run clippy checks on the NumRS2 codebase
# with appropriate settings and configurations

set -e  # Exit on error

# Print banner
echo "================================================"
echo "NumRS2 Clippy Check Tool"
echo "================================================"
echo "Running clippy to find and fix code quality issues"
echo ""

# Default flags
CLIPPY_FLAGS=(
    -W clippy::all                   # Enable all clippy lints
    -W clippy::pedantic              # Enable pedantic lints
    -W clippy::nursery               # Enable experimental lints
    -W clippy::unwrap_used           # Warn on unwrap() usage
    -W clippy::expect_used           # Warn on expect() usage
    -A clippy::module_name_repetitions  # Allow module name repetitions
    -A clippy::similar_names           # Allow similar names
    -A clippy::cast_possible_truncation # Allow potential truncation in casts (common in numerical code)
    -A clippy::cast_lossless           # Allow lossless casts
    -A clippy::cast_precision_loss      # Allow precision loss casts (common in numerical code)
    -A clippy::too_many_arguments       # Allow functions with many arguments
    -A clippy::must_use_candidate       # Allow non-must-use functions
    -A clippy::missing_errors_doc       # Allow missing error documentation
)

# Function to run clippy
run_clippy() {
    local features=$1
    local fix_mode=$2

    echo "Running clippy with features: $features"
    
    if [ "$fix_mode" = "fix" ]; then
        cargo clippy $features --all-targets -- "${CLIPPY_FLAGS[@]}" --fix
    else
        cargo clippy $features --all-targets -- "${CLIPPY_FLAGS[@]}"
    fi
}

# Parse command line arguments
FIX_MODE=""
if [ "$1" = "--fix" ]; then
    FIX_MODE="fix"
    echo "Running in fix mode - clippy will automatically fix eligible issues"
    echo ""
fi

# Run clippy with different feature combinations
echo "Running clippy checks with default features..."
run_clippy "" "$FIX_MODE"

echo -e "\nRunning clippy checks with matrix_decomp feature..."
run_clippy "--features matrix_decomp" "$FIX_MODE"

echo -e "\nRunning clippy checks with scirs feature..."
run_clippy "--features scirs" "$FIX_MODE"

echo -e "\nRunning clippy checks with all features..."
run_clippy "--all-features" "$FIX_MODE"

# Final summary
echo -e "\n================================================"
echo "Clippy checks completed"
echo "================================================"

# Display instructions if running in check mode
if [ "$FIX_MODE" != "fix" ]; then
    echo "To automatically fix eligible issues, run: ./scripts/clippy_check.sh --fix"
fi