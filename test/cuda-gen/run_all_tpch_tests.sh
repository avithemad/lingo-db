#!/bin/bash

# Script to run TPC-H tests with different parameter combinations
# Usage: ./run_all_tpch_tests.sh [scale_factor]

SCALE_FACTOR="$1"
if [ -z "$SCALE_FACTOR" ]; then
  SCALE_FACTOR=1
  echo "No scale factor provided, defaulting to 1"
fi

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="tpch_tests_sf${SCALE_FACTOR}_${TIMESTAMP}.log"
echo "Logging output to: $LOG_FILE"

# Function to log with timestamp
log_with_timestamp() {
  while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
  done
}

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE" | log_with_timestamp)
exec 2>&1

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TPCH_SCRIPT="$SCRIPT_DIR/run_tpch_tests.sh"

# Check if the run_tpch_tests.sh script exists
if [ ! -f "$RUN_TPCH_SCRIPT" ]; then
  echo "Error: run_tpch_tests.sh not found at $RUN_TPCH_SCRIPT"
  exit 1
fi

# Array to store test results
PASSED_CONFIGS=()
FAILED_CONFIGS=()

# Function to run tests and capture results
run_test_config() {
  local config_name="$1"
  shift  # Remove first argument, rest are the parameters
  local params="$@"
  
  echo "========================================"
  echo "Running configuration: $config_name"
  echo "Parameters: $params"
  echo "========================================"
  
  # Run the test
  $RUN_TPCH_SCRIPT $params
  local exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    echo -e "\033[0;32m✓ Configuration '$config_name' PASSED\033[0m"
    PASSED_CONFIGS+=("$config_name")
  else
    echo -e "\033[0;31m✗ Configuration '$config_name' FAILED\033[0m"
    FAILED_CONFIGS+=("$config_name")
  fi
  
  echo ""
  return $exit_code
}

echo "Starting TPC-H tests with scale factor: $SCALE_FACTOR"
echo "All output will be logged to: $LOG_FILE"
echo "========================================"

# Test Configuration 1: Basic run
run_test_config "Basic" $SCALE_FACTOR

# Test Configuration 2: With smaller hash tables
run_test_config "Smaller Hash Tables" $SCALE_FACTOR --smaller-hash-tables

# Test Configuration 3: With bloom filters
run_test_config "Bloom Filters" $SCALE_FACTOR --smaller-hash-tables --use-bloom-filters

# Test Configuration 4: With pyper shuffle
run_test_config "Pyper Shuffle" $SCALE_FACTOR --smaller-hash-tables --pyper-shuffle

# Test Configuration 5: With shuffle all ops
run_test_config "Shuffle All Ops" $SCALE_FACTOR --smaller-hash-tables --shuffle-all-ops

echo "========================================"
echo "FINAL RESULTS SUMMARY"
echo "========================================"
echo "Scale Factor: $SCALE_FACTOR"
echo "Total Configurations Tested: $((${#PASSED_CONFIGS[@]} + ${#FAILED_CONFIGS[@]}))"
echo ""

if [ ${#PASSED_CONFIGS[@]} -gt 0 ]; then
  echo -e "\033[0;32mPASSED CONFIGURATIONS (${#PASSED_CONFIGS[@]}):\033[0m"
  for config in "${PASSED_CONFIGS[@]}"; do
    echo -e "  \033[0;32m✓ $config\033[0m"
  done
  echo ""
fi

if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
  echo -e "\033[0;31mFAILED CONFIGURATIONS (${#FAILED_CONFIGS[@]}):\033[0m"
  for config in "${FAILED_CONFIGS[@]}"; do
    echo -e "  \033[0;31m✗ $config\033[0m"
  done
  echo ""
fi

# Final exit code
if [ ${#FAILED_CONFIGS[@]} -eq 0 ]; then
  echo -e "\033[0;32m🎉 ALL CONFIGURATIONS PASSED! 🎉\033[0m"
  echo "Complete log saved to: $LOG_FILE"
  exit 0
else
  echo -e "\033[0;31m❌ SOME CONFIGURATIONS FAILED ❌\033[0m"
  echo "Complete log saved to: $LOG_FILE"
  exit 1
fi
