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
RUN_HYPER_SCRIPT="$SCRIPT_DIR/../run_tpch_tests.sh"

# Check if the run_tpch_tests.sh script exists
if [ ! -f "$RUN_HYPER_SCRIPT" ]; then
  echo "Error: run_tpch_tests.sh not found at $RUN_HYPER_SCRIPT"
  exit 1
fi

RUN_CRYSTAL_SCRIPT="$SCRIPT_DIR/../run_tpch_crystal_tests.sh"

# Check if the run_tpch_crystal_perf_tests.sh script exists
if [ ! -f "$RUN_CRYSTAL_SCRIPT" ]; then
  echo "Error: run_tpch_crystal_perf_tests.sh not found at $RUN_CRYSTAL_SCRIPT"
  exit 1
fi

RUN_PROFILE_SCRIPT="$SCRIPT_DIR/run_tpch_ncu_profile.sh"

# Check if the run_tpch_ncu_profile.sh script exists
if [ ! -f "$RUN_PROFILE_SCRIPT" ]; then
  echo "Error: run_tpch_ncu_profile.sh not found at $RUN_PROFILE_SCRIPT"
  exit 1
fi

# Array to store test results
PASSED_CONFIGS=()
FAILED_CONFIGS=()

# Function to run tests and capture results
run_test_config() {
  local script_path="$1"
  shift
  local config_name="$1"
  shift  # Remove first argument, rest are the parameters
  local params="$@"
  
  echo "========================================"
  echo "Running configuration: $config_name"
  echo "Parameters: $params"
  echo "========================================"
  
  # Run the test
  $script_path $params
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

run_hyper_test_config() {
    run_test_config "$RUN_HYPER_SCRIPT" "$@" --profiling
    run_test_config "$RUN_PROFILE_SCRIPT" "$@"
}

run_crystal_test_config() {
    run_test_config "$RUN_CRYSTAL_SCRIPT" "$@" --profiling
    run_test_config "$RUN_PROFILE_SCRIPT" "$@" --crystal
}

echo "Starting TPC-H tests with scale factor: $SCALE_FACTOR"
echo "All output will be logged to: $LOG_FILE"
echo "========================================"

# Test Configuration 1: Basic run
run_hyper_test_config "Basic" $SCALE_FACTOR

# Test Configuration 2: With smaller hash tables
run_hyper_test_config "Smaller Hash Tables" $SCALE_FACTOR --smaller-hash-tables

# Test Configuration 3: With bloom filters
run_hyper_test_config "Bloom Filters" $SCALE_FACTOR --smaller-hash-tables --use-bloom-filters

# Test Configuration 4: With bloom filters and large hash tables
run_hyper_test_config "Bloom Filters" $SCALE_FACTOR --smaller-hash-tables --use-bloom-filters-for-large-ht

# Test Configuration 5: With bloom filters - large hash tables and small bf
run_hyper_test_config "Bloom Filters" $SCALE_FACTOR --smaller-hash-tables --use-bloom-filters-for-large-ht-small-bf

# Test Configuration 6: With bloom filters - large hash tables and fit bf
run_hyper_test_config "Bloom Filters" $SCALE_FACTOR --smaller-hash-tables --use-bloom-filters-for-large-ht-fit-bf

# Test Configuration 7: With pyper shuffle
run_hyper_test_config "Pyper Shuffle" $SCALE_FACTOR --smaller-hash-tables --pyper-shuffle

# Test Configuration 8: With shuffle all ops
run_hyper_test_config "Shuffle All Ops" $SCALE_FACTOR --smaller-hash-tables --shuffle-all-ops

# Test Configuration 9: Basic crystal run
run_crystal_test_config "Basic" $SCALE_FACTOR

# Test Configuration 10: Crystal with smaller hash tables
run_crystal_test_config "Smaller Hash Tables" $SCALE_FACTOR --smaller-hash-tables

# Test Configuration 11: Crystal with two items per thread
run_crystal_test_config "Two Items Per Thread" $SCALE_FACTOR --smaller-hash-tables --two-items-per-thread

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
