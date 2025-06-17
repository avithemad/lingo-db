#!/bin/bash

# The first argument is the directory where sql-plan-compiler is
CUR_GPU="A6000" # TODO: Change this when you change GPUs
# SQL_PLAN_COMPILER_DIR="$1"
# if [ -z "$SQL_PLAN_COMPILER_DIR" ]; then
#   echo "Usage: $0 <sql-plan-compiler-dir> <cuco-src-path>"
#   exit 1
# fi

# # Second argument is the CUCO source path
# CUCO_SRC_PATH="$2"
# if [ -z "$CUCO_SRC_PATH" ]; then
#   echo "Usage: $0 <sql-plan-compiler-dir> <cuco-src-path>"
#   exit 1
# fi

SCALE_FACTOR=$1
if [ -z "$SCALE_FACTOR" ]; then
  echo "Usage: $0 <scale_factor>"
  exit 1
fi

# Check if --crystal flag is present in command line arguments
CRYSTAL_FLAG=false
CRYSTAL_SUFFIX=""
QUERY_SUFFIX=""
for arg in "$@"; do
  if [ "$arg" == "--crystal" ]; then
    CRYSTAL_FLAG=true
    CRYSTAL_SUFFIX="-crystal"
    QUERY_SUFFIX=".crystal"
    break
  fi
done

# Set build name variable if not already set
if [ -z "$BUILD_NAME" ]; then
  BUILD_NAME="lingodb-debug"
fi

# Get the path of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the path of the parent directory
PROFILE_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$(dirname "$PROFILE_DIR")"
REPO_DIR="$(dirname "$TEST_DIR")"

TPCH_DIR="$REPO_DIR/resources/sql/tpch"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$TPCH_DATA_DIR" ]; then
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-$SCALE_FACTOR"
fi

# List of queries to run - 1, 3, 5, 6, 7, 8, 9
QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  CD_CMD="cd $SQL_PLAN_COMPILER_DIR/gpu-db/tpch-$SCALE_FACTOR"
  echo $CD_CMD
  $CD_CMD

  REPORT_BASE_FOLDER="$SQL_PLAN_COMPILER_DIR/reports/ncu"
  REPORT_FOLDER="$REPORT_BASE_FOLDER/$CUR_GPU/tpch-$SCALE_FACTOR$CRYSTAL_SUFFIX"

  MAKE_REPORT_FOLDER="mkdir -p $REPORT_FOLDER"
  echo $MAKE_REPORT_FOLDER
  $MAKE_REPORT_FOLDER

  RUN_PROFILE_CMD="ncu --set full -f --export $REPORT_FOLDER/q$QUERY-tpch-$SCALE_FACTOR$CRYSTAL_SUFFIX.ncu-rep ./build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $QUERY$QUERY_SUFFIX"
  echo $RUN_PROFILE_CMD
  $RUN_PROFILE_CMD # > op | tee 2>&1

  cd -

done