#!/bin/bash

# The first argument is the directory where sql-plan-compiler is
SQL_PLAN_COMPILER_DIR="$1"
if [ -z "$SQL_PLAN_COMPILER_DIR" ]; then
  echo "Usage: $0 <sql-plan-compiler-dir> <cuco-src-path>"
  exit 1
fi

# Second argument is the CUCO source path
CUCO_SRC_PATH="$2"
if [ -z "$CUCO_SRC_PATH" ]; then
  echo "Usage: $0 <sql-plan-compiler-dir> <cuco-src-path>"
  exit 1
fi

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
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-1"
fi

# List of queries to run - 1, 3, 5, 6, 7, 8, 9
QUERIES=(1 3 5 6 7 9 13)

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  CD_CMD="cd $SQL_PLAN_COMPILER_DIR/gpu-db/tpch"
  echo $CD_CMD
  $CD_CMD

  REPORT_BASE_FOLDER="$SQL_PLAN_COMPILER_DIR/reports/ncu"
  CUR_GPU="4060" # TODO: Change this when you change GPUs
  REPORT_FOLDER="$REPORT_BASE_FOLDER/$CUR_GPU/tpch-1" # TODO: Get this from the data-dir

  MAKE_REPORT_FOLDER="mkdir -p $REPORT_FOLDER"
  echo $MAKE_REPORT_FOLDER
  $MAKE_REPORT_FOLDER

  RUN_PROFILE_CMD="ncu --set full -f --export $REPORT_FOLDER/q$QUERY-tpch.ncu-rep ./build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $QUERY"
  echo $RUN_PROFILE_CMD
  $RUN_PROFILE_CMD # > op | tee 2>&1

  cd -

done