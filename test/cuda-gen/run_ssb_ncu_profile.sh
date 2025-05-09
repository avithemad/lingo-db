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
TEST_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TEST_DIR")"

SSB_DIR="$REPO_DIR/resources/sql/ssb"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$TSSB_DATA_DIR" ]; then
  SSB_DATA_DIR="$REPO_DIR/resources/data/ssb-1"
fi

# List of queries to run
QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  CD_CMD="cd $SQL_PLAN_COMPILER_DIR/gpu-db/ssb"
  echo $CD_CMD
  $CD_CMD

  REPORT_BASE_FOLDER="$SQL_PLAN_COMPILER_DIR/reports/ncu"
  CUR_GPU="4060" # TODO: Change this when you change GPUs
  REPORT_FOLDER="$REPORT_BASE_FOLDER/$CUR_GPU/ssb-1"

  MAKE_REPORT_FOLDER="mkdir -p $REPORT_FOLDER"
  echo $MAKE_REPORT_FOLDER
  $MAKE_REPORT_FOLDER

  RUN_PROFILE_CMD="ncu --set full --export $REPORT_FOLDER/q$QUERY-ssb.ncu-rep ./build/dbruntime --data_dir $SSB_DATA_DIR/ --query_num $QUERY"
  echo $RUN_PROFILE_CMD
  $RUN_PROFILE_CMD # > op | tee 2>&1

  cd -

done