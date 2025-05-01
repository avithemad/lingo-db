#!/bin/bash

# The first argument is the directory where sql-plan-compiler is
SQL_PLAN_COMPILER_DIR="$1"
if [ -z "$SQL_PLAN_COMPILER_DIR" ]; then
  echo "Usage: $0 <sql-plan-compiler-dir>"
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

TPCH_DIR="$REPO_DIR/resources/sql/tpch"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"


# Set the data directory if not already set
if [ -z "$TPCH_DATA_DIR" ]; then
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-1"
fi

# List of queries to run - 1, 3, 5, 6, 7, 8, 9
# QUERIES=(1 3 5 6 7 9 13)
QUERIES=(13)

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  # First run the run-sql tool to generate CUDA and get reference output
  RUN_SQL="$BUILD_DIR/run-sql $TPCH_DIR/$QUERY.sql $TPCH_DATA_DIR"
  OUTPUT_FILE="tpch-$QUERY-ref.csv"
  echo $RUN_SQL
  $RUN_SQL > $OUTPUT_FILE

  # Now run the generated CUDA code
  CP_CMD="cp output.cu $SQL_PLAN_COMPILER_DIR/gpu-db/tpch/q$QUERY.codegen.cu"
  echo $CP_CMD
  $CP_CMD

  CD_CMD="cd $SQL_PLAN_COMPILER_DIR/gpu-db/tpch"
  echo $CD_CMD
  $CD_CMD

  MAKE_QUERY="make query Q=$QUERY CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_QUERY
  $MAKE_QUERY

  MAKE_RUNTIME="make build-runtime CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_RUNTIME
  $MAKE_RUNTIME

  RUN_QUERY_CMD="build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $QUERY"
  echo $RUN_QUERY_CMD
  $RUN_QUERY_CMD > "cuda-tpch-$QUERY.csv"

  cd -

  PYTHON_CMD="python $SCRIPT_DIR/compare_tpch_outputs.py $OUTPUT_FILE $SQL_PLAN_COMPILER_DIR/gpu-db/tpch/cuda-tpch-$QUERY.csv"
  echo $PYTHON_CMD
  $PYTHON_CMD
done