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
TEST_DIR="$(dirname $(dirname "$SCRIPT_DIR"))"
REPO_DIR="$(dirname "$TEST_DIR")"

TPCH_DIR="$REPO_DIR/resources/sql/tpch"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"


# Set the data directory if not already set
if [ -z "$TPCH_DATA_DIR" ]; then
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-1"
fi

# List of queries to run - 1, 3, 5, 6, 7, 8, 9
# QUERIES=(1 3 5 6 7 9 13)
# 3, 9, 18
QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)
# QUERIES=(1)

pushd $SQL_PLAN_COMPILER_DIR/gpu-db/tpch
MAKE_RUNTIME="make build-runtime CUCO_SRC_PATH=$CUCO_SRC_PATH"
echo $MAKE_RUNTIME
$MAKE_RUNTIME
popd

OUTPUT_FILE=$SCRIPT_DIR/tpch-hyper-nocount-perf.csv
echo "Output file: $OUTPUT_FILE"

# Empty the output file
echo -n "" > $OUTPUT_FILE

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  # First run the run-sql tool to generate CUDA and get reference output
  COMPILE_SQL="$BUILD_DIR/run-sql $TPCH_DIR/$QUERY.sql $TPCH_DATA_DIR --gen-cuda-code-no-count --gen-kernel-timing"
  echo $COMPILE_SQL
  $COMPILE_SQL

  NOCOUNT="$QUERY.nocount"

  # Now run the generated CUDA code
  CP_CMD="cp output.cu $SQL_PLAN_COMPILER_DIR/gpu-db/tpch/q$NOCOUNT.codegen.cu"
  echo $CP_CMD
  $CP_CMD

  CD_CMD="cd $SQL_PLAN_COMPILER_DIR/gpu-db/tpch"
  echo $CD_CMD
  $CD_CMD

  MAKE_QUERY="make query Q=$NOCOUNT CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_QUERY
  $MAKE_QUERY

  # Append the query number to the output file
  echo "---" >> $OUTPUT_FILE
  echo "tpch-q$NOCOUNT" >> $OUTPUT_FILE

  RUN_QUERY_CMD="build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $NOCOUNT"
  echo $RUN_QUERY_CMD
  $RUN_QUERY_CMD >> $OUTPUT_FILE

  cd -

done
