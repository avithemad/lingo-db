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

SSB_DIR="$REPO_DIR/resources/sql/ssb"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"


# Set the data directory if not already set
if [ -z "$TSSB_DATA_DIR" ]; then
  SSB_DATA_DIR="$REPO_DIR/resources/data/ssb-1"
fi

# List of queries to run
QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)
# QUERIES=(11)

pushd $SQL_PLAN_COMPILER_DIR/gpu-db/ssb
MAKE_RUNTIME="make build-runtime CUCO_SRC_PATH=$CUCO_SRC_PATH"
echo $MAKE_RUNTIME
$MAKE_RUNTIME
popd

OUTPUT_FILE=$SCRIPT_DIR/ssb-hyper-perf.csv
echo "Output file: $OUTPUT_FILE"

# Empty the output file
echo -n "" > $OUTPUT_FILE

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  # First run the run-sql tool to generate CUDA and get reference output
  COMPILE_SQL="$BUILD_DIR/run-sql $SSB_DIR/$QUERY.sql $SSB_DATA_DIR --gen-cuda-code --ssb --gen-kernel-timing"
  echo $COMPILE_SQL
  $COMPILE_SQL

  # Now run the generated CUDA code
  CP_CMD="cp output.cu $SQL_PLAN_COMPILER_DIR/gpu-db/ssb/q$QUERY.codegen.cu"
  echo $CP_CMD
  $CP_CMD

  CD_CMD="cd $SQL_PLAN_COMPILER_DIR/gpu-db/ssb"
  echo $CD_CMD
  $CD_CMD

  MAKE_QUERY="make query Q=$QUERY CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_QUERY
  $MAKE_QUERY

  # Append the query number to the output file
  echo "---" >> $OUTPUT_FILE
  echo "ssb-q$QUERY" >> $OUTPUT_FILE

  RUN_QUERY_CMD="build/dbruntime --data_dir $SSB_DATA_DIR/ --query_num $QUERY"
  echo $RUN_QUERY_CMD
  $RUN_QUERY_CMD >> $OUTPUT_FILE

  cd -

done

