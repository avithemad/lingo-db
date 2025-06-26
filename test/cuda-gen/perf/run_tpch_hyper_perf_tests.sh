#!/bin/bash

# The first argument is the scale factor
SCALE_FACTOR="$1"
if [ -z "$SCALE_FACTOR" ]; then
  echo "Usage: $0 <scale_factor>"
  exit 1
fi

# Check if SQL_PLAN_COMPILER_DIR environment variable is set
if [ -n "$SQL_PLAN_COMPILER_DIR" ]; then
  echo "Using SQL_PLAN_COMPILER_DIR from environment variable: $SQL_PLAN_COMPILER_DIR"
else
  echo "SQL_PLAN_COMPILER_DIR environment variable is not set."
  exit 1
fi

# Check if CUCO_SRC_PATH environment variable is set
if [ -n "$CUCO_SRC_PATH" ]; then
  echo "Using CUCO_SRC_PATH from environment variable: $CUCO_SRC_PATH"
else
  echo "CUCO_SRC_PATH environment variable is not set."
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
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-$SCALE_FACTOR"
fi


QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)

TPCH_CUDA_GEN_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/tpch-$SCALE_FACTOR"
echo "TPCH_CUDA_GEN_DIR: $TPCH_CUDA_GEN_DIR"
pushd $TPCH_CUDA_GEN_DIR
MAKE_RUNTIME="make build-runtime CUCO_SRC_PATH=$CUCO_SRC_PATH"
echo $MAKE_RUNTIME
$MAKE_RUNTIME

# Check if the make command was successful
if [ $? -ne 0 ]; then
  echo -e "\033[0;31mError building runtime!\033[0m"
  exit 1
fi

OUTPUT_FILE=$SCRIPT_DIR/tpch-$SCALE_FACTOR-hyper-perf.csv
echo "Output file: $OUTPUT_FILE"

# Empty the output file
echo -n "" > $OUTPUT_FILE

# generate the cuda files
for QUERY in "${QUERIES[@]}"; do
  # First run the run-sql tool to generate CUDA and get reference output
  RUN_SQL="$BUILD_DIR/run-sql $TPCH_DIR/$QUERY.sql $TPCH_DATA_DIR --gen-cuda-code --gen-kernel-timing"
  echo $RUN_SQL
  $RUN_SQL > /dev/null # ignore the output. We are not comparing the results.

  # Now run the generated CUDA code
  CP_CMD="cp output.cu $TPCH_CUDA_GEN_DIR/q$QUERY.codegen.cu"
  echo $CP_CMD
  $CP_CMD
done

# generate the cuda files
for QUERY in "${QUERIES[@]}"; do
  rm -f build/q$QUERY.codegen.so
  MAKE_QUERY="make query Q=$QUERY CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_QUERY
  $MAKE_QUERY &
  
  # Check if the make command was successful
done

wait

FAILED_QUERIES=()
for QUERY in "${QUERIES[@]}"; do
  if [ ! -f build/q$QUERY.codegen.so ]; then
    echo -e "\033[0;31mError compiling Query $QUERY\033[0m"
    FAILED_QUERIES+=($QUERY)
  fi
done

# run all the queries
# Convert QUERIES array to comma-separated string
QUERIES_STR=$(IFS=,; echo "${QUERIES[*]}")

RUN_QUERY_CMD="build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $QUERIES_STR --op_file $OUTPUT_FILE --scale_factor $SCALE_FACTOR"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

cd -

