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

if [ -n "$CUR_GPU" ]; then
  echo "Using CUR_GPU from environment variable: $CUR_GPU"
else
  echo "CUR_GPU environment variable is not set."
  exit 1
fi

# Set build name variable if not already set
if [ -z "$BUILD_NAME" ]; then
  BUILD_NAME="lingodb-release"
fi

# Get the path of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the path of the parent directory
CUDA_GEN_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$(dirname $(dirname "$SCRIPT_DIR"))"
REPO_DIR="$(dirname "$TEST_DIR")"

TPCH_DIR="$REPO_DIR/resources/sql/tpch"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"
export QUERY_RUNS=10

# Set the data directory if not already set
if [ -z "$TPCH_DATA_DIR" ]; then
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-$SCALE_FACTOR"
fi


QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)
if [ $SCALE_FACTOR -gt 60 ]; then
  QUERIES=(1 3 5 6 7 8 10 12 13 14 16 17 19 20) # Queries 4, 9 and 18 have large hash tables. Run out of memory on A6000
fi

TPCH_CUDA_GEN_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/tpch-$SCALE_FACTOR"
echo "TPCH_CUDA_GEN_DIR: $TPCH_CUDA_GEN_DIR"
pushd $TPCH_CUDA_GEN_DIR

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/tpch-$SCALE_FACTOR/CPU
mkdir -p $OUTPUT_DIR

GEN_CUDF="$BUILD_DIR/gen-cuda $TPCH_DATA_DIR"
for QUERY in "${QUERIES[@]}"; do
# First run the run-sql tool to generate CUDA and get reference output
RES_CSV=$CUDA_GEN_DIR/"tpch-$QUERY-ref.csv"
GEN_CUDF="$GEN_CUDF $TPCH_DIR/$QUERY.sql $TPCH_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $RES_CSV"
done
echo $GEN_CUDF
$GEN_CUDF > /dev/null # ignore the output. We are not comparing

CP_CMD="cp $TPCH_CUDA_GEN_DIR/output.csv $OUTPUT_DIR/tpch-$SCALE_FACTOR-CPU-perf.csv"
echo $CP_CMD
$CP_CMD

cd -

