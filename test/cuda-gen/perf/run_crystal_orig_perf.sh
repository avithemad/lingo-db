#!/bin/bash

CODEGEN_OPTIONS="--smaller-hash-tables"
PROFILING=0
SKIP_GEN=1
# The first argument is the scale factor
SCALE_FACTOR="$1"
if [ -z "$SCALE_FACTOR" ]; then
  echo "Usage: $0 <scale_factor>"
  exit 1
fi

if [ -n "$CRYSTAL_PATH" ]; then
  echo "Using CRYSTAL_PATH from environment variable: $CRYSTAL_PATH"
else
  echo "CRYSTAL_PATH environment variable is not set."
  exit 1
fi

# Check if SQL_PLAN_COMPILER_DIR environment variable is set
if [ -n "$SQL_PLAN_COMPILER_DIR" ]; then
  echo "Using SQL_PLAN_COMPILER_DIR from environment variable: $SQL_PLAN_COMPILER_DIR"
else
  echo "SQL_PLAN_COMPILER_DIR environment variable is not set."
  exit 1
fi

# Set build name variable if not already set
if [ -z "$BUILD_NAME" ]; then
  BUILD_NAME="lingodb-debug"
fi

if [ -n "$CUR_GPU" ]; then
  echo "Using CUR_GPU from environment variable: $CUR_GPU"
else
  echo "CUR_GPU environment variable is not set."
  exit 1
fi

# Get the path of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the path of the parent directory
TEST_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TEST_DIR")"

# Set the data directory if not already set
if [ -z "$SSB_DATA_DIR" ]; then
  SSB_DATA_DIR="$REPO_DIR/resources/data/ssb-$SCALE_FACTOR"
fi

SSB_QUERY_DIR="$REPO_DIR/resources/sql/ssb"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$SSB_DATA_DIR" ]; then
  SSB_DATA_DIR="$REPO_DIR/resources/data/ssb-$SCALE_FACTOR"
fi

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/ssb-$SCALE_FACTOR/crystal-orig
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

if [ -z "$QUERIES" ]; then
  QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)
fi

SSB_CUDA_GEN_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/ssb-$SCALE_FACTOR"
echo "SSB_CUDA_GEN_DIR: $SSB_CUDA_GEN_DIR"
pushd $SSB_CUDA_GEN_DIR


if [ $SKIP_GEN -eq 0 ]; then

  # cleanup the result files, built shared objects
  rm -f build/*.codegen.so # do this so that we don't run other queries by mistake
  rm -f $SCRIPT_DIR/*.csv
  rm -f $SSB_CUDA_GEN_DIR/*.csv
  rm -f $SSB_CUDA_GEN_DIR/*.log
  rm -f $SSB_CUDA_GEN_DIR/*.codegen.cu

  # generate new files
  FILE_SUFFIX=".crystal"
  GEN_CUDF="$BUILD_DIR/gen-cuda $SSB_DATA_DIR --ssb $CODEGEN_OPTIONS"
  for QUERY in "${QUERIES[@]}"; do
    # First run the run-sql tool to generate CUDA and get reference output
    OUTPUT_FILE=$SSB_CUDA_GEN_DIR/ssb-$QUERY-ref.csv
    GEN_CUDF="$GEN_CUDF $SSB_QUERY_DIR/$QUERY.sql $SSB_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $OUTPUT_FILE" 
  done
  echo $GEN_CUDF
  $GEN_CUDF > /dev/null # ignore the output. We are not comparing
fi

# run the crystal orig tests
SSB_UTILS_H="$CRYSTAL_PATH/src/ssb/ssb_utils.h"
sed -i "s/#define SF 1/#define SF $SCALE_FACTOR/" "$SSB_UTILS_H"

pushd $CRYSTAL_PATH

CLEAN_CRYSTAL="make clean"
echo $CLEAN_CRYSTAL
$CLEAN_CRYSTAL

SETUP_CRYSTAL="make setup"
echo $SETUP_CRYSTAL
$SETUP_CRYSTAL

# compile the queries
for QUERY in "${QUERIES[@]}"; do
  MAKE_QUERY="make bin/ssb/q$QUERY"
  echo $MAKE_QUERY
  $MAKE_QUERY &
done

wait

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/ssb-$SCALE_FACTOR/
mkdir -p $OUTPUT_DIR
OUTPUT_FILE=$OUTPUT_DIR/ssb-$SCALE_FACTOR-crystal-orig-perf.csv
echo '' > $OUTPUT_FILE # clear the file
for QUERY in "${QUERIES[@]}"; do
  echo '---' >> $OUTPUT_FILE
  echo "ssb-q$QUERY.crystal" >> $OUTPUT_FILE
  QUERY_PATH="$CRYSTAL_PATH/bin/ssb/q$QUERY"  
  RUN_QUERY="$QUERY_PATH --quiet --timing=1"
  echo $RUN_QUERY
  $RUN_QUERY >> $OUTPUT_FILE
done