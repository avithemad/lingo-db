#!/bin/bash

CODEGEN_OPTIONS="--smaller-hash-tables"
PROFILING=0
SKIP_GEN=0
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
TEST_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TEST_DIR")"

SSB_QUERY_DIR="$REPO_DIR/resources/sql/ssb"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$SSB_DATA_DIR" ]; then
  SSB_DATA_DIR="$REPO_DIR/resources/data/ssb-$SCALE_FACTOR"
fi

if [ -z "$QUERIES" ]; then
  QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)
fi

SSB_CUDA_GEN_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/ssb-$SCALE_FACTOR"
echo "SSB_CUDA_GEN_DIR: $SSB_CUDA_GEN_DIR"
pushd $SSB_CUDA_GEN_DIR
MAKE_RUNTIME="make build-runtime CUCO_SRC_PATH=$CUCO_SRC_PATH"
echo $MAKE_RUNTIME
$MAKE_RUNTIME

# Check if the make command was successful
if [ $? -ne 0 ]; then
  echo -e "\033[0;31mError building runtime!\033[0m"
  exit 1
fi

if [ $SKIP_GEN -eq 0 ]; then

  # cleanup the result files, built shared objects
  rm -f build/*.codegen.so # do this so that we don't run other queries by mistake
  rm -f $SCRIPT_DIR/*.csv
  rm -f $SSB_CUDA_GEN_DIR/*.csv
  rm -f $SSB_CUDA_GEN_DIR/*.log
  rm -f $SSB_CUDA_GEN_DIR/*.codegen.cu

  # generate new files
  FILE_SUFFIX=".crystal"
  GEN_CUDF="$BUILD_DIR/gen-cuda $SSB_DATA_DIR --gen-cuda-crystal-code --ssb $CODEGEN_OPTIONS"
  for QUERY in "${QUERIES[@]}"; do
    # First run the run-sql tool to generate CUDA and get reference output
    OUTPUT_FILE=$SSB_CUDA_GEN_DIR/"ssb-$QUERY-ref.csv"
    GEN_CUDF="$GEN_CUDF $SSB_DIR/$QUERY.sql $SSB_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $OUTPUT_FILE" 
  done
  echo $GEN_CUDF
  $GEN_CUDF > /dev/null # ignore the output. We are not comparing

  for QUERY in "${QUERIES[@]}"; do
    # format the generated cuda code
    FORMAT_CMD="clang-format -i $SSB_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu -style=Microsoft"
    echo $FORMAT_CMD
    $FORMAT_CMD
  done
fi

# compile the cuda files
for QUERY in "${QUERIES[@]}"; do
  MAKE_QUERY="make query Q=$QUERY.crystal CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_QUERY
  $MAKE_QUERY &
  
done

wait

# Check if the make command was successful
FAILED_QUERIES=()
for QUERY in "${QUERIES[@]}"; do
  if [ ! -f build/q$QUERY.crystal.codegen.so ]; then
    echo -e "\033[0;31mError compiling Query $QUERY\033[0m"
    FAILED_QUERIES+=($QUERY)
    exit 1
  fi
done

if [ $PROFILING -eq 1 ]; then
  echo "Profiling enabled. Skipping execution and output comparison."
  exit 0
fi

# run all the queries
# Convert QUERIES array to comma-separated string
QUERIES_STR=""
for i in "${QUERIES[@]}"; do
  QUERIES_STR+="$i.crystal,"
done
QUERIES_STR="${QUERIES_STR%,*}" # remove trailing comma

RUN_QUERY_CMD="build/dbruntime --data_dir $SSB_DATA_DIR/ --query_num $QUERIES_STR"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

  cd -

for QUERY in "${QUERIES[@]}"; do
  OUTPUT_FILE="ssb-$QUERY-ref.csv"
  PYTHON_CMD="python $SCRIPT_DIR/compare_tpch_outputs.py $SSB_CUDA_GEN_DIR/$OUTPUT_FILE $SSB_CUDA_GEN_DIR/cuda-ssb-$QUERY.crystal.csv"
  echo $PYTHON_CMD
  $PYTHON_CMD

  # If the comparison fails, add query to the failed list
  if [ $? -ne 0 ]; then
    echo -e "\033[0;31mQuery $QUERY failed\033[0m"
    FAILED_QUERIES+=($QUERY)
  else
    echo -e "\033[0;32mQuery $QUERY passed\033[0m"
    PASSED_QUERIES+=($QUERY)
  fi
done

# Print the results
# Print some new lines and then a seperator
echo -e "\n"
echo "=============================="
echo "SSB TEST RESULTS"
echo "=============================="
echo "Passed queries: ${PASSED_QUERIES[@]}"
echo "Failed queries: ${FAILED_QUERIES[@]}"
echo -e "\n"
if [ ${#FAILED_QUERIES[@]} -eq 0 ]; then
  # Print "TEST PASSED" in green
  echo -e "\033[0;32mTEST PASSED!\033[0m"
  exit 0
else
  # Print "TEST FAILED" in red
  echo -e "\033[0;31mTEST FAILED!\033[0m"
  exit 1
fi