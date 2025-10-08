#!/bin/bash

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

# Set the data directory if not already set
if [ -z "$SSB_DATA_DIR" ]; then
  SSB_DATA_DIR="$REPO_DIR/resources/data/ssb-$SCALE_FACTOR"
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

SSB_QUERY_DIR="$REPO_DIR/resources/sql/ssb"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/ssb-$SCALE_FACTOR/crystal-orig
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)

# cleanup the result files and logs
rm -f $SCRIPT_DIR/*.csv
rm -f $TPCH_CUDF_PY_DIR/*.csv
rm -f $TPCH_CUDF_PY_DIR/*.log

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

# compiler the queries
for QUERY in "${QUERIES[@]}"; do
  MAKE_QUERY="make bin/ssb/q$QUERY"
  echo $MAKE_QUERY
  $MAKE_QUERY &
done

wait

CRYSTAL_OP_PATH="$CRYSTAL_PATH/results/ssb-$SCALE_FACTOR/"
mkdir -p $CRYSTAL_OP_PATH
for QUERY in "${QUERIES[@]}"; do
  QUERY_PATH="$CRYSTAL_PATH/bin/ssb/q$QUERY"
  OUTPUT_FILE=$CRYSTAL_OP_PATH/ssb-$QUERY-crystal-orig.csv
  RUN_QUERY="$QUERY_PATH --quiet"
  echo $RUN_QUERY
  $RUN_QUERY > $OUTPUT_FILE
done

exit 1

# generate the ref csv files
FILE_SUFFIX=".crystal"
GEN_CUDF="$BUILD_DIR/gen-cuda $TPCH_DATA_DIR --gen-cuda-crystal-code $CODEGEN_OPTIONS"
for QUERY in "${QUERIES[@]}"; do
  # First run the run-sql tool to generate CUDA and get reference output
  OUTPUT_FILE=$SCRIPT_DIR/"tpch-$QUERY-ref.csv"
  GEN_CUDF="$GEN_CUDF $TPCH_DIR/$QUERY.sql $TPCH_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $OUTPUT_FILE" 
done
echo $GEN_CUDF
$GEN_CUDF > /dev/null # ignore the output. We are not comparing

for QUERY in "${QUERIES[@]}"; do
  # format the generated cuda code
  FORMAT_CMD="clang-format -i $TPCH_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu -style=Microsoft"
  echo $FORMAT_CMD
  $FORMAT_CMD
done

FAILED_QUERIES=()

# run all the queries
# Convert QUERIES array to comma-separated string
QUERIES_STR=$(IFS=,; echo "${QUERIES[*]}")

TPCH_CUDF_PY_DIR="$SQL_PLAN_COMPILER_DIR/cudf/tpch"
echo "TPCH_CUDF_PY_DIR: $TPCH_CUDF_PY_DIR"
pushd $TPCH_CUDF_PY_DIR

RUN_QUERY_CMD="python run_cudf_tpch_queries.py $SSB_DATA_DIR/ $QUERIES_STR $OUTPUT_DIR/ssb-$SCALE_FACTOR-cudf.csv"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

cd -

for QUERY in "${QUERIES[@]}"; do
  OUTPUT_FILE="ssb-$QUERY-ref.csv"
  PYTHON_CMD="python $SCRIPT_DIR/compare_tpch_outputs.py $SCRIPT_DIR/$OUTPUT_FILE $TPCH_CUDF_PY_DIR/cudf-ssb-$QUERY.csv --test-has-header"
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
echo "TPCH TEST RESULTS"
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