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
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-$SCALE_FACTOR"
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

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/tpch-$SCALE_FACTOR/cudf
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)
if [ $CUR_GPU == "4090" ]; then
  if [ $SCALE_FACTOR -gt 10 ]; then
    QUERIES=(3 4 5 6 7 8 10 12 13 14 16 17 18 19 20)
  fi
  if [ $SCALE_FACTOR -gt 20 ]; then
    QUERIES=(3 4 5 6 7 8 10 12 13 14 16 17 18 19 20)
  fi
fi
if [ $CUR_GPU == "A6000" ]; then
  if [ $SCALE_FACTOR -gt 20 ]; then
    QUERIES=(1 3 4 5 6 7 8 10 12 13 14 16 17 18 19 20)
  fi
  if [ $SCALE_FACTOR -gt 30 ]; then
    QUERIES=(3 4 5 6 7 8 10 12 13 14 16 17 18 19 20)
  fi
fi

# cleanup the result files and logs
rm -f $SCRIPT_DIR/*.csv
rm -f $TPCH_CUDF_PY_DIR/*.csv
rm -f $TPCH_CUDF_PY_DIR/*.log

# generate the ref csv files
for QUERY in "${QUERIES[@]}"; do
  # First run the run-sql tool to generate CUDA and get reference output
  OUTPUT_FILE=$SCRIPT_DIR/"tpch-$QUERY-ref.csv"
  RUN_SQL="$BUILD_DIR/run-sql $TPCH_DIR/$QUERY.sql $TPCH_DATA_DIR"
  echo $RUN_SQL
  $RUN_SQL > $OUTPUT_FILE

  if [ $? -ne 0 ]; then
    echo -e "\033[0;31mError running run-sql for Query $QUERY\033[0m"
    exit 1
  fi
done

FAILED_QUERIES=()

# run all the queries
# Convert QUERIES array to comma-separated string
QUERIES_STR=$(IFS=,; echo "${QUERIES[*]}")

TPCH_CUDF_PY_DIR="$SQL_PLAN_COMPILER_DIR/cudf/tpch"
echo "TPCH_CUDF_PY_DIR: $TPCH_CUDF_PY_DIR"
pushd $TPCH_CUDF_PY_DIR

RUN_QUERY_CMD="python run_cudf_tpch_queries.py $TPCH_DATA_DIR/ $QUERIES_STR $OUTPUT_DIR/tpch-$SCALE_FACTOR-cudf.csv"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

cd -

for QUERY in "${QUERIES[@]}"; do
  OUTPUT_FILE="tpch-$QUERY-ref.csv"
  PYTHON_CMD="python $SCRIPT_DIR/compare_tpch_outputs.py $SCRIPT_DIR/$OUTPUT_FILE $TPCH_CUDF_PY_DIR/cudf-tpch-$QUERY.csv --test-has-header"
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