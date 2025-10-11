#!/bin/bash

CODEGEN_OPTIONS="--threads-always-alive" # --smaller-hash-tables"
FILE_SUFFIX="" 
USE_RUN_SQL=0 # set to 1 to use run-sql to generate cuda code. 0 to use batch gen-cuda
PROFILING=0
SKIP_GEN=0
# for each arg in args
for arg in "$@"; do
  case $arg in
    --smaller-hash-tables)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --smaller-hash-tables" # make this default for now
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --use-bloom-filters)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --use-bloom-filters-for-large-ht)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters --bloom-filter-policy-large-ht"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --use-bloom-filters-for-large-ht-small-bf)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters --bloom-filter-policy-large-ht-small-bf"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --use-bloom-filters-for-large-ht-fit-bf)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters --bloom-filter-policy-large-ht-fit-bf"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --threads-always-alive)
      # CODEGEN_OPTIONS="$CODEGEN_OPTIONS --threads-always-alive"
      # Remove this specific argument from $@ # make this default for now
      set -- "${@/$arg/}"
      ;;
    --pyper-shuffle)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --pyper-shuffle"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --profiling)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --profiling"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      USE_RUN_SQL=0
      PROFILING=1
      ;;
    --shuffle-all-ops)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --shuffle-all-ops"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --use-partition-hash-join)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-partition-hash-join"
      FILE_SUFFIX=".phj"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --enable-logging)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --enable-logging"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --skip-gen)
      SKIP_GEN=1
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
  esac
done

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

TPCH_DIR="$REPO_DIR/resources/sql/tpch"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$TPCH_DATA_DIR" ]; then
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-$SCALE_FACTOR"
fi

if [ -z "$QUERIES" ]; then
  QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)
fi

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

if [ $SKIP_GEN -eq 0 ]; then

  # cleanup the result files, built shared objects
  rm -f build/*.codegen.so # do this so that we don't run other queries by mistake
  rm -f $SCRIPT_DIR/*.csv
  rm -f $TPCH_CUDA_GEN_DIR/q*$FILE_SUFFIX.codegen.cu
  rm -f $TPCH_CUDA_GEN_DIR/*.csv
  rm -f $TPCH_CUDA_GEN_DIR/*.log
  
  echo "Using batch gen-cuda"    
  GEN_CUDF="$BUILD_DIR/gen-cuda $TPCH_DATA_DIR --gen-cuda-code $CODEGEN_OPTIONS"
  # First run the run-sql tool to generate CUDA and get reference output
  for QUERY in "${QUERIES[@]}"; do      
    OUTPUT_FILE=$SCRIPT_DIR/"tpch-$QUERY-ref.csv"
    GEN_CUDF="$GEN_CUDF $TPCH_DIR/$QUERY.sql $TPCH_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $OUTPUT_FILE" 
  done

  echo $GEN_CUDF
  $GEN_CUDF > /dev/null # ignore the output

  for QUERY in "${QUERIES[@]}"; do
    # format the generated cuda code
    FORMAT_CMD="clang-format -i $TPCH_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu -style=Microsoft"
    echo $FORMAT_CMD
    $FORMAT_CMD
  done
fi

# delete the temporary output.cu file
rm output.cu

# compile the cuda files
for QUERY in "${QUERIES[@]}"; do
  MAKE_QUERY="make query Q=$QUERY$FILE_SUFFIX CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_QUERY
  $MAKE_QUERY &
  
  # Check if the make command was successful
done

wait

FAILED_QUERIES=()
for QUERY in "${QUERIES[@]}"; do
  if [ ! -f build/q$QUERY$FILE_SUFFIX.codegen.so ]; then
    echo -e "\033[0;31mError compiling Query $QUERY\033[0m"
    exit 1
    FAILED_QUERIES+=($QUERY)
  fi
done

if [ $PROFILING -eq 1 ]; then
  echo "Profiling enabled. Skipping execution and output comparison."
  exit 0
fi

# run all the queries
# Convert QUERIES array to comma-separated string
QUERIES_WITH_SUFFIX=()
for Q in "${QUERIES[@]}"; do
  QUERIES_WITH_SUFFIX+=("$Q$FILE_SUFFIX")
done
QUERIES_STR=$(IFS=,; echo "${QUERIES_WITH_SUFFIX[*]}")

echo $QUERIES_STR

RUN_QUERY_CMD="build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $QUERIES_STR"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

cd -

for QUERY in "${QUERIES[@]}"; do
  OUTPUT_FILE="tpch-$QUERY-ref.csv"
  PYTHON_CMD="python $SCRIPT_DIR/compare_tpch_outputs.py $SCRIPT_DIR/$OUTPUT_FILE $TPCH_CUDA_GEN_DIR/cuda-tpch-$QUERY$FILE_SUFFIX.csv"
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