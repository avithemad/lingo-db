#!/bin/bash

CODEGEN_OPTIONS="--threads-always-alive"
FILE_SUFFIX=""
# for each arg in args
SUB_DIR="."
SUFFIX=""
for arg in "$@"; do
  case $arg in
    --smaller-hash-tables)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --smaller-hash-tables" # make this default for now
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32"
      SUFFIX="-ht32"
      ;;
    --use-bloom-filters)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF"
      SUFFIX="-ht32-bf"
      ;;
    --use-bloom-filters-for-large-ht)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF_LargeHT"
      SUFFIX="-ht32-bf-largeht"
      ;;
    --use-bloom-filters-for-large-ht-small-bf)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF_LargeHT_SmallBF"
      SUFFIX="-ht32-bf-largeht-smallbf"
      ;;
    --threads-always-alive)
      # CODEGEN_OPTIONS="$CODEGEN_OPTIONS --threads-always-alive" # make this default for now
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --pyper-shuffle)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --pyper-shuffle"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_Pyper_Two_Warps_128"
      SUFFIX="-ht32-pyper-two-warps-128"
      ;;
    --shuffle-all-ops)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --shuffle-all-ops"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_Pyper_Shuffle_All_128"
      SUFFIX="-ht32-pyper-shuffle-all-128"
      ;;
    --print-hash-table-sizes)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --print-hash-table-sizes"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUFFIX="$SUFFIX-HTSIZE"
      ;;
    --use-partition-hash-join)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-partition-hash-join"
      FILE_SUFFIX=".phj"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
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

if [ -n "$CUR_GPU" ]; then
  echo "Using CUR_GPU from environment variable: $CUR_GPU"
else
  echo "CUR_GPU environment variable is not set."
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
if [ $SCALE_FACTOR -gt 60 ]; then
  QUERIES=(1 3 5 6 7 8 10 12 13 14 16 17 19 20) # Queries 4, 9 and 18 have large hash tables. Run out of memory on A6000
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

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/tpch-$SCALE_FACTOR/$SUB_DIR
mkdir -p $OUTPUT_DIR
OUTPUT_FILE=$OUTPUT_DIR/tpch-$SCALE_FACTOR-hyper$SUFFIX-perf.csv
echo "Output file: $OUTPUT_FILE"

# Empty the output file
echo -n "" > $OUTPUT_FILE

# generate the cuda files
for QUERY in "${QUERIES[@]}"; do
  # First run the run-sql tool to generate CUDA and get reference output
  RUN_SQL="$BUILD_DIR/run-sql $TPCH_DIR/$QUERY.sql $TPCH_DATA_DIR --gen-cuda-code --gen-kernel-timing $CODEGEN_OPTIONS"
  echo $RUN_SQL
  $RUN_SQL > /dev/null # ignore the output. We are not comparing the results.

  # format the generated cuda code
  FORMAT_CMD="clang-format -i output.cu -style=Microsoft"
  echo $FORMAT_CMD
  $FORMAT_CMD

  # Now run the generated CUDA code
  CP_CMD="cp output.cu $TPCH_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu"
  echo $CP_CMD
  $CP_CMD
done

rm -f build/*.codegen.so # do this so that we don't run other queries by mistake

# generate the cuda files
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
    FAILED_QUERIES+=($QUERY)
    exit 1
  fi
done

# run all the queries
# Convert QUERIES array to comma-separated string
QUERIES_WITH_SUFFIX=()
for Q in "${QUERIES[@]}"; do
  QUERIES_WITH_SUFFIX+=("$Q$FILE_SUFFIX")
done
QUERIES_STR=$(IFS=,; echo "${QUERIES_WITH_SUFFIX[*]}")

echo $QUERIES_STR

RUN_QUERY_CMD="build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $QUERIES_STR --op_file $OUTPUT_FILE --scale_factor $SCALE_FACTOR"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

cd -

