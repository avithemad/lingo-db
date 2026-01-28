#!/bin/bash

CODEGEN_OPTIONS="--threads-always-alive"
FILE_SUFFIX=""
# for each arg in args
SUB_DIR="."
SUFFIX=""
SKIP_GEN=0
CONTINUOUS_ARG=""
LOAD_COLUMNS_PER_QUERY=""
BENCHMARK_NAME="tpch"
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
    --use-bloom-filters-high-sel)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters --bloom-filter-policy-high-sel"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF_HighSel"
      SUFFIX="-ht32-bf-highsel"
      ;;
    --use-bloom-filters-for-large-ht)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters --bloom-filter-policy-large-ht"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF_LargeHT"
      SUFFIX="-ht32-bf-largeht"
      ;;
    --use-bloom-filters-for-large-ht-small-bf)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters --bloom-filter-policy-large-ht-small-bf"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF_LargeHT_SmallBF"
      SUFFIX="-ht32-bf-largeht-smallbf"
      ;;
    --use-bloom-filters-for-large-ht-fit-bf)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters --bloom-filter-policy-large-ht-fit-bf"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF_LargeHT_FitBF"
      SUFFIX="-ht32-bf-largeht-fitbf"
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
      LOAD_COLUMNS_PER_QUERY="--load-columns-per-query"
      SUB_DIR="HT32_PHJ"
      FILE_SUFFIX=".phj"
      SUFFIX="$SUFFIX-phj"
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
    --continuous)
      CONTINUOUS_ARG="--continuous"
      echo "Continuous mode enabled."
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --tile-hashtables)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --tile-hashtables"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="${SUB_DIR}_TiledHT"
      SUFFIX="$SUFFIX-tiledht"
      ;;
    --use-ballot-shuffle)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-ballot-shuffle"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --ssb)
      BENCHMARK_NAME="ssb"
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --ssb --use-multi-map"
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

if [ -n "$CUR_GPU" ]; then
  echo "Using CUR_GPU from environment variable: $CUR_GPU"
else
  echo "CUR_GPU environment variable is not set."
  CUR_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print $2}')
fi

# Set build name variable if not already set
if [ -z "$BUILD_NAME" ]; then
  BUILD_NAME="lingodb-debug"
fi

# Get the path of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the path of the parent directory
CUDA_GEN_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$(dirname $(dirname "$SCRIPT_DIR"))"
REPO_DIR="$(dirname "$TEST_DIR")"

SQL_DIR="$REPO_DIR/resources/sql/$BENCHMARK_NAME"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="$REPO_DIR/resources/data/$BENCHMARK_NAME-$SCALE_FACTOR"
fi

if [ -z "$QUERIES" ]; then
  if [ "$BENCHMARK_NAME" == "ssb" ]; then
    QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)
  else
    QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)
    if [ $SCALE_FACTOR -gt 60 ]; then
      QUERIES=(1 3 5 6 7 8 10 12 13 14 16 17 19 20) # Queries 4, 9 and 18 have large hash tables. Run out of memory on A6000
    fi
  fi
fi

CUDA_GEN_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/$BENCHMARK_NAME-$SCALE_FACTOR"
echo "CUDA_GEN_DIR: $CUDA_GEN_DIR"
pushd $CUDA_GEN_DIR
MAKE_RUNTIME="make build-runtime CUCO_SRC_PATH=$CUCO_SRC_PATH"
echo $MAKE_RUNTIME
$MAKE_RUNTIME

# Check if the make command was successful
if [ $? -ne 0 ]; then
  echo -e "\033[0;31mError building runtime!\033[0m"
  exit 1
fi

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/$BENCHMARK_NAME-$SCALE_FACTOR/$SUB_DIR
mkdir -p $OUTPUT_DIR
OUTPUT_FILE=$OUTPUT_DIR/$BENCHMARK_NAME-$SCALE_FACTOR-hyper$SUFFIX-perf.csv
echo "Output file: $OUTPUT_FILE"

# Empty the output file
echo -n "" > $OUTPUT_FILE
if [ $SKIP_GEN -eq 0 ]; then
  GEN_CUDF="$BUILD_DIR/gen-cuda $DATA_DIR --gen-cuda-code --gen-kernel-timing $CODEGEN_OPTIONS"
  for QUERY in "${QUERIES[@]}"; do
    # First run the run-sql tool to generate CUDA and get reference output
    RES_CSV=$CUDA_GEN_DIR/"$BENCHMARK_NAME-$QUERY-ref.csv"
    GEN_CUDF="$GEN_CUDF $SQL_DIR/$QUERY.sql $CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $RES_CSV"
  done
  echo $GEN_CUDF
  $GEN_CUDF > /dev/null # ignore the output. We are not comparing

  for QUERY in "${QUERIES[@]}"; do
    # format the generated cuda code
    FORMAT_CMD="clang-format -i $CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu -style=Microsoft"
    echo $FORMAT_CMD
    $FORMAT_CMD
  done
fi

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

RUN_QUERY_CMD="build/dbruntime --data_dir $DATA_DIR/ --query_num $QUERIES_STR --op_file $OUTPUT_FILE --scale_factor $SCALE_FACTOR $CONTINUOUS_ARG $LOAD_COLUMNS_PER_QUERY"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

cd -

