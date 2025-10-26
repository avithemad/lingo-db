#!/bin/bash

CODEGEN_OPTIONS="--smaller-hash-tables"
# for each arg in args
SUB_DIR="."
SUFFIX=""
SKIP_GEN=0
CONTINUOUS_ARG=""
for arg in "$@"; do
  case $arg in
    --smaller-hash-tables)
      # CODEGEN_OPTIONS="$CODEGEN_OPTIONS --smaller-hash-tables" # make this default for now
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32"
      SUFFIX="-ht32"
      ;;
    --use-bloom-filters)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-bloom-filters"
      echo "Use bloom filters option is not supported in crystal codegen."
      exit 1
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF"
      SUFFIX="-ht32-bf"
      ;;
    --threads-always-alive)
      echo "Threads always alive option is not supported in crystal codegen."
      exit 1
      ;;
    --pyper-shuffle)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --pyper-shuffle"
      echo "Pyper shuffle option is not supported in crystal codegen."
      exit 1
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --shuffle-all-ops)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --shuffle-all-ops"
      echo "Shuffle all ops option is not supported in crystal codegen."
      exit 1
      ;;
    --print-hash-table-sizes)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --print-hash-table-sizes"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUFFIX="-HTSIZE"
      ;;
    --two-items-per-thread)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --two-items-per-thread"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR+="_Two_Items_Per_Thread"
      SUFFIX+="-two-items-per-thread"
      ;;
    --one-item-per-thread)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --one-item-per-thread"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR+="_One_Item_Per_Thread"
      SUFFIX+="-one-item-per-thread"
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
CUDA_GEN_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$(dirname $(dirname "$SCRIPT_DIR"))"
REPO_DIR="$(dirname "$TEST_DIR")"

SSB_DIR="$REPO_DIR/resources/sql/ssb"
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

OUTPUT_DIR=$SQL_PLAN_COMPILER_DIR/reports/ncu/$CUR_GPU/ssb-$SCALE_FACTOR-crystal/$SUB_DIR
mkdir -p $OUTPUT_DIR
OUTPUT_FILE=$OUTPUT_DIR/ssb-$SCALE_FACTOR-crystal$SUFFIX-perf.csv
echo "Output file: $OUTPUT_FILE"

# Empty the output file
echo -n "" > $OUTPUT_FILE
if [ $SKIP_GEN -eq 0 ]; then
  FILE_SUFFIX=".crystal"
  GEN_CUDF="$BUILD_DIR/gen-cuda $SSB_DATA_DIR --gen-cuda-crystal-code --gen-kernel-timing --ssb $CODEGEN_OPTIONS"
  for QUERY in "${QUERIES[@]}"; do
    # First run the run-sql tool to generate CUDA and get reference output
    RES_CSV=$CUDA_GEN_DIR/"ssb-$QUERY-ref.csv"
    GEN_CUDF="$GEN_CUDF $SSB_DIR/$QUERY.sql $SSB_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $RES_CSV"
  done
  echo $GEN_CUDF
  $GEN_CUDF > /dev/null # ignore the output. We are not comparing

  if [ $? -ne 0 ]; then
    echo -e "\033[0;31mError generating CUDA code!\033[0m"
    exit 1
  fi

  for QUERY in "${QUERIES[@]}"; do
    # format the generated cuda code
    FORMAT_CMD="clang-format -i $SSB_CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu -style=Microsoft"
    echo $FORMAT_CMD
    $FORMAT_CMD
  done
fi

rm -f build/*.codegen.so # do this so that we don't run other queries by mistake

# generate the cuda files
for QUERY in "${QUERIES[@]}"; do
  MAKE_QUERY="make query Q=$QUERY.crystal CUCO_SRC_PATH=$CUCO_SRC_PATH"
  echo $MAKE_QUERY
  $MAKE_QUERY &
  
  # Check if the make command was successful
done

wait

FAILED_QUERIES=()
for QUERY in "${QUERIES[@]}"; do
  NOCOUNT="$QUERY.crystal"
  if [ ! -f build/q$NOCOUNT.codegen.so ]; then
    echo -e "\033[0;31mError compiling Query $QUERY\033[0m"
    FAILED_QUERIES+=($QUERY)
    exit 1
  fi
done

# run all the queries
# Convert QUERIES array to comma-separated string
QUERIES_STR=""
for i in "${QUERIES[@]}"; do
  QUERIES_STR+="$i.crystal,"
done
QUERIES_STR="${QUERIES_STR%,*}" # remove trailing comma
echo "QUERIES_STR: $QUERIES_STR"

RUN_QUERY_CMD="build/dbruntime --data_dir $SSB_DATA_DIR/ --query_num $QUERIES_STR --op_file $OUTPUT_FILE --scale_factor $SCALE_FACTOR $CONTINUOUS_ARG"
echo $RUN_QUERY_CMD
$RUN_QUERY_CMD

cd -

