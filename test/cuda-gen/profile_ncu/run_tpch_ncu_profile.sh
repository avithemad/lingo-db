#!/bin/bash

CODEGEN_OPTIONS="--threads-always-alive"
# for each arg in args
SUB_DIR="."
SUFFIX=""
CRYSTAL_FLAG=false
CRYSTAL_SUFFIX=""
QUERY_SUFFIX=""
PROFILE_OPTIONS=""
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
    --use-partition-hash-join)
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_PHJ"
      SUFFIX+="-ht32-phj"
      QUERY_SUFFIX=".phj"
      PROFILE_OPTIONS="--nvtx --nvtx-include PROFILE_RANGE/"
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
    --threads-always-alive)
      # CODEGEN_OPTIONS="$CODEGEN_OPTIONS --threads-always-alive" # make this default for now
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      ;;
    --shuffle-all-ops)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --shuffle-all-ops"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_Pyper_Shuffle_All_128"
      SUFFIX="-ht32-pyper-shuffle-all-128"
      ;;
    --crystal)
      CRYSTAL_FLAG=true
      CRYSTAL_SUFFIX="-crystal"
      QUERY_SUFFIX=".crystal"
      set -- "${@/$arg/}"
      ;;
    --two-items-per-thread)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --two-items-per-thread"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR+="_Two_Items_Per_Thread"
      SUFFIX+="-two-items-per-thread"
      CRYSTAL_FLAG=true
      CRYSTAL_SUFFIX="-crystal"
      QUERY_SUFFIX=".crystal"
      ;;
    --one-item-per-thread)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --one-item-per-thread"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR+="_One_Item_Per_Thread"
      SUFFIX+="-one-item-per-thread"
      CRYSTAL_FLAG=true
      CRYSTAL_SUFFIX="-crystal"
      QUERY_SUFFIX=".crystal"
      ;;
    --use-partition-hash-join)
      CODEGEN_OPTIONS="$CODEGEN_OPTIONS --use-partition-hash-join"
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      QUERY_SUFFIX=".phj"
      SUB_DIR="HT32_PHJ"
      SUFFIX="-ht32-phj"
      ;;
  esac
done

SUFFIX+=$PROFILE_SUFFIX

# Assert that --two-items-per-thread is only used with --smaller-hash-tables
if [[ "$CODEGEN_OPTIONS" == *"--two-items-per-thread"* ]] && [[ "$CODEGEN_OPTIONS" != *"--smaller-hash-tables"* ]]; then
  echo "Error: --two-items-per-thread can only be used with --smaller-hash-tables"
  exit 1
fi

if [ -n "$CUR_GPU" ]; then
  echo "Using CUR_GPU from environment variable: $CUR_GPU"
else
  echo "CUR_GPU environment variable is not set."
  exit 1
fi

SCALE_FACTOR=$1
if [ -z "$SCALE_FACTOR" ]; then
  echo "Usage: $0 <scale_factor> [--crystal] "
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
TEST_DIR="$(dirname "$CUDA_GEN_DIR")"
REPO_DIR="$(dirname "$TEST_DIR")"

TPCH_DIR="$REPO_DIR/resources/sql/tpch"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$TPCH_DATA_DIR" ]; then
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-$SCALE_FACTOR"
fi

# List of queries to run - 1, 3, 5, 6, 7, 8, 9
if [ -z "$QUERIES" ]; then
  QUERIES=(1 3 4 5 6 7 8 9 10 12 13 14 16 17 18 19 20)
fi

SRC_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/tpch-$SCALE_FACTOR"
CD_CMD="cd $SRC_DIR"
echo $CD_CMD
$CD_CMD

REPORT_BASE_FOLDER="$SQL_PLAN_COMPILER_DIR/reports/ncu"
REPORT_FOLDER="$REPORT_BASE_FOLDER/$CUR_GPU/tpch-$SCALE_FACTOR$CRYSTAL_SUFFIX/$SUB_DIR"

MAKE_REPORT_FOLDER="mkdir -p $REPORT_FOLDER"
echo $MAKE_REPORT_FOLDER
$MAKE_REPORT_FOLDER

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  RUN_PROFILE_CMD="ncu --set full $PROFILE_OPTIONS -f --export $REPORT_FOLDER/q$QUERY-tpch-$SCALE_FACTOR$CRYSTAL_SUFFIX$SUFFIX.ncu-rep ./build/dbruntime --data_dir $TPCH_DATA_DIR/ --query_num $QUERY$QUERY_SUFFIX"
  echo $RUN_PROFILE_CMD
  $RUN_PROFILE_CMD # > op | tee 2>&1
done

CUDA_CP_CMD="cp $SRC_DIR/q*.codegen.cu $REPORT_FOLDER"
echo $CUDA_CP_CMD
$CUDA_CP_CMD

cd -