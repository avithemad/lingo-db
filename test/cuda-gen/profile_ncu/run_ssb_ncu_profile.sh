#!/bin/bash

CODEGEN_OPTIONS="--threads-always-alive --smaller-hash-tables"
# for each arg in args
SUB_DIR="."
SUFFIX=""
CRYSTAL_FLAG=false
CRYSTAL_SUFFIX=""
QUERY_SUFFIX=""
SKIP_GEN=0
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
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
      SUB_DIR="HT32_BF"
      SUFFIX="-ht32-bf"
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
    --skip-gen)
      SKIP_GEN=1
      # Remove this specific argument from $@
      set -- "${@/$arg/}"
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

SSB_QUERY_DIR="$REPO_DIR/resources/sql/ssb"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$SSB_DATA_DIR" ]; then
  SSB_DATA_DIR="$REPO_DIR/resources/data/ssb-$SCALE_FACTOR"
fi

if [ -z "$QUERIES" ]; then
  QUERIES=(11 12 13 21 22 23 31 32 33 34 41 42 43)
fi

if [ $SKIP_GEN -eq 0 ]; then
  # Generate the files
  GEN_CUDF="$BUILD_DIR/gen-cuda $SSB_DATA_DIR --gen-cuda$CRYSTAL_SUFFIX-code --ssb $CODEGEN_OPTIONS --profiling"
  for QUERY in "${QUERIES[@]}"; do
    # First run the run-sql tool to generate CUDA and get reference output
    OUTPUT_FILE=$CUDA_GEN_DIR/"ssb-$QUERY-ref.csv"
    GEN_CUDF="$GEN_CUDF $SSB_QUERY_DIR/$QUERY.sql $CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu $OUTPUT_FILE" 
  done
  echo $GEN_CUDF
  $GEN_CUDF > /dev/null # ignore the output. We are not comparing

  if [ $? -ne 0 ]; then
    echo -e "\033[0;31mError generating CUDA code!\033[0m"
    exit 1
  fi

  for QUERY in "${QUERIES[@]}"; do
    # format the generated cuda code
    FORMAT_CMD="clang-format -i $CUDA_GEN_DIR/q$QUERY$FILE_SUFFIX.codegen.cu -style=Microsoft"
    echo $FORMAT_CMD
    $FORMAT_CMD
  done
fi

SRC_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/ssb-$SCALE_FACTOR"
CD_CMD="cd $SRC_DIR"
echo $CD_CMD
$CD_CMD

REPORT_BASE_FOLDER="$SQL_PLAN_COMPILER_DIR/reports/ncu"
REPORT_FOLDER="$REPORT_BASE_FOLDER/$CUR_GPU/ssb-$SCALE_FACTOR$CRYSTAL_SUFFIX/$SUB_DIR"

MAKE_REPORT_FOLDER="mkdir -p $REPORT_FOLDER"
echo $MAKE_REPORT_FOLDER
$MAKE_REPORT_FOLDER

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  RUN_PROFILE_CMD="ncu --set full -f --export $REPORT_FOLDER/q$QUERY-ssb-$SCALE_FACTOR$CRYSTAL_SUFFIX$SUFFIX.ncu-rep ./build/dbruntime --data_dir $SSB_DATA_DIR/ --query_num $QUERY$QUERY_SUFFIX"
  echo $RUN_PROFILE_CMD
  $RUN_PROFILE_CMD # > op | tee 2>&1
done

CUDA_CP_CMD="cp $SRC_DIR/q*.codegen.cu $REPORT_FOLDER"
echo $CUDA_CP_CMD
$CUDA_CP_CMD

cd -