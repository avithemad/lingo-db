#!/bin/bash

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
TEST_DIR="$(dirname $(dirname "$SCRIPT_DIR"))"
REPO_DIR="$(dirname "$TEST_DIR")"

TPCH_DIR="$REPO_DIR/resources/sql/tpch"
BUILD_DIR="$REPO_DIR/build/$BUILD_NAME"

# Set the data directory if not already set
if [ -z "$TPCH_DATA_DIR" ]; then
  TPCH_DATA_DIR="$REPO_DIR/resources/data/tpch-$SCALE_FACTOR"
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
if [ $SCALE_FACTOR -gt 20 ]; then
  QUERIES=(1 3 4 5 6 7 8 10 12 13 14 16 17 18 19 20)
fi
if [ $SCALE_FACTOR -gt 30 ]; then
  QUERIES=(3 4 5 6 7 8 10 12 13 14 16 17 18 19 20)
fi
QUERIES=(16 17 18 19 20)

# cleanup the result files and logs
rm -f $SCRIPT_DIR/*.csv
rm -f $TPCH_CUDF_PY_DIR/*.csv
rm -f $TPCH_CUDF_PY_DIR/*.log

SRC_DIR="$SQL_PLAN_COMPILER_DIR/gpu-db/tpch-$SCALE_FACTOR"
CD_CMD="cd $SRC_DIR"
echo $CD_CMD
$CD_CMD

REPORT_BASE_FOLDER="$SQL_PLAN_COMPILER_DIR/reports/ncu"
REPORT_FOLDER="$REPORT_BASE_FOLDER/$CUR_GPU/tpch-$SCALE_FACTOR$CRYSTAL_SUFFIX/cudf"

MAKE_REPORT_FOLDER="mkdir -p $REPORT_FOLDER"
echo $MAKE_REPORT_FOLDER
$MAKE_REPORT_FOLDER

CUDF_RUNNER="python $SQL_PLAN_COMPILER_DIR/cudf/tpch/run_cudf_tpch_queries.py"

# Iterate over the queries
for QUERY in "${QUERIES[@]}"; do
  RUN_PROFILE_CMD="ncu --set full -f --export $REPORT_FOLDER/q$QUERY-tpch-$SCALE_FACTOR$CRYSTAL_SUFFIX-cudf.ncu-rep $CUDF_RUNNER $TPCH_DATA_DIR $QUERY timing.csv 1"
  echo $RUN_PROFILE_CMD
  $RUN_PROFILE_CMD # > op | tee 2>&1
done

CUDA_CP_CMD="cp $SRC_DIR/q*.codegen.cu $REPORT_FOLDER"
echo $CUDA_CP_CMD
$CUDA_CP_CMD

cd -