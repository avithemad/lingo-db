SCALE_FACTOR=$1
if [ -z "$SCALE_FACTOR" ]; then
    echo "Usage: $0 <scale_factor>"
    exit 1
fi

if [ -z "$SQL_PLAN_COMPILER_DIR" ]; then
  echo "SQL_PLAN_COMPILER_DIR environment variable is not set."
  exit 1
fi

TPCH_DIR=$SQL_PLAN_COMPILER_DIR/gpu-db/tpch-$SCALE_FACTOR/
HEADER_FILE=$TPCH_DIR/ht_profile_config.h
export QUERIES=(9)
ranks=(1 2 3)
ht_options=(0 1)
for rank in ${ranks[@]}; do
    for ht_option in ${ht_options[@]}; do
        sed -i "s/#define HASH_TABLE_RANK .*/#define HASH_TABLE_RANK $rank/" $HEADER_FILE
        sed -i "s/#define ENABLE_HASH_TABLE .*/#define ENABLE_HASH_TABLE $ht_option/" $HEADER_FILE
        if [ $ht_option -eq 1 ]; then
            export PROFILE_SUFFIX="-r${rank}-ht"
        else
            export PROFILE_SUFFIX="-r${rank}-no-ht"
        fi
        cd ..
        ./run_tpch_tests.sh $SCALE_FACTOR --smaller-hash-tables --profiling --skip-gen
        cd ./profile_ncu
        ./run_tpch_ncu_profile.sh $SCALE_FACTOR --smaller-hash-tables
    done
done