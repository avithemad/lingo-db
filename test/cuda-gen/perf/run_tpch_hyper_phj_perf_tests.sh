#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Forward all arguments to run_tpch_hyper_perf_tests.sh with --use-partition-hash-join
"$DIR/run_tpch_hyper_perf_tests.sh" "$@" "-phj" --use-partition-hash-join 