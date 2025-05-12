mkdir -p intermediate_mlir
for i in $(seq 22 22)
do
    build/lingodb-debug/sql-to-mlir resources/sql/tpch/$i.sql resources/data/tpch-1 > intermediate_mlir/tpch_q$i.mlir
    # build/lingodb-debug/mlir-db-opt-old --use-db resources/data/tpch-1 --relalg-query-opt  intermediate_mlir/tpch_q$i.mlir  > intermediate_mlir/tpch_q$i.optimized.mlir
    build/lingodb-debug/mlir-db-opt --use-db resources/data/tpch-1 --gen-cuda-code --tpch --relalg-query-opt intermediate_mlir/tpch_q$i.mlir   > intermediate_mlir/tpch_q$i.optimized.new.mlir
    if [ $? -eq 0 ]; then 
        echo "Copying..."
        cp output.cu ../sql-plan-compiler/gpu-db/tpch/q$i.codegen.cu 
        echo "" > output.cu
    else 
        rm intermediate_mlir/tpch_q$i.optimized.new.mlir
    fi
done