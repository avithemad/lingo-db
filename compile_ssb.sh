mkdir -p intermediate_mlir
for i in 11 12 13 21 22 23 31 32 33 34 41 42 43
do
    build/lingodb-debug/sql-to-mlir resources/sql/ssb/$i.sql resources/data/ssb-1 > intermediate_mlir/ssb_q$i.mlir
    build/lingodb-debug/mlir-db-opt-old --use-db resources/data/ssb-1 --relalg-query-opt intermediate_mlir/ssb_q$i.mlir  > intermediate_mlir/ssb_q$i.optimized.mlir
    build/lingodb-debug/mlir-db-opt --use-db resources/data/ssb-1 --relalg-query-opt intermediate_mlir/ssb_q$i.mlir  > intermediate_mlir/ssb_q$i.optimized.new.mlir
    if [ $? -eq 0 ]; then 
        echo "Copying..."
        cp output.cu ../sql-plan-compiler/gpu-db/ssb/q$i.codegen.cu 
        echo "" > output.cu
    else 
        rm intermediate_mlir/ssb_q$i.optimized.new.mlir
    fi
done