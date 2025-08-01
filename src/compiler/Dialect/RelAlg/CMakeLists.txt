add_mlir_dialect_library(MLIRRelAlg
        RelAlgDialect.cpp
        RelAlgOps.cpp
        RelAlgOpsEnums.cpp
        OperatorInterfaceImpl.cpp
        Transforms/CudaCodeGen.cpp
        Transforms/CudaCodeGenNoCount.cpp
        Transforms/CudaCrystalCodeGen.cpp
        Transforms/CudaCrystalCodeGenNoCount.cpp
        Transforms/CudaCodeGenHelper.cpp
        Transforms/MetaData.cpp
        Transforms/InferNotNullConditions.cpp
        Transforms/ColumnFolding.cpp
        Transforms/ExtractNestedOperators.cpp
        Transforms/SimplifyAggregations.cpp
        Transforms/DecomposeLambdas.cpp
        Transforms/CombinePredicates.cpp
        Transforms/OptimizeImplementations.cpp
        Transforms/PropagateConstraints.cpp
        Transforms/IntroduceTmp.cpp
        Transforms/Unnesting.cpp
        Transforms/Pushdown.cpp
        Transforms/ImplicitToExplicitJoins.cpp
        Transforms/OptimizeJoinOrder.cpp
        Transforms/TrackTuples.cpp
        Transforms/ShuffleAnalysis.cpp
        Transforms/queryopt/utils.cpp
        Transforms/queryopt/QueryGraphBuilder.cpp
        Transforms/queryopt/DPhyp.cpp
        Transforms/queryopt/GOO.cpp
        Transforms/queryopt/QueryGraph.cpp
        Passes.cpp
        ADDITIONAL_HEADER_DIRS
        ../../../../include/lingodb/compiler/Dialect/RelAlg

        DEPENDS
        MLIRRelAlgOpsIncGen
        MLIRDBOpsIncGen
        MLIRDSAOpsIncGen
        MLIRUtilOpsIncGen
        MLIRTupleStreamOpsIncGen
        LINK_LIBS mlir-support fmt PUBLIC
        MLIRIR
)
