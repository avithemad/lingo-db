#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace lingodb::compiler::dialect;

namespace {

// Big TODO: Rewrite this using the mlir::Analysis and mlir Rewriter framework.
class ColumnDetail {
  std::string columnName;
  std::string tableName;
  mlir::Operation* definingOp;
public:
  ColumnDetail(std::string table, std::string column, mlir::Operation* definingOp)
      : columnName(column), tableName(table), definingOp(definingOp) {}

  std::string getColumnName() const { return columnName; }
  std::string getTableName() const { return tableName; }
  mlir::Operation* getDefiningOp() const { return definingOp; }
  void setDefiningOp(mlir::Operation* op) {
    definingOp = op;
  }
};

class TupleStreamCode {
  std::vector<ColumnDetail> columns;
  uint32_t id;
  friend class ShuffleAnalysisPass;
public:
  TupleStreamCode(uint32_t streamID, relalg::BaseTableOp scanOp) : id(streamID) {    
    std::string tableName = scanOp.getTableIdentifier().data();
    for (auto namedAttr : scanOp.getColumns().getValue()) {
        auto columnName = namedAttr.getName().str();
        columns.emplace_back(tableName, columnName, scanOp);
    }
  }
  void MarkColumnsTouchedByJoin(relalg::InnerJoinOp joinOp, std::shared_ptr<TupleStreamCode> buildStreamCode) {

    for (auto& touchedColumn : buildStreamCode->columns) {
      auto it = std::find_if(columns.begin(), columns.end(),
                     [&](const ColumnDetail& col) {
                       return col.getColumnName() == touchedColumn.getColumnName() &&
                              col.getTableName() == touchedColumn.getTableName();
                     });
      if (it == columns.end()) {
        // If the column is not already in the list, add it
        columns.push_back(ColumnDetail(touchedColumn.getTableName(), touchedColumn.getColumnName(), joinOp));
      } else {
          it->setDefiningOp(joinOp); // Update the defining operation to the join operation
      }
    }
  }
};

class ShuffleAnalysisPass
    : public mlir::PassWrapper<ShuffleAnalysisPass, mlir::OperationPass<mlir::ModuleOp>> {

  std::map<mlir::Operation*, uint32_t> streamID; // every operation belongs to a stream
  std::map<uint32_t, std::shared_ptr<TupleStreamCode>> streamCodeMap; // map of stream ID to TupleStreamCode
  std::map<relalg::InnerJoinOp, bool> shouldAllocateShuffleBuffer; // map of InnerJoinOp to whether it should allocate a shuffle buffer
  uint32_t nextStreamID = 0;

  void _createNewStream(mlir::Operation* op) {
    // This function creates a new stream for the given operation.
    assert(streamID.find(op) == streamID.end() && "Stream ID already assigned");
    streamCodeMap[nextStreamID] = std::make_shared<TupleStreamCode>(nextStreamID, mlir::cast<relalg::BaseTableOp>(op));
    streamID[op] = nextStreamID++;
  }

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShuffleAnalysisPass);

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<relalg::RelAlgDialect>();
  }

  void walkInnerJoin(relalg::InnerJoinOp innerJoinOp) {
    // This function is a placeholder for any specific logic needed for InnerJoinOp.
    // Currently, it just dumps the operation.
    innerJoinOp->dump();

    // the join builds a hash table in one pipeline and probes it in another
    auto buildSideOp = innerJoinOp.getLeftMutable().get().getDefiningOp();
    auto probeSideOp = innerJoinOp.getRightMutable().get().getDefiningOp();

    buildSideOp->dump();
    probeSideOp->dump();

    auto buildSideStreamID = streamID.find(buildSideOp);
    auto probeSideStreamID = streamID.find(probeSideOp);

    assert(buildSideStreamID != streamID.end() && "Join build side stream doesn't exist!");
    assert(probeSideStreamID != streamID.end() && "Join probe side stream doesn't exist!");

    auto currentStreamID = probeSideStreamID->second;
    streamID[innerJoinOp] = currentStreamID;

    // The probe is going to change the defining op of the columns on the probe side StreamID
    auto probeStreamCode = streamCodeMap[currentStreamID];
    assert(probeStreamCode && "Probe side stream code not found!");
    probeStreamCode->MarkColumnsTouchedByJoin(innerJoinOp, streamCodeMap[buildSideStreamID->second]);
  }

  void _checkForColumnInStream(std::shared_ptr<TupleStreamCode> streamCode, const std::string& table, const std::string& column, mlir::Operation* op) {
    // This function checks if the column is present in the stream code and marks it as touched.
    auto it = std::find_if(streamCode->columns.begin(), streamCode->columns.end(),
                           [&](const ColumnDetail& col) {
                             return col.getColumnName() == column && col.getTableName() == table;
                           });
    if (it != streamCode->columns.end()) {
      auto op = it->getDefiningOp();
      assert(op && "Column defining operation must not be null");

      if (mlir::isa<relalg::InnerJoinOp>(op)) {
        op->setAttr("shouldAllocateShuffleBuffer", mlir::BoolAttr::get(op->getContext(), true));
      }
    }
  }

  template <typename ColumnAttrTy>
  std::string getColumnName(const ColumnAttrTy& colAttr) {
    for (auto n : colAttr.getName().getNestedReferences()) {
        return n.getAttr().str();
    }
    assert(false && "No column for columnattr found");
    return "";
  }

  template <typename ColumnAttrTy>
  static std::string getTableName(const ColumnAttrTy& colAttr) {
    return colAttr.getName().getRootReference().str();
  }

  void walkAggregation(relalg::AggregationOp aggregationOp) {
    aggregationOp->dump();

    // There are two types of column references in aggregation:
    // 1. the group by columns
    // 2. the columns used by the AggrFuncOp inside the aggregation function to compute the actual aggregation
    // We need to find the root of both these columns and mark their join ops as ops that need allocation in shuffle buffer
    auto op = aggregationOp.getRelMutable().get().getDefiningOp();
    assert(op && "Aggregation operation must have a defining operation for the tuple stream");

    auto streamIDIt = streamID.find(op);
    assert(streamIDIt != streamID.end() && "Stream ID for aggregation operation not found");
    uint32_t nextStreamID = streamIDIt->second;
    assert(streamCodeMap.find(nextStreamID) != streamCodeMap.end() && "Stream code for aggregation operation not found");
    auto streamCode = streamCodeMap[nextStreamID];
    assert(streamCode && "Stream code for aggregation operation is null");

    // columns that aggregationOp uses
    mlir::ArrayAttr groupByKeys = aggregationOp.getGroupByCols();
    for (auto i = 0ull; i < groupByKeys.size(); i++) {
      tuples::ColumnRefAttr key = mlir::cast<tuples::ColumnRefAttr>(groupByKeys[i]);
      auto table = getTableName<tuples::ColumnRefAttr>(key); // .getName().getRootReference().str();
      auto column = getColumnName<tuples::ColumnRefAttr>(key);
      _checkForColumnInStream(streamCode, table, column, op);
    }

    // TODO: add column defs to the tuplestreamcode


    streamID[aggregationOp] = nextStreamID - 1; // aggregations just break the pipeline. 

    // TODO: we need to start a new pipeline here?
  }

  void walkScan(relalg::BaseTableOp scanOp) {
    // This function is a placeholder for any specific logic needed for BaseTableOp.
    // Currently, it just dumps the operation.
    scanOp->dump();

    // Assign a new stream ID for the scan operation
    assert(streamID.find(scanOp) == streamID.end() && "Stream ID already assigned");
    _createNewStream(scanOp);
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::Operation* op){
       if (auto innerJoinOp = mlir::dyn_cast<relalg::InnerJoinOp>(op)) {
        walkInnerJoin(innerJoinOp);
      } else if (auto aggregationOp = mlir::dyn_cast<relalg::AggregationOp>(op)) {
        walkAggregation(aggregationOp);
      } else if (auto scanOp = mlir::dyn_cast<relalg::BaseTableOp>(op)) {
        walkScan(scanOp);
      } else {
        streamID[op] = nextStreamID - 1; // use the current stream for other ops
      }
    });
    auto moduleOp = getOperation();
    moduleOp.dump();
  }

private:
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createShuffleAnalysisPass() {
  return std::make_unique<ShuffleAnalysisPass>();
}