
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <chrono>
#include <iomanip>
#include <locale>
#include <sstream>
#include <fmt/core.h>

namespace {
using namespace lingodb::compiler::dialect;
enum class KernelType {
   Main,
   Count
};
enum class ColumnType {
   Direct,
   Mapped
};

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

struct ColumnMetadata {
   int joinOrder;
   std::string loadExpression;
   ColumnType type;
   std::vector<tuples::ColumnRefAttr> dependencies;
};
struct ColumnDetail {
   std::string column;
   std::string table;
   mlir::Type type;
   ColumnDetail(std::string column, std::string table, mlir::Type type) : column(column), table(table), type(type) {}
   ColumnDetail(const tuples::ColumnRefAttr& colAttr) {
      table = getTableName<tuples::ColumnRefAttr>(colAttr);
      column = getColumnName<tuples::ColumnRefAttr>(colAttr);
      type = colAttr.getColumn().type;
   }
   ColumnDetail(const tuples::ColumnDefAttr& colAttr) {
      table = getTableName<tuples::ColumnDefAttr>(colAttr);
      column = getColumnName<tuples::ColumnDefAttr>(colAttr);
      type = colAttr.getColumn().type;
   }
   std::string getMlirSymbol() {
      return table + "__" + column;
   }
};

static std::string mlirTypeToCudaType(mlir::Type ty) {
   if (mlir::isa<db::StringType>(ty))
      return "DBStringType";
   else if (ty.isInteger(32))
      return "DBI32Type";
   else if (ty.isInteger(64))
      return "DBI64Type";
   else if (mlir::isa<db::DecimalType>(ty))
      return "DBDecimalType"; // TODO(avinash, p3): change appropriately to float or double based on decimal type's parameters
   else if (mlir::isa<db::DateType>(ty))
      return "DBDateType";
   else if (mlir::isa<db::CharType>(ty))
      return "DBCharType";
   else if (mlir::isa<db::NullableType>(ty))
      return mlirTypeToCudaType(mlir::dyn_cast_or_null<db::NullableType>(ty).getType());
   ty.dump();
   assert(false && "unhandled type");
   return "";
}


class TupleStreamCode {
   std::vector<std::string> mainCode;
   std::vector<std::string> countCode;

   std::map<std::string, ColumnMetadata> columnData;
   std::set<std::string> loadedColumns;

   std::map<std::string, std::string> mlirToGlobalSymbol; // used when launching the kernel.

   std::map<std::string, std::string> mainArgs;
   std::map<std::string, std::string> countArgs;

   void appendKernel(std::string stmt, KernelType ty) {
      if (ty == KernelType::Main)
         mainCode.push_back(stmt);
      else
         countCode.push_back(stmt);
   }

   public:
   TupleStreamCode() {}
   std::string LoadColumn(const tuples::ColumnRefAttr& attr, KernelType ty) {
      ColumnDetail detail(attr);
      auto mlirSymbol = detail.getMlirSymbol();

      if (columnData.find(mlirSymbol) == columnData.end()) {
         assert(false && "Column ref not in tuple stream");
      }
      auto cudaId = fmt::format("reg_{0}", mlirSymbol);
      if (loadedColumns.find(mlirSymbol) == loadedColumns.end()) {
         loadedColumns.insert(mlirSymbol);
         auto colData = columnData[mlirSymbol];
         if (colData.type == ColumnType::Mapped) {
            for (auto dep : colData.dependencies) {
               LoadColumn(dep, ty);
            }
         }
         appendKernel(fmt::format("auto {1} = {0};", colData.loadExpression, cudaId), ty);
      }
      if (ty == KernelType::Main) {
         mainArgs[mlirSymbol] = mlirTypeToCudaType(detail.type);
      } else {
         countArgs[mlirSymbol] = mlirTypeToCudaType(detail.type);
      }
      return cudaId;
   }
};

class CudaCodeGen : public mlir::PassWrapper<CudaCodeGen, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-cuda-code-gen"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaCodeGen)

   std::map<mlir::Operation*, TupleStreamCode*> streamCodeMap;
   std::vector<TupleStreamCode*> kernelSchedule;

   void runOnOperation() override {
      getOperation().walk([&](mlir::Operation* op) {
         if (auto selection = llvm::dyn_cast<relalg::SelectionOp>(op)) {
            mlir::Operation* stream = selection.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) assert(false && "No downstream operation found for selection.");

            mlir::Region& predicate = selection.getPredicate();
            // std::string condition = translateSelection(predicate, streamCode);

            // streamCode->appendKernel(fmt::format("if (!({0})) return;", condition), KernelType::Main);
            // streamCode->appendKernel(fmt::format("if (!({0})) return;", condition), KernelType::Count);
            streamCodeMap[op] = streamCode;
         } else if (auto joinOp = llvm::dyn_cast<relalg::InnerJoinOp>(op)) {
         } else if (auto aggregationOp = llvm::dyn_cast<relalg::AggregationOp>(op)) {
         } else if (auto scanOp = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
         } else if (auto sortOp = llvm::dyn_cast<relalg::SortOp>(op)) {
         } else if (auto materializeOp = llvm::dyn_cast<relalg::MaterializeOp>(op)) {
         }
      });
   }
};

}

std::unique_ptr<mlir::Pass> relalg::createCudaCodeGenPass() { return std::make_unique<CudaCodeGen>(); }
