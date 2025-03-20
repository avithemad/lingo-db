
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

//TODO(avinash): this function is incorrect (given by chatgpt)
static int daysSinceEpoch(const std::string& dateStr) {
   std::tm t = {};
   std::istringstream ss(dateStr);
   ss >> std::get_time(&t, "%Y-%m-%d"); // Parse the date string
   if (ss.fail()) {
      assert(false && "Could not convert date time string");
   }

   // Convert to time_point
   std::chrono::system_clock::time_point tp =
      std::chrono::system_clock::from_time_t(std::mktime(&t));

   // Calculate days since epoch
   auto epoch = std::chrono::system_clock::from_time_t(0);
   auto duration = std::chrono::duration_cast<std::chrono::hours>(tp - epoch);

   return (duration.count() / 24) + 1; // Convert hours to days
}

static std::string mlirTypeToCudaType(const mlir::Type& ty) {
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
static std::string translateConstantOp(db::ConstantOp& constantOp) {
   std::string result = "";
   auto ty = constantOp.getResult().getType();
   if (mlir::isa<db::DateType>(ty)) {
      auto dateAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constantOp.getValue());
      assert(dateAttr != 0x0 && "Expected date attribute to be a string attribute");

      return std::to_string(daysSinceEpoch(dateAttr.str()));
   } else if (mlir::isa<db::DecimalType>(ty)) {
      auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(constantOp.getValue());
      auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constantOp.getValue());
      auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constantOp.getValue());

      assert((strAttr != 0x0 || intAttr != 0x0 || floatAttr != 0x0) && "Expected Decimal constant attribute to be floatattr or integer attr");

      if (intAttr) return std::to_string(intAttr.getInt());
      if (floatAttr) return std::to_string(floatAttr.getValueAsDouble());
      return strAttr.str();
   } else if (mlir::isa<db::StringType>(ty)) {
      auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constantOp.getValue());
      assert(strAttr != 0x0 && "Expected string constant attribute to be strattr");

      return ("\"" + strAttr.str() + "\"");
   } else if (ty.isInteger(32) || ty.isInteger(64)) {
      auto integerAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constantOp.getValue());
      assert(integerAttr != 0x0 && "Expected integer constant attribute to be i64 or i32");

      return std::to_string(integerAttr.getInt());
   } else if (mlir::isa<db::CharType>(ty)) {
      // TODO(avinash): handle char types
      auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constantOp.getValue());
      assert(strAttr != 0x0 && "Expected character constant attribute to be strattr");

      return ("\"" + strAttr.str() + "\"");
   } else {
      assert(false && "Constant op not handled");
      return "";
   }
}

class TupleStreamCode {
   std::vector<std::string> mainCode;
   std::vector<std::string> countCode;
   std::vector<std::string> controlCode;

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
         mainArgs[mlirSymbol] = mlirTypeToCudaType(detail.type) + "*"; // columns are always a 1d array
      } else {
         countArgs[mlirSymbol] = mlirTypeToCudaType(detail.type) + "*";
      }
      return cudaId;
   }
   std::string selectionOpDfs(mlir::Operation* op) {
      if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
         LoadColumn(getColOp.getAttr(), KernelType::Count);
         return LoadColumn(getColOp.getAttr(), KernelType::Main);
      } else if (auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(op)) {
         return translateConstantOp(constOp);
      } else if (auto compareOp = mlir::dyn_cast_or_null<db::CmpOp>(op)) {
         auto left = compareOp.getLeft();
         std::string leftOperand = selectionOpDfs(left.getDefiningOp());

         auto right = compareOp.getRight();
         std::string rightOperand = selectionOpDfs(right.getDefiningOp());

         auto cmp = compareOp.getPredicate();
         std::string predicate = "";
         switch (cmp) {
            case db::DBCmpPredicate::eq:
               predicate = "Predicate::eq";
               break;
            case db::DBCmpPredicate::neq:
               predicate = "Predicate::neq";
               break;
            case db::DBCmpPredicate::lt:
               predicate = "Predicate::lt";
               break;
            case db::DBCmpPredicate::gt:
               predicate = "Predicate::gt";
               break;
            case db::DBCmpPredicate::lte:
               predicate = "Predicate::lte";
               break;
            case db::DBCmpPredicate::gte:
               predicate = "Predicate::gte";
               break;
            default:
               assert(false && "Predicate not handled");
               break;
         }
         return fmt::format("evaluatePredicate({0}, {1}, {2})", leftOperand, rightOperand, predicate);
      } else if (auto runtimeOp = mlir::dyn_cast_or_null<db::RuntimeCall>(op)) { // or a like operation
         // TODO(avinash, p1): handle runtime predicate like operator
         assert(false && "TODO: handle runtime predicates\n");
      } else if (auto betweenOp = mlir::dyn_cast_or_null<db::BetweenOp>(op)) {
         std::string operand = selectionOpDfs(betweenOp.getVal().getDefiningOp());
         std::string lower = selectionOpDfs(betweenOp.getLower().getDefiningOp());
         std::string upper = selectionOpDfs(betweenOp.getUpper().getDefiningOp());

         std::string lpred = betweenOp.getLowerInclusive() ? "Predicate::gte" : "Predicate::gt";
         std::string rpred = betweenOp.getUpperInclusive() ? "Predicate::lte" : "Predicate::lt";
         return fmt::format("evaluatePredicate({0}, {1}, {2}) && evaluatePredicate({0}, {3}, {4})",
                            operand, lower, lpred, upper, rpred);
      } else if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(op)) {
         std::string res = "(";
         std::string sep = "";
         for (auto v : andOp.getVals()) {
            res += sep + selectionOpDfs(v.getDefiningOp());
            sep = " && (";
            res += ")";
         }
         return res;
         // return "true";
      } else if (auto orOp = mlir::dyn_cast_or_null<db::OrOp>(op)) {
         std::string res = "(";
         std::string sep = "";
         for (auto v : orOp.getVals()) {
            res += sep + selectionOpDfs(v.getDefiningOp());
            sep = " || (";
            res += ")";
         }
         return res;
      }
      op->dump();
      assert(false && "Selection predicate not handled");
      return "";
   }
   void addSelectionPredicate(mlir::Region& predicate) {
      auto terminator = mlir::cast<tuples::ReturnOp>(predicate.front().getTerminator());
      if (!terminator.getResults().empty()) {
         auto& predicateBlock = predicate.front();
         if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predicateBlock.getTerminator())) {
            mlir::Value matched = returnOp.getResults()[0];
            std::string condition = selectionOpDfs(matched.getDefiningOp());
            appendKernel(fmt::format("if (!({0})) return;", condition), KernelType::Count);
            appendKernel(fmt::format("if (!({0})) return;", condition), KernelType::Main);
            return;
         } else {
            assert(false && "expected return op to be in the end of the predicate region");
         }
      }
      predicate.front().dump();
      assert(false && "Predicate is not implemented");
      return;
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
            streamCode->addSelectionPredicate(predicate);
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
