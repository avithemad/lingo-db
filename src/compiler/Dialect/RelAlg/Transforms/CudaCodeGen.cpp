
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"

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

void emitControlFunctionSignature(std::ostream& outputFile);

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

template <typename ObjType>
class IdGenerator {
   std::map<ObjType, std::string> m_objectIds;
   int32_t m_id = 0;

   public:
   IdGenerator() {}

   std::string getId(ObjType obj) {
      if (m_objectIds.find(obj) == m_objectIds.end()) {
         m_objectIds[obj] = std::to_string(m_id);
         m_id++;
      }
      return m_objectIds[obj];
   }
};

std::string GetId(const void* op) {
   static IdGenerator<const void*> idGen;
   std::string result = idGen.getId(op);
   return result;
}

static std::string HT(const void* op) {
   return "HT_" + GetId(op);
}
static std::string KEY(const void* op) {
   return "KEY_" + GetId(op);
}
static std::string SLOT(const void* op) {
   return "SLOT_" + GetId(op);
}
static std::string BUF(const void* op) {
   return "BUF_" + GetId(op);
}
static std::string BUF_IDX(const void* op) {
   return "BUF_IDX_" + GetId(op);
}
static std::string buf_idx(const void* op) {
   return "buf_idx_" + GetId(op);
}
static std::string COUNT(const void* op) {
   return "COUNT" + GetId(op);
}
static std::string MAT(const void* op) {
   return "MAT" + GetId(op);
}
static std::string MAT_IDX(const void* op) {
   return "MAT_IDX" + GetId(op);
}
static std::string mat_idx(const void* op) {
   return "mat_idx" + GetId(op);
}
static std::string slot_first(const void* op) {
   return "slot_first" + GetId(op);
}
static std::string slot_second(const void* op) {
   return "slot_second" + GetId(op);
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
struct ColumnMetadata {
   std::string loadExpression;
   ColumnType type;
   std::string rid; // valid if type is mapped
   int streamId;
   std::vector<tuples::ColumnRefAttr> dependencies; // valid if type is Mapped
   std::string globalId;
   ColumnMetadata(const std::string& le, ColumnType ty, int streamId, const std::string& globalId) : loadExpression(le), type(ty), streamId(streamId), globalId(globalId) {}
   ColumnMetadata(const std::string& le, ColumnType ty, int streamId, const std::vector<tuples::ColumnRefAttr>& dep)
      : loadExpression(le), type(ty), streamId(streamId), dependencies(dep) {}
   ColumnMetadata(ColumnMetadata* metadata) : loadExpression(metadata->loadExpression),
                                              type(metadata->type),
                                              rid(metadata->rid),
                                              streamId(metadata->streamId),
                                              dependencies(metadata->dependencies),
                                              globalId(metadata->globalId) {}
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
   else if (mlir::isa<db::CharType>(ty)) {
      auto charTy = mlir::dyn_cast_or_null<db::CharType>(ty);
      if (charTy.getBytes() > 1) return "DBStringType";
      return "DBCharType";
   } else if (mlir::isa<db::NullableType>(ty))
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

      if (intAttr) return std::to_string(intAttr.getInt()) + ".0";
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
      if (strAttr.str().size() == 1) {
         return ("\'" + strAttr.str() + "\'");
      } else {
         return ("\"" + strAttr.str() + "\"");
      }
   } else {
      assert(false && "Constant op not handled");
      return "";
   }
}
static int StreamId = 0;

class TupleStreamCode {
   std::vector<std::string> mainCode;
   std::vector<std::string> countCode;
   std::vector<std::string> controlCode;
   int forEachScopes = 0;
   std::map<std::string, ColumnMetadata*> columnData;
   std::set<std::string> loadedColumns;
   std::set<std::string> loadedCountColumns;
   std::set<std::string> deviceFrees;
   std::set<std::string> hostFrees;

   std::map<std::string, std::string> mlirToGlobalSymbol; // used when launching the kernel.

   std::map<std::string, std::string> mainArgs;
   std::map<std::string, std::string> countArgs;
   int id;
   void appendKernel(std::string stmt, KernelType ty) {
      if (ty == KernelType::Main)
         mainCode.push_back(stmt);
      else
         countCode.push_back(stmt);
   }

   void appendControl(std::string stmt) {
      controlCode.push_back(stmt);
   }

   std::string launchKernel(KernelType ty) {
      std::string _kernelName;
      std::map<std::string, std::string> _args;
      if (ty == KernelType::Main) {
         _kernelName = "main";
         _args = mainArgs;
      } else {
         _kernelName = "count";
         _args = countArgs;
      }
      std::string size = "";
      for (auto p : _args)
         if (p.second == "size_t") size = p.first;
      if (size == "") assert(false && "No size argument for this kernel");
      std::string args = "", sep = "";
      for (auto p : _args) {
         args += fmt::format("{1}{0}", mlirToGlobalSymbol[p.first], sep);
         sep = ", ";
      }
      return fmt::format("{0}_{1}<<<std::ceil((float){2}/128.), 128>>>({3});", _kernelName, GetId((void*) this), size, args);
   }

   public:
   TupleStreamCode(relalg::BaseTableOp& baseTableOp) {
      std::string tableName = baseTableOp.getTableIdentifier().data();
      std::string tableSize = tableName + "_size";
      mlirToGlobalSymbol[tableSize] = tableSize;
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only

      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;", KernelType::Main);
      appendKernel(fmt::format("if (tid >= {}) return;", tableSize), KernelType::Main);
      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;", KernelType::Count);
      appendKernel(fmt::format("if (tid >= {}) return;", tableSize), KernelType::Count);
      for (auto namedAttr : baseTableOp.getColumns().getValue()) {
         auto columnName = namedAttr.getName().str();
         ColumnDetail detail(mlir::cast<tuples::ColumnDefAttr>(namedAttr.getValue()));
         auto globalSymbol = fmt::format("d_{0}__{1}", tableName, columnName);
         auto mlirSymbol = detail.getMlirSymbol();
         mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
         ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
         metadata->rid = "tid";
         columnData[mlirSymbol] = metadata;

         if (mlirTypeToCudaType(detail.type) == "DBStringType") {
            ColumnMetadata* encoded_metadata = new ColumnMetadata(mlirSymbol + "_encoded", ColumnType::Direct, StreamId, globalSymbol + "_encoded");
            encoded_metadata->rid = "tid";
            columnData[mlirSymbol + "_encoded"] = encoded_metadata;
            mlirToGlobalSymbol[mlirSymbol + "_encoded"] = globalSymbol + "_encoded";
         }
      }
      id = StreamId;
      StreamId++;
      return;
   }
   TupleStreamCode(mlir::Operation* op) {
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "Expected aggregation operation");
      std::string tableSize = COUNT(op);

      mlirToGlobalSymbol[tableSize] = tableSize;
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only

      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;", KernelType::Main);
      appendKernel(fmt::format("if (tid >= {0}) return;", tableSize), KernelType::Main);
      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;", KernelType::Count);
      appendKernel(fmt::format("if (tid >= {0}) return;", tableSize), KernelType::Count);

      auto groupByKeys = aggOp.getGroupByCols();
      auto computedCols = aggOp.getComputedCols();
      for (auto& col : groupByKeys) {
         ColumnDetail detail(mlir::cast<tuples::ColumnRefAttr>(col));
         if (mlirTypeToCudaType(detail.type) == "DBStringType") {
            auto mlirSymbol = detail.getMlirSymbol() + "_encoded";
            auto globalSymbol = fmt::format("d_{0}", KEY(op) + mlirSymbol);
            mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
            ColumnMetadata* encoded_metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
            encoded_metadata->rid = "tid";
            columnData[mlirSymbol] = encoded_metadata;
         } else {
            auto mlirSymbol = detail.getMlirSymbol();
            auto globalSymbol = fmt::format("d_{0}", KEY(op) + mlirSymbol);
            mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
            ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
            metadata->rid = "tid";
            columnData[mlirSymbol] = metadata;
         }
      }
      for (auto& col : computedCols) {
         ColumnDetail detail(mlir::cast<tuples::ColumnDefAttr>(col));
         if (mlirTypeToCudaType(detail.type) == "DBStringType") {
            auto mlirSymbol = detail.getMlirSymbol() + "_encoded";
            auto globalSymbol = fmt::format("d_{0}", mlirSymbol);
            mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
            ColumnMetadata* encoded_metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
            encoded_metadata->rid = "tid";
            columnData[mlirSymbol] = encoded_metadata;
         } else {
            auto mlirSymbol = detail.getMlirSymbol();
            auto globalSymbol = fmt::format("d_{0}", mlirSymbol);
            mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
            ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
            metadata->rid = "tid";
            columnData[mlirSymbol] = metadata;
         }
      }
      id = StreamId;
      StreamId++;
      return;
   }
   ~TupleStreamCode() {
      for (auto p : columnData) delete p.second;
   }
   void RenamingOp(relalg::RenamingOp renamingOp) {
      // renamingOp.dump();
      for (mlir::Attribute attr : renamingOp.getColumns()) {
         // attr.dump();
         auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         mlir::Attribute from = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[0];
         // from.dump();
         auto relationRefAttr = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(from);
         ColumnDetail detailRef(relationRefAttr), detailDef(relationDefAttr);
         auto colData = columnData[detailRef.getMlirSymbol()];
         std::string ty = mlirTypeToCudaType(detailRef.type);
         if (colData == nullptr && mlirTypeToCudaType(detailRef.type) == "DBStringType") {
            colData = columnData[detailRef.getMlirSymbol() + "_encoded"];
            ty = "DBI6Type";
         }
         if (colData == nullptr) {
            assert(false && "Renaming op: column ref not in tuple stream");
         }
         mainArgs[detailRef.getMlirSymbol()] = ty + "*";
         countArgs[detailRef.getMlirSymbol()] = ty + "*";
         columnData[detailDef.getMlirSymbol()] =
            new ColumnMetadata(colData);
      }
   }
   template <int enc = 0>
   std::string LoadColumn(const tuples::ColumnRefAttr& attr, KernelType ty) {
      ColumnDetail detail(attr);
      if (enc != 0) detail.column += "_encoded"; // use for string encoded columns
      auto mlirSymbol = detail.getMlirSymbol();

      if (columnData.find(mlirSymbol) == columnData.end()) {
         std::clog << mlirSymbol << std::endl;
         assert(false && "Column ref not in tuple stream");
      }
      auto cudaId = fmt::format("reg_{0}", mlirSymbol);
      if (ty == KernelType::Main && loadedColumns.find(mlirSymbol) != loadedColumns.end()) {
         return cudaId;
      } else if (ty == KernelType::Count && loadedCountColumns.find(mlirSymbol) != loadedCountColumns.end()) {
         return cudaId;
      }
      if (ty == KernelType::Main)
         loadedColumns.insert(mlirSymbol);
      else
         loadedCountColumns.insert(mlirSymbol);
      auto colData = columnData[mlirSymbol];
      if (colData->type == ColumnType::Mapped) {
         for (auto dep : colData->dependencies) {
            LoadColumn(dep, ty);
         }
      }
      appendKernel(fmt::format("auto {1} = {0};", colData->loadExpression + (colData->type == ColumnType::Direct ? "[" + colData->rid + "]" : ""), cudaId), ty);
      if (colData->type == ColumnType::Direct) {
         auto cudaTy = mlirTypeToCudaType(detail.type);
         if (ty == KernelType::Main) {
            auto cudaTy = mlirTypeToCudaType(detail.type);
            if (enc == 0)
               mainArgs[mlirSymbol] = cudaTy + "*"; // columns are always a 1d array
            else
               mainArgs[mlirSymbol] = "DBI16Type*";
         } else {
            if (enc == 0)
               countArgs[mlirSymbol] = cudaTy + "*"; // columns are always a 1d array
            else
               countArgs[mlirSymbol] = "DBI16Type*";
         }
      }
      return cudaId;
   }
   std::string SelectionOpDfs(mlir::Operation* op) {
      if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
         LoadColumn(getColOp.getAttr(), KernelType::Count);
         return LoadColumn(getColOp.getAttr(), KernelType::Main);
      } else if (auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(op)) {
         return translateConstantOp(constOp);
      } else if (auto compareOp = mlir::dyn_cast_or_null<db::CmpOp>(op)) {
         auto left = compareOp.getLeft();
         std::string leftOperand = SelectionOpDfs(left.getDefiningOp());

         auto right = compareOp.getRight();
         std::string rightOperand = SelectionOpDfs(right.getDefiningOp());

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
         // assert(false && "TODO: handle runtime predicates\n");
         std::string function = runtimeOp.getFn().str();
         std::string args = "";
         std::string sep = "";
         for (auto v : runtimeOp.getArgs()) {
            args += sep + SelectionOpDfs(v.getDefiningOp());
            sep = ", ";
         }
         return fmt::format("{0}({1})", function, args);
      } else if (auto betweenOp = mlir::dyn_cast_or_null<db::BetweenOp>(op)) {
         std::string operand = SelectionOpDfs(betweenOp.getVal().getDefiningOp());
         std::string lower = SelectionOpDfs(betweenOp.getLower().getDefiningOp());
         std::string upper = SelectionOpDfs(betweenOp.getUpper().getDefiningOp());

         std::string lpred = betweenOp.getLowerInclusive() ? "Predicate::gte" : "Predicate::gt";
         std::string rpred = betweenOp.getUpperInclusive() ? "Predicate::lte" : "Predicate::lt";
         return fmt::format("evaluatePredicate({0}, {1}, {2}) && evaluatePredicate({0}, {3}, {4})",
                            operand, lower, lpred, upper, rpred);
      } else if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(op)) {
         std::string res = "(";
         std::string sep = "";
         for (auto v : andOp.getVals()) {
            res += sep + SelectionOpDfs(v.getDefiningOp());
            sep = " && (";
            res += ")";
         }
         return res;
         // return "true";
      } else if (auto orOp = mlir::dyn_cast_or_null<db::OrOp>(op)) {
         std::string res = "(";
         std::string sep = "";
         for (auto v : orOp.getVals()) {
            res += sep + SelectionOpDfs(v.getDefiningOp());
            sep = " || (";
            res += ")";
         }
         return res;
      } else if (auto constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(op)) {
         if (auto c = mlir::cast<mlir::BoolAttr>(constantOp.getValue())) {
            bool value = c.getValue();
            return value ? "true" : "false";
         }
      } else if (auto notOp = mlir::dyn_cast_or_null<db::NotOp>(op)) {
         std::string res = "!(";
         mlir::Operation* arg = notOp.getVal().getDefiningOp();
         res += SelectionOpDfs(arg);
         res += ")";
         return res;
      } else if (auto isNullOp = mlir::dyn_cast_or_null<db::IsNullOp>(op)) {
         // TODO(avinash):
         // nullable datatypes not handled for now
         return "false";
      } else if (auto asNullableOp = mlir::dyn_cast_or_null<db::AsNullableOp>(op)) {
         return "true";
      } else if (auto oneOfOp = mlir::dyn_cast_or_null<db::OneOfOp>(op)) {
         std::string res = "(";
         std::string sep = "";
         std::string val = SelectionOpDfs(oneOfOp.getVal().getDefiningOp());
         for (auto v : oneOfOp.getVals()) {
            res += sep + fmt::format("evaluatePredicate({0}, {1}, Predicate::eq)", val, SelectionOpDfs(v.getDefiningOp()));
            sep = " || (";
            res += ")";
         }
         return res;
      }
      op->dump();
      assert(false && "Selection predicate not handled");
      return "";
   }
   void AddSelectionPredicate(mlir::Region& predicate) {
      auto terminator = mlir::cast<tuples::ReturnOp>(predicate.front().getTerminator());
      if (!terminator.getResults().empty()) {
         auto& predicateBlock = predicate.front();
         if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predicateBlock.getTerminator())) {
            mlir::Value matched = returnOp.getResults()[0];
            std::string condition = SelectionOpDfs(matched.getDefiningOp());
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
   void MaterializeCount(mlir::Operation* op) {
      countArgs[COUNT(op)] = "uint64_t*";
      mlirToGlobalSymbol[COUNT(op)] = fmt::format("d_{}", COUNT(op));
      appendKernel("//Materialize count", KernelType::Count);
      appendKernel(fmt::format("atomicAdd((int*){0}, 1);", COUNT(op)), KernelType::Count);

      appendControl("//Materialize count");
      appendControl(fmt::format("uint64_t* d_{0};", COUNT(op)));
      appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(uint64_t));", COUNT(op)));
      deviceFrees.insert(fmt::format("d_{0}", COUNT(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", COUNT(op)));
      appendControl(launchKernel(KernelType::Count));
      appendControl(fmt::format("uint64_t {0};", COUNT(op)));
      appendControl(fmt::format("cudaMemcpy(&{0}, d_{0}, sizeof(uint64_t), cudaMemcpyDeviceToHost);", COUNT(op)));
      // appendControl(fmt::format("cudaFree(d_{0});", COUNT(op)));
   }
   std::string MakeKeys(mlir::Operation* op, const mlir::ArrayAttr& keys, KernelType kernelType) {
      //TODO(avinash, p3): figure a way out for double keys
      appendKernel(fmt::format("uint64_t {0} = 0;", KEY(op)), kernelType);
      std::map<std::string, int> allowedKeysToSize;
      allowedKeysToSize["DBCharType"] = 1;
      allowedKeysToSize["DBStringType"] = 2;
      allowedKeysToSize["DBI32Type"] = 4;
      allowedKeysToSize["DBDateType"] = 4;
      allowedKeysToSize["DBI64Type"] = 4; // TODO(avinash): This is a temporary fix for date grouping.
      std::string sep = "";
      int totalKeySize = 0;
      for (auto i = 0ull; i < keys.size(); i++) {
         tuples::ColumnRefAttr key = mlir::cast<tuples::ColumnRefAttr>(keys[i]);
         auto baseType = mlirTypeToCudaType(key.getColumn().type);
         // handle string type differently (assume that string encoded column is available)
         if (allowedKeysToSize.find(baseType) == allowedKeysToSize.end()) {
            keys.dump();
            assert(false && "Type is not hashable");
         }
         std::string cudaIdentifierKey;
         if (baseType == "DBStringType") {
            cudaIdentifierKey = LoadColumn<1>(key, kernelType);
         } else {
            cudaIdentifierKey = LoadColumn(key, kernelType);
         }
         appendKernel(sep, kernelType);
         if (i < keys.size() - 1) {
            tuples::ColumnRefAttr next_key = mlir::cast<tuples::ColumnRefAttr>(keys[i + 1]);
            auto next_base_type = mlirTypeToCudaType(next_key.getColumn().type);

            sep = fmt::format("{0} <<= {1};", KEY(op), std::to_string(allowedKeysToSize[next_base_type] * 8));
         } else {
            sep = "";
         }
         if (baseType == "DBI64Type") {
            appendKernel(fmt::format("{0} |= (DBI32Type){1};", KEY(op), cudaIdentifierKey), kernelType);
         } else {
            appendKernel(fmt::format("{0} |= {1};", KEY(op), cudaIdentifierKey), kernelType);
         }
         totalKeySize += allowedKeysToSize[baseType];
         if (totalKeySize > 8) {
            std::clog << totalKeySize << std::endl;
            keys.dump();
            assert(false && "Total hash key exceeded 8 bytes");
         }
      }
      return KEY(op);
   }

   std::vector<std::pair<int, std::string>> getBaseRelations(const std::map<std::string, ColumnMetadata*>& columnData) {
      std::set<std::pair<int, std::string>> temp;
      for (auto p : columnData) {
         if (p.second == nullptr) continue;
         auto metadata = p.second;
         if (metadata->type == ColumnType::Direct)
            temp.insert(std::make_pair(metadata->streamId, metadata->rid));
      }
      std::vector<std::pair<int, std::string>> baseRelations(temp.begin(), temp.end());
      std::sort(baseRelations.begin(), baseRelations.end());
      return baseRelations;
   }
   void BuildHashTableSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::SemiJoinOp>(op);
      if (!joinOp) assert(false && "Build hash table accepts only semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Insert hash table kernel;", KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}, 1}});", HT(op), key), KernelType::Main);

      mainArgs[HT(op)] = "HASHTABLE_INSERT_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      appendControl("// Insert hash table control;");
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{(int64_t)-1}},cuco::empty_value{{(int64_t)-1}},thrust::equal_to<int64_t>{{}},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>() }};",
                                HT(op), COUNT(op)));
      appendControl(launchKernel(KernelType::Main));
   }
   void BuildHashTableAntiSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::AntiSemiJoinOp>(op);
      if (!joinOp) assert(false && "Build hash table accepts only anti semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Insert hash table kernel;", KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}, 1}});", HT(op), key), KernelType::Main);

      mainArgs[HT(op)] = "HASHTABLE_INSERT_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      appendControl("// Insert hash table control;");
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{(int64_t)-1}},cuco::empty_value{{(int64_t)-1}},thrust::equal_to<int64_t>{{}},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>() }};",
                                HT(op), COUNT(op)));
      appendControl(launchKernel(KernelType::Main));
   }
   void ProbeHashTableSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::SemiJoinOp>(op);
      if (!joinOp) assert(false && "Probe hash table accepts only semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      MakeKeys(op, keys, KernelType::Count);
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("//Probe Hash table", KernelType::Main);
      appendKernel("//Probe Hash table", KernelType::Count);
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Main);
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Count);
      appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)), KernelType::Main);
      appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)), KernelType::Count);

      mainArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      countArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
   }
   void ProbeHashTableAntiSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::AntiSemiJoinOp>(op);
      if (!joinOp) assert(false && "Probe hash table accepts only anti semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      MakeKeys(op, keys, KernelType::Count);
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("//Probe Hash table", KernelType::Main);
      appendKernel("//Probe Hash table", KernelType::Count);
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Main);
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Count);
      appendKernel(fmt::format("if (!({0} == {1}.end())) return;", SLOT(op), HT(op)), KernelType::Main);
      appendKernel(fmt::format("if (!({0} == {1}.end())) return;", SLOT(op), HT(op)), KernelType::Count);

      mainArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      countArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
   }
   std::map<std::string, ColumnMetadata*> BuildHashTable(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Insert hash table accepts only inner join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Insert hash table kernel;", KernelType::Main);
      appendKernel(fmt::format("auto {0} = atomicAdd((int*){1}, 1);", buf_idx(op), BUF_IDX(op)), KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}, {2}}});", HT(op), key, buf_idx(op)), KernelType::Main);
      auto baseRelations = getBaseRelations(columnData);
      int i = 0;
      for (auto br : baseRelations) {
         appendKernel(fmt::format("{0}[{1} * {2} + {3}] = {4};",
                                  BUF(op),
                                  buf_idx(op),
                                  std::to_string(baseRelations.size()),
                                  i++,
                                  br.second),
                      KernelType::Main);
      }

      mainArgs[BUF_IDX(op)] = "uint64_t*";
      mainArgs[HT(op)] = "HASHTABLE_INSERT";
      mainArgs[BUF(op)] = "uint64_t*";
      mlirToGlobalSymbol[BUF_IDX(op)] = fmt::format("d_{}", BUF_IDX(op));
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      mlirToGlobalSymbol[BUF(op)] = fmt::format("d_{}", BUF(op));
      appendControl("// Insert hash table control;");
      appendControl(fmt::format("uint64_t* d_{0};", BUF_IDX(op)));
      appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(uint64_t));", BUF_IDX(op)));
      deviceFrees.insert(fmt::format("d_{0}", BUF_IDX(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", BUF_IDX(op)));
      appendControl(fmt::format("uint64_t* d_{0};", BUF(op)));
      appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(uint64_t) * {1} * {2});", BUF(op), COUNT(op), baseRelations.size()));
      deviceFrees.insert(fmt::format("d_{0}", BUF(op)));
      appendControl(fmt::format("auto d_{0} = cuco::experimental::static_multimap{{ (int){1}*2, cuco::empty_key{{(int64_t)-1}},cuco::empty_value{{(int64_t)-1}},thrust::equal_to<int64_t>{{}},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>() }};",
                                HT(op), COUNT(op)));
      // appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{(int64_t)-1}},cuco::empty_value{{(int64_t)-1}},thrust::equal_to<int64_t>{{}},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>() }};",
      //                           HT(op), COUNT(op)));
      appendControl(launchKernel(KernelType::Main));
      // appendControl(fmt::format("cudaFree(d_{0});", BUF_IDX(op)));
      return columnData;
   }

   void ProbeHashTable(mlir::Operation* op, const std::map<std::string, ColumnMetadata*>& leftColumnData) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Probe hash table accepts only inner join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      MakeKeys(op, keys, KernelType::Count);
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("//Probe Hash table", KernelType::Main);
      appendKernel("//Probe Hash table", KernelType::Count);
      // appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Main);
      // appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Count);
      // appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)), KernelType::Main);
      // appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)), KernelType::Count);
      appendKernel(fmt::format("{0}.for_each({1}, [&] __device__ (auto const {2}) {{", HT(op), key, SLOT(op)), KernelType::Main);
      appendKernel(fmt::format("auto const [{0}, {1}] = {2};", slot_first(op), slot_second(op), SLOT(op)), KernelType::Main);
      appendKernel(fmt::format("{0}.for_each({1}, [&] __device__ (auto const {2}) {{\n", HT(op), key, SLOT(op)), KernelType::Count);
      appendKernel(fmt::format("auto const [{0}, {1}] = {2};", slot_first(op), slot_second(op), SLOT(op)), KernelType::Count);
      forEachScopes++;

      // add all leftColumn data to this data
      auto baseRelations = getBaseRelations(leftColumnData);
      std::map<int, int> streamIdToBufId;
      int i = 0;
      // TODO(avinash): Check this logic, there is a bug in this refer q9.
      for (auto br : baseRelations) {
         streamIdToBufId[br.first] = i;
         i++;
      }
      for (auto colData : leftColumnData) {
         if (colData.second == nullptr) continue;
         if (colData.second->type == ColumnType::Direct) {
            // colData.second->rid = fmt::format("{3}[{0}->second * {1} + {2}]",
            //                                   SLOT(op),
            //                                   std::to_string(baseRelations.size()),
            //                                   streamIdToBufId[colData.second->streamId],
            //                                   BUF(op));
            colData.second->rid = fmt::format("{3}[{0} * {1} + {2}]",
                                              slot_second(op),
                                              std::to_string(baseRelations.size()),
                                              streamIdToBufId[colData.second->streamId],
                                              BUF(op));
            // colData.second->streamId = id;
            columnData[colData.first] = colData.second;
            mlirToGlobalSymbol[colData.second->loadExpression] = colData.second->globalId;
         }
         columnData[colData.first] = colData.second;
      }
      mainArgs[HT(op)] = "HASHTABLE_PROBE";
      mainArgs[BUF(op)] = "uint64_t*";
      countArgs[HT(op)] = "HASHTABLE_PROBE";
      countArgs[BUF(op)] = "uint64_t*";
      // mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::for_each)", HT(op));
      mlirToGlobalSymbol[BUF(op)] = fmt::format("d_{}", BUF(op));
   }
   void CreateAggregationHashTable(mlir::Operation* op) {
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "CreateAggregationHashTable expects aggregation op as a parameter!");
      mlir::ArrayAttr groupByKeys = aggOp.getGroupByCols();
      auto key = MakeKeys(op, groupByKeys, KernelType::Count);
      appendKernel("//Create aggregation hash table", KernelType::Count);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}, 1}});", HT(op), key), KernelType::Count);
      countArgs[HT(op)] = "HASHTABLE_INSERT";

      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      std::string ht_size = "0";
      // TODO(avinash, p2): this is a hacky way, actually check if --use-db flag is enabled and query optimization is performed
      if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(op->getAttr("rows"))) {
         if (std::floor(floatAttr.getValueAsDouble()) != 0)
            ht_size = std::to_string((size_t) std::ceil(floatAttr.getValueAsDouble()));
         else {
            for (auto p : countArgs) {
               if (p.second == "size_t")
                  ht_size = p.first;
            }
         }
      }
      assert(ht_size != "0" && "hash table for aggregation is sizing to be 0!!");
      appendControl("//Create aggregation hash table");
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{(int64_t)-1}},\
cuco::empty_value{{(int64_t)-1}},\
thrust::equal_to<int64_t>{{}},\
cuco::linear_probing<1, cuco::default_hash_function<int64_t>>() }};",
                                HT(op), ht_size));
      appendControl(launchKernel(KernelType::Count));
      appendControl(fmt::format("size_t {0} = d_{1}.size();", COUNT(op), HT(op)));
      // TODO(avinash): deallocate the old hash table and create a new one to save space in gpu when estimations are way off
      appendControl(fmt::format("thrust::device_vector<int64_t> keys_{0}({2}), vals_{0}({2});\n\
d_{1}.retrieve_all(keys_{0}.begin(), vals_{0}.begin());\n\
d_{1}.clear();\n\
int64_t* raw_keys{0} = thrust::raw_pointer_cast(keys_{0}.data());\n\
insertKeys<<<std::ceil((float){2}/128.), 128>>>(raw_keys{0}, d_{1}.ref(cuco::insert), {2});",
                                GetId(op), HT(op), COUNT(op)));
   }
   void AggregateInHashTable(mlir::Operation* op) {
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "CreateAggregationHashTable expects aggregation op as a parameter!");
      mlir::ArrayAttr groupByKeys = aggOp.getGroupByCols();
      auto key = MakeKeys(op, groupByKeys, KernelType::Main);
      mainArgs[HT(op)] = "HASHTABLE_FIND";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
      appendKernel("//Aggregate in hashtable", KernelType::Main);
      appendKernel(fmt::format("auto {0} = {1}.find({2})->second;", buf_idx(op), HT(op), key), KernelType::Main);
      auto& aggRgn = aggOp.getAggrFunc();
      mlir::ArrayAttr computedCols = aggOp.getComputedCols(); // these are columndefs
      appendControl("//Aggregate in hashtable");
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(aggRgn.front().getTerminator())) {
         int i = 0;
         for (mlir::Value col : returnOp.getResults()) {
            // map each aggrfunc which is col.getDefiningOp to computedColName
            auto newcol = mlir::cast<tuples::ColumnDefAttr>(computedCols[i]);
            ColumnDetail detail(newcol);
            auto newbuffername = detail.getMlirSymbol();
            auto bufferColType = mlirTypeToCudaType(detail.type);
            if (bufferColType == "DBStringType") {
               newbuffername = newbuffername + "_encoded";
               bufferColType = "DBI16Type";
            }
            mainArgs[newbuffername] = bufferColType + "*";
            mlirToGlobalSymbol[newbuffername] = fmt::format("d_{}", newbuffername);
            appendControl(fmt::format("{0}* d_{1};", bufferColType, newbuffername));
            appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof({1}) * {2});", newbuffername, bufferColType, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", newbuffername));
            appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof({1}) * {2});", newbuffername, bufferColType, COUNT(op)));
            if (auto aggrFunc = llvm::dyn_cast<relalg::AggrFuncOp>(col.getDefiningOp())) {
               auto slot = fmt::format("{0}[{1}]", newbuffername, buf_idx(op));
               auto fn = aggrFunc.getFn();
               ColumnDetail aggrCol = ColumnDetail(mlir::cast<tuples::ColumnRefAttr>(aggrFunc.getAttr()));
               std::string val = "";
               if (mlirTypeToCudaType(detail.type) == "DBStringType") {
                  appendControl(fmt::format("auto {0}_map = {1}_map;", detail.getMlirSymbol(), aggrCol.getMlirSymbol()));
                  val = LoadColumn<1>(mlir::cast<tuples::ColumnRefAttr>(aggrFunc.getAttr()), KernelType::Main);
               } else {
                  val = LoadColumn(mlir::cast<tuples::ColumnRefAttr>(aggrFunc.getAttr()), KernelType::Main);
               }
               switch (fn) {
                  case relalg::AggrFunc::sum: {
                     appendKernel(fmt::format("aggregate_sum(&{0}, {1});", slot, val), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::count: {
                     appendKernel(fmt::format("aggregate_sum(&{0}, 1);", slot), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::any: {
                     appendKernel(fmt::format("aggregate_any(&{0}, {1});", slot, val), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::avg: {
                     assert(false && "average should be split into sum and divide");
                  } break;
                  case relalg::AggrFunc::min: {
                     appendKernel(fmt::format("aggregate_min(&{0}, {1});", slot, val), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::max: {
                     appendKernel(fmt::format("aggregate_max(&{0}, {1});", slot, val), KernelType::Main);
                  } break;
                  default:
                     assert(false && "this aggregation is not handled");
                     break;
               }
            } else if (auto countFunc = llvm::dyn_cast<relalg::CountRowsOp>(col.getDefiningOp())) {
               auto slot = newbuffername + "[" + buf_idx(op) + "]";
               appendKernel(fmt::format("aggregate_sum(&{0}, 1);", slot), KernelType::Main);
            } else {
               col.dump();
               col.getDefiningOp()->dump();
               assert(false && "No aggregation function for the new column in aggregation");
            }
            i++;
         }
      } else {
         assert(false && "nothing to aggregate!!");
      }

      // append control to allocate column defs
      for (auto& col : groupByKeys) {
         ColumnDetail detail(mlir::cast<tuples::ColumnRefAttr>(col));
         std::string mlirSymbol = detail.getMlirSymbol();
         std::string keyColumnType = mlirTypeToCudaType(detail.type);
         if (keyColumnType == "DBStringType") {
            std::string keyColumnName = KEY(op) + mlirSymbol + "_encoded";
            mainArgs[keyColumnName] = "DBI16Type*";
            mlirToGlobalSymbol[keyColumnName] = fmt::format("d_{}", keyColumnName);
            appendControl(fmt::format("DBI16Type* d_{0};", keyColumnName));
            appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(DBI16Type) * {1});", keyColumnName, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", keyColumnName));
            appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(DBI16Type) * {1});", keyColumnName, COUNT(op)));
            auto key = LoadColumn<1>(mlir::cast<tuples::ColumnRefAttr>(col), KernelType::Main);
            appendKernel(fmt::format("{0}[{1}] = {2};", keyColumnName, buf_idx(op), key), KernelType::Main);
         } else {
            std::string keyColumnName = KEY(op) + mlirSymbol;
            mainArgs[keyColumnName] = keyColumnType + "*";
            mlirToGlobalSymbol[keyColumnName] = fmt::format("d_{}", keyColumnName);
            appendControl(fmt::format("{0}* d_{1};", keyColumnType, keyColumnName));
            appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof({1}) * {2});", keyColumnName, keyColumnType, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", keyColumnName));
            appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof({1}) * {2});", keyColumnName, keyColumnType, COUNT(op)));
            auto key = LoadColumn(mlir::cast<tuples::ColumnRefAttr>(col), KernelType::Main);
            appendKernel(fmt::format("{0}[{1}] = {2};", keyColumnName, buf_idx(op), key), KernelType::Main);
         }
      }
      appendControl(launchKernel(KernelType::Main));
   }
   void MaterializeBuffers(mlir::Operation* op) {
      auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(op);
      if (!materializeOp) assert(false && "Materialize buffer needs materialize op as argument.");

      appendControl("//Materialize buffers");
      appendControl(fmt::format("uint64_t* d_{0};", MAT_IDX(op)));
      appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(uint64_t));", MAT_IDX(op)));
      deviceFrees.insert(fmt::format("d_{0}", MAT_IDX(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", MAT_IDX(op)));
      mainArgs[MAT_IDX(op)] = "uint64_t*";
      mlirToGlobalSymbol[MAT_IDX(op)] = "d_" + MAT_IDX(op);
      appendKernel("//Materialize buffers", KernelType::Main);
      appendKernel(fmt::format("auto {0} = atomicAdd((int*){1}, 1);", mat_idx(op), MAT_IDX(op)), KernelType::Main);
      for (auto col : materializeOp.getCols()) {
         auto columnAttr = mlir::cast<tuples::ColumnRefAttr>(col);
         auto detail = ColumnDetail(columnAttr);

         std::string mlirSymbol = detail.getMlirSymbol();
         std::string type = mlirTypeToCudaType(detail.type);

         if (type == "DBStringType") {
            std::string newBuffer = MAT(op) + mlirSymbol + "_encoded";
            appendControl(fmt::format("auto {0} = (DBI16Type*)malloc(sizeof(DBI16Type) * {1});", newBuffer, COUNT(op)));
            hostFrees.insert(newBuffer);
            appendControl(fmt::format("DBI16Type* d_{0};", newBuffer));
            appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(DBI16Type) * {1});", newBuffer, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", newBuffer));
            mainArgs[newBuffer] = "DBI16Type*";
            mlirToGlobalSymbol[newBuffer] = "d_" + newBuffer;
            auto key = LoadColumn<1>(columnAttr, KernelType::Main);
            appendKernel(fmt::format("{0}[{2}] = {1};", newBuffer, key, mat_idx(op)), KernelType::Main);
         } else {
            std::string newBuffer = MAT(op) + mlirSymbol;
            appendControl(fmt::format("auto {0} = ({1}*)malloc(sizeof({1}) * {2});", newBuffer, type, COUNT(op)));
            hostFrees.insert(newBuffer);
            appendControl(fmt::format("{1}* d_{0};", newBuffer, type));
            appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof({1}) * {2});", newBuffer, type, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", newBuffer));
            mainArgs[newBuffer] = type + "*";
            mlirToGlobalSymbol[newBuffer] = "d_" + newBuffer;
            auto key = LoadColumn(columnAttr, KernelType::Main);
            appendKernel(fmt::format("{0}[{2}] = {1};", newBuffer, key, mat_idx(op)), KernelType::Main);
         }
      }
      appendControl(launchKernel(KernelType::Main));
      // appendControl(fmt::format("cudaFree(d_{0});", MAT_IDX(op)));
      std::string printStmts;
      std::string delimiter = "|";
      bool first = true;
      for (auto col : materializeOp.getCols()) {
         auto columnAttr = mlir::cast<tuples::ColumnRefAttr>(col);
         auto detail = ColumnDetail(columnAttr);

         std::string mlirSymbol = detail.getMlirSymbol();
         std::string type = mlirTypeToCudaType(detail.type);
         if (type == "DBStringType") {
            std::string newBuffer = MAT(op) + mlirSymbol + "_encoded";

            appendControl(fmt::format("cudaMemcpy({0}, d_{0}, sizeof(DBI16Type) * {1}, cudaMemcpyDeviceToHost);",
                                      newBuffer, COUNT(op)));
            printStmts += fmt::format("std::cout << \"{0}\" << {2}[{1}[i]];\n", first ? "" : delimiter, newBuffer, mlirSymbol + "_map");
         } else {
            std::string newBuffer = MAT(op) + mlirSymbol;

            appendControl(fmt::format("cudaMemcpy({0}, d_{0}, sizeof({1}) * {2}, cudaMemcpyDeviceToHost);",
                                      newBuffer, type, COUNT(op)));
            printStmts += fmt::format("std::cout << \"{0}\" << {1}[i];\n", first ? "" : delimiter, newBuffer);
         }
         first = false;
      }
      appendControl(fmt::format("for (auto i=0ull; i < {0}; i++) {{ {1}std::cout << std::endl; }}",
                                COUNT(op), printStmts));
   }

   std::string mapOpDfs(mlir::Operation* op, std::vector<tuples::ColumnRefAttr>& dep) {
      // leaf condition
      if (auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(op)) {
         return translateConstantOp(constOp);
      }
      if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
         dep.push_back(getColOp.getAttr());
         ColumnDetail detail(getColOp.getAttr());
         return fmt::format("reg_{}", detail.getMlirSymbol());
      }
      if (auto binaryOp = mlir::dyn_cast_or_null<db::MulOp>(op)) {
         return fmt::format("({0}) * ({1})", mapOpDfs(binaryOp.getLeft().getDefiningOp(), dep),
                            mapOpDfs(binaryOp.getRight().getDefiningOp(), dep));
      } else if (auto binaryOp = mlir::dyn_cast_or_null<db::AddOp>(op)) {
         return fmt::format("({0}) + ({1})", mapOpDfs(binaryOp.getLeft().getDefiningOp(), dep),
                            mapOpDfs(binaryOp.getRight().getDefiningOp(), dep));
      } else if (auto binaryOp = mlir::dyn_cast_or_null<db::SubOp>(op)) {
         return fmt::format("({0}) - ({1})", mapOpDfs(binaryOp.getLeft().getDefiningOp(), dep),
                            mapOpDfs(binaryOp.getRight().getDefiningOp(), dep));
      } else if (auto binaryOp = mlir::dyn_cast_or_null<db::DivOp>(op)) {
         return fmt::format("({0}) / ({1})", mapOpDfs(binaryOp.getLeft().getDefiningOp(), dep),
                            mapOpDfs(binaryOp.getRight().getDefiningOp(), dep));
      } else if (auto castOp = mlir::dyn_cast_or_null<db::CastOp>(op)) {
         mlir::Type ty = castOp.getRes().getType();
         std::string cudaType = mlirTypeToCudaType(ty);
         return fmt::format("({1})({0})", mapOpDfs(castOp.getVal().getDefiningOp(), dep), cudaType);
      } else if (auto runtimeOp = mlir::dyn_cast_or_null<db::RuntimeCall>(op)) {
         std::string function = runtimeOp.getFn().str();
         std::string args = "";
         std::string sep = "";
         for (auto v : runtimeOp.getArgs()) {
            args += sep + mapOpDfs(v.getDefiningOp(), dep);
            sep = ", ";
         }
         return fmt::format("{0}({1})", function, args);
      } else if (auto scfIfOp = mlir::dyn_cast_or_null<mlir::scf::IfOp>(op)) {
         std::string cond = mapOpDfs(scfIfOp.getCondition().getDefiningOp(), dep);
         std::string thenBlock = "";
         std::string elseBlock = "";
         for (auto& block : scfIfOp.getThenRegion().getBlocks()) {
            for (auto& op : block) {
               if (mlir::isa<mlir::scf::YieldOp>(op)) {
                  thenBlock += mapOpDfs(&op, dep);
               }
            }
         }
         for (auto& block : scfIfOp.getElseRegion().getBlocks()) {
            for (auto& op : block) {
               if (mlir::isa<mlir::scf::YieldOp>(op)) {
                  elseBlock += mapOpDfs(&op, dep);
               }
            }
         }
         return fmt::format("({0}) ? ({1}) : ({2})", cond, thenBlock, elseBlock);
      } else if (auto deriveTruthOp = mlir::dyn_cast_or_null<db::DeriveTruth>(op)) {
         std::string expr = mapOpDfs(deriveTruthOp.getVal().getDefiningOp(), dep);
         return fmt::format("({0})", expr);
      } else if (auto compareOp = mlir::dyn_cast_or_null<db::CmpOp>(op)) {
         auto left = compareOp.getLeft();
         std::string leftOperand = SelectionOpDfs(left.getDefiningOp());

         auto right = compareOp.getRight();
         std::string rightOperand = SelectionOpDfs(right.getDefiningOp());

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
      } else if (auto yieldOp = mlir::dyn_cast_or_null<mlir::scf::YieldOp>(op)) {
         std::string expr = "";
         for (auto v : yieldOp.getResults()) {
            expr += mapOpDfs(v.getDefiningOp(), dep);
         }
         return fmt::format("({0})", expr);
      }
      op->dump();
      assert(false && "Unexpected compute graph");
   }

   void TranslateMapOp(mlir::Operation* op) {
      auto mapOp = mlir::dyn_cast_or_null<relalg::MapOp>(op);
      if (!mapOp) assert(false && "Translate map op expects a map operation");
      auto computedCols = mapOp.getComputedCols();
      auto& predRegion = mapOp.getPredicate();
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predRegion.front().getTerminator())) {
         assert(returnOp.getResults().size() == computedCols.size() && "Computed cols size not equal to result size");
         auto i = 0ull;
         for (auto col : computedCols) {
            auto colAttr = mlir::cast<tuples::ColumnDefAttr>(col);
            ColumnDetail detail(colAttr);
            std::string mlirSymbol = detail.getMlirSymbol();
            std::vector<tuples::ColumnRefAttr> dep;
            std::string expr = mapOpDfs(returnOp.getResults()[i].getDefiningOp(), dep);
            columnData[mlirSymbol] = new ColumnMetadata(expr, ColumnType::Mapped, StreamId, dep);
            i++;
         }
      } else {
         assert(false && "No return op found for the map operation region");
      }
   }
   void printKernel(KernelType ty, std::ostream& stream) {
      std::map<std::string, std::string> _args;
      std::string _kernelName;
      if (ty == KernelType::Main) {
         _args = mainArgs;
         _kernelName = "main";
      } else {
         _args = countArgs;
         _kernelName = "count";
      }
      bool hasHash = false;
      for (auto p : _args) hasHash |= (p.second == "HASHTABLE_FIND" || p.second == "HASHTABLE_INSERT" || p.second == "HASHTABLE_PROBE" || p.second == "HASHTABLE_INSERT_SJ" || p.second == "HASHTABLE_PROBE_SJ");
      if (hasHash) {
         stream << "template<";
         bool find = false, insert = false, probe = false;
         bool insertSJ = false, probeSJ = false;
         std::string sep = "";
         for (auto p : _args) {
            if (p.second == "HASHTABLE_FIND" && !find) {
               find = true;
               stream << sep + "typename " + p.second;
               sep = ", ";
            } else if (p.second == "HASHTABLE_INSERT" && !insert) {
               insert = true;
               stream << sep + "typename " + p.second;
               sep = ", ";
            } else if (p.second == "HASHTABLE_PROBE" && !probe) {
               probe = true;
               stream << sep + "typename " + p.second;
               sep = ", ";
            } else if (p.second == "HASHTABLE_INSERT_SJ" && !insertSJ) {
               insertSJ = true;
               stream << sep + "typename " + p.second;
               sep = ", ";
            } else if (p.second == "HASHTABLE_PROBE_SJ" && !probeSJ) {
               probeSJ = true;
               stream << sep + "typename " + p.second;
               sep = ", ";
            }
         }
         stream << ">\n";
      }
      stream << fmt::format("__global__ void {0}_{1}(", _kernelName, GetId((void*) this));
      std::string sep = "";
      for (auto p : _args) {
         stream << fmt::format("{0}{1} {2}", sep, p.second, p.first);
         sep = ", ";
      }
      stream << ") {\n";
      if (KernelType::Main == ty) {
         for (auto line : mainCode) { stream << line << std::endl; }
      } else {
         for (auto line : countCode) { stream << line << std::endl; }
      }
      for (int i = 0; i < forEachScopes; i++) {
         stream << "});\n";
      }
      stream << "}\n";
   }
   void printControl(std::ostream& stream) {
      for (auto line : controlCode) {
         stream << line << std::endl;
      }
   }
   void printFrees(std::ostream& stream) {
      for (auto df : deviceFrees) {
         stream << fmt::format("cudaFree({});\n", df);
      }
      for (auto hf : hostFrees) {
         stream << fmt::format("free({});\n", hf);
      }
   }
};

class CudaCodeGen : public mlir::PassWrapper<CudaCodeGen, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-cuda-code-gen"; }

   bool m_compilingSSB;

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaCodeGen)

   std::map<mlir::Operation*, TupleStreamCode*> streamCodeMap;
   std::vector<TupleStreamCode*> kernelSchedule;

   CudaCodeGen(bool compilingSSB)
      : m_compilingSSB(compilingSSB) {}

   void runOnOperation() override {
      getOperation().walk([&](mlir::Operation* op) {
         if (auto selection = llvm::dyn_cast<relalg::SelectionOp>(op)) {
            mlir::Operation* stream = selection.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation found for selection.");
            }

            mlir::Region& predicate = selection.getPredicate();
            streamCode->AddSelectionPredicate(predicate);
            streamCodeMap[op] = streamCode;
         } else if (auto joinOp = llvm::dyn_cast<relalg::InnerJoinOp>(op)) {
            auto leftStream = joinOp.getLeftMutable().get().getDefiningOp();
            auto rightStream = joinOp.getRightMutable().get().getDefiningOp();
            auto leftStreamCode = streamCodeMap[leftStream];
            auto rightStreamCode = streamCodeMap[rightStream];
            if (!leftStreamCode) {
               leftStream->dump();
               assert(false && "No downstream operation build side of hash join found");
            }
            if (!rightStreamCode) {
               rightStream->dump();
               assert(false && "No downstream operation probe side of hash join found");
            }
            leftStreamCode->MaterializeCount(op); // count of left
            auto leftCols = leftStreamCode->BuildHashTable(op); // main of left
            kernelSchedule.push_back(leftStreamCode);
            rightStreamCode->ProbeHashTable(op, leftCols);
            mlir::Region& predicate = joinOp.getPredicate();
            rightStreamCode->AddSelectionPredicate(predicate);

            streamCodeMap[op] = rightStreamCode;
         } else if (auto aggregationOp = llvm::dyn_cast<relalg::AggregationOp>(op)) {
            mlir::Operation* stream = aggregationOp.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for aggregation found");
            }

            streamCode->CreateAggregationHashTable(op); // count part
            streamCode->AggregateInHashTable(op); // main part
            kernelSchedule.push_back(streamCode);

            auto newStreamCode = new TupleStreamCode(op);
            streamCodeMap[op] = newStreamCode;
         } else if (auto scanOp = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
            std::string tableName = scanOp.getTableIdentifier().data();
            TupleStreamCode* streamCode = new TupleStreamCode(scanOp);

            streamCodeMap[op] = streamCode;
         } else if (auto mapOp = llvm::dyn_cast<relalg::MapOp>(op)) {
            auto stream = mapOp.getRelMutable().get().getDefiningOp();
            auto streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for map op found");
            }

            streamCode->TranslateMapOp(op);
            streamCodeMap[op] = streamCode;
         } else if (auto materializeOp = llvm::dyn_cast<relalg::MaterializeOp>(op)) {
            auto stream = materializeOp.getRelMutable().get().getDefiningOp();
            auto streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for materialize found");
            }

            streamCode->MaterializeCount(op);
            streamCode->MaterializeBuffers(op);
            kernelSchedule.push_back(streamCode);
         } else if (auto sortOp = llvm::dyn_cast<relalg::SortOp>(op)) {
            std::clog << "WARNING: This operator has not been implemented, bypassing it.\n";
            op->dump();
            streamCodeMap[op] = streamCodeMap[sortOp.getRelMutable().get().getDefiningOp()];
         } else if (auto renamingOp = llvm::dyn_cast<relalg::RenamingOp>(op)) {
            mlir::Operation* stream = renamingOp.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for renaming operation found");
            }

            streamCode->RenamingOp(renamingOp);
            streamCodeMap[op] = streamCode;
         } else if (auto semiJoinOp = llvm::dyn_cast<relalg::SemiJoinOp>(op)) {
            auto leftStream = semiJoinOp.getLeftMutable().get().getDefiningOp();
            auto rightStream = semiJoinOp.getRightMutable().get().getDefiningOp();
            auto leftStreamCode = streamCodeMap[leftStream];
            auto rightStreamCode = streamCodeMap[rightStream];
            if (!leftStreamCode) {
               leftStream->dump();
               assert(false && "No downstream operation left side of hash join found");
            }
            if (!rightStreamCode) {
               rightStream->dump();
               assert(false && "No downstream operation right side of hash join found");
            }
            rightStreamCode->MaterializeCount(op); // count of left
            rightStreamCode->BuildHashTableSemiJoin(op); // main of left
            kernelSchedule.push_back(rightStreamCode);
            leftStreamCode->ProbeHashTableSemiJoin(op);
            mlir::Region& predicate = semiJoinOp.getPredicate();
            leftStreamCode->AddSelectionPredicate(predicate);

            streamCodeMap[op] = leftStreamCode;
         } else if (auto antiSemiJoinOp = llvm::dyn_cast<relalg::AntiSemiJoinOp>(op)) {
            auto leftStream = antiSemiJoinOp.getLeftMutable().get().getDefiningOp();
            auto rightStream = antiSemiJoinOp.getRightMutable().get().getDefiningOp();
            auto leftStreamCode = streamCodeMap[leftStream];
            auto rightStreamCode = streamCodeMap[rightStream];
            if (!leftStreamCode) {
               leftStream->dump();
               assert(false && "No downstream operation left side of hash join found");
            }
            if (!rightStreamCode) {
               rightStream->dump();
               assert(false && "No downstream operation right side of hash join found");
            }
            rightStreamCode->MaterializeCount(op); // count of left
            rightStreamCode->BuildHashTableAntiSemiJoin(op); // main of left
            kernelSchedule.push_back(rightStreamCode);
            leftStreamCode->ProbeHashTableAntiSemiJoin(op);
            mlir::Region& predicate = antiSemiJoinOp.getPredicate();
            leftStreamCode->AddSelectionPredicate(predicate);

            streamCodeMap[op] = leftStreamCode;
         }
      });
      std::ofstream outputFile("output.cu");
      outputFile << "#include <cuco/static_map.cuh>\n\
#include <cuco/static_multimap.cuh>\n\
#include <thrust/copy.h>\n\
#include <thrust/device_vector.h>\n\
#include <thrust/host_vector.h>\n";
      outputFile << "#include \"cudautils.cuh\"\n\
#include \"db_types.h\"\n\
#include \"dbruntime.h\"\n";
      for (auto code : kernelSchedule) {
         code->printKernel(KernelType::Count, outputFile);
         code->printKernel(KernelType::Main, outputFile);
      }

      emitControlFunctionSignature(outputFile);

      for (auto code : kernelSchedule) {
         code->printControl(outputFile);
      }
      for (auto code : kernelSchedule) {
         code->printFrees(outputFile);
      }
      outputFile << "}";
      outputFile.close();
   }
};
}

static bool gCudaCodeGenEnabled = false;
static bool gCudaCodeGenNoCountEnabled = false;
static bool gCudaCrystalCodeGenEnabled = false;
static bool gCudaCrystalCodeGenNoCountEnabled = false;
static bool gCompilingSSB = false;

void emitControlFunctionSignature(std::ostream& outputFile) {
   if (!gCompilingSSB)
      outputFile << "extern \"C\" void control (DBI32Type * d_nation__n_nationkey, DBStringType * d_nation__n_name, DBI32Type * d_nation__n_regionkey, DBStringType * d_nation__n_comment, size_t nation_size, DBI32Type * d_supplier__s_suppkey, DBI32Type * d_supplier__s_nationkey, DBStringType * d_supplier__s_name, DBStringType * d_supplier__s_address, DBStringType * d_supplier__s_phone, DBDecimalType * d_supplier__s_acctbal, DBStringType * d_supplier__s_comment, size_t supplier_size, DBI32Type * d_partsupp__ps_suppkey, DBI32Type * d_partsupp__ps_partkey, DBI32Type * d_partsupp__ps_availqty, DBDecimalType * d_partsupp__ps_supplycost, DBStringType * d_partsupp__ps_comment, size_t partsupp_size, DBI32Type * d_part__p_partkey, DBStringType * d_part__p_name, DBStringType * d_part__p_mfgr, DBStringType * d_part__p_brand, DBStringType * d_part__p_type, DBI32Type * d_part__p_size, DBStringType * d_part__p_container, DBDecimalType * d_part__p_retailprice, DBStringType * d_part__p_comment, size_t part_size, DBI32Type * d_lineitem__l_orderkey, DBI32Type * d_lineitem__l_partkey, DBI32Type * d_lineitem__l_suppkey, DBI64Type * d_lineitem__l_linenumber, DBDecimalType * d_lineitem__l_quantity, DBDecimalType * d_lineitem__l_extendedprice, DBDecimalType * d_lineitem__l_discount, DBDecimalType * d_lineitem__l_tax, DBCharType * d_lineitem__l_returnflag, DBCharType * d_lineitem__l_linestatus, DBI32Type * d_lineitem__l_shipdate, DBI32Type * d_lineitem__l_commitdate, DBI32Type * d_lineitem__l_receiptdate, DBStringType * d_lineitem__l_shipinstruct, DBStringType * d_lineitem__l_shipmode, DBStringType * d_lineitem__comments, size_t lineitem_size, DBI32Type * d_orders__o_orderkey, DBCharType * d_orders__o_orderstatus, DBI32Type * d_orders__o_custkey, DBDecimalType * d_orders__o_totalprice, DBI32Type * d_orders__o_orderdate, DBStringType * d_orders__o_orderpriority, DBStringType * d_orders__o_clerk, DBI32Type * d_orders__o_shippriority, DBStringType * d_orders__o_comment, size_t orders_size, DBI32Type * d_customer__c_custkey, DBStringType * d_customer__c_name, DBStringType * d_customer__c_address, DBI32Type * d_customer__c_nationkey, DBStringType * d_customer__c_phone, DBDecimalType * d_customer__c_acctbal, DBStringType * d_customer__c_mktsegment, DBStringType * d_customer__c_comment, size_t customer_size, DBI32Type * d_region__r_regionkey, DBStringType * d_region__r_name, DBStringType * d_region__r_comment, size_t region_size, DBI16Type* d_nation__n_name_encoded, std::unordered_map<DBI16Type, DBStringType> &nation__n_name_map, std::unordered_map<DBI16Type, DBStringType> &n1___n_name_map, std::unordered_map<DBI16Type, DBStringType> &n2___n_name_map, DBI16Type* d_orders__o_orderpriority_encoded, std::unordered_map<DBI16Type, std::string>& orders__o_orderpriority_map, DBI16Type* d_customer__c_name_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_name_map, DBI16Type* d_customer__c_comment_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_comment_map, DBI16Type* d_customer__c_phone_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_phone_map, DBI16Type* d_customer__c_address_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_address_map, DBI16Type* d_supplier__s_name_encoded, std::unordered_map<DBI16Type, std::string>& supplier__s_name_map, DBI16Type* d_part__p_brand_encoded, std::unordered_map<DBI16Type, std::string>& part__p_brand_map, DBI16Type* d_part__p_type_encoded, std::unordered_map<DBI16Type, std::string>& part__p_type_map) {\n";
   else
      outputFile << "extern \"C\" void control (DBI32Type* d_supplier__s_suppkey, DBStringType* d_supplier__s_name, DBStringType* d_supplier__s_address, DBStringType* d_supplier__s_city, DBStringType* d_supplier__s_nation, DBStringType* d_supplier__s_region, DBStringType* d_supplier__s_phone, size_t supplier_size, DBI32Type* d_part__p_partkey, DBStringType* d_part__p_name, DBStringType* d_part__p_mfgr, DBStringType* d_part__p_category, DBStringType* d_part__p_brand1, DBStringType* d_part__p_color, DBStringType* d_part__p_type, DBI32Type* d_part__p_size, DBStringType* d_part__p_container, size_t part_size, DBI32Type* d_lineorder__lo_orderkey, DBI32Type* d_lineorder__lo_linenumber, DBI32Type* d_lineorder__lo_custkey, DBI32Type* d_lineorder__lo_partkey, DBI32Type* d_lineorder__lo_suppkey, DBDateType* d_lineorder__lo_orderdate, DBDateType* d_lineorder__lo_commitdate, DBStringType* d_lineorder__lo_orderpriority, DBCharType* d_lineorder__lo_shippriority, DBI32Type* d_lineorder__lo_quantity, DBDecimalType* d_lineorder__lo_extendedprice, DBDecimalType* d_lineorder__lo_ordtotalprice, DBDecimalType* d_lineorder__lo_revenue, DBDecimalType* d_lineorder__lo_supplycost, DBI32Type* d_lineorder__lo_discount, DBI32Type* d_lineorder__lo_tax, DBStringType* d_lineorder__lo_shipmode, size_t lineorder_size, DBI32Type* d_date__d_datekey, DBStringType* d_date__d_date, DBStringType* d_date__d_dayofweek, DBStringType* d_date__d_month, DBI32Type* d_date__d_year, DBI32Type* d_date__d_yearmonthnum, DBStringType* d_date__d_yearmonth, DBI32Type* d_date__d_daynuminweek, DBI32Type* d_date__d_daynuminmonth, DBI32Type* d_date__d_daynuminyear, DBI32Type* d_date__d_monthnuminyear, DBI32Type* d_date__d_weeknuminyear, DBStringType* d_date__d_sellingseason, DBI32Type* d_date__d_lastdayinweekfl, DBI32Type* d_date__d_lastdayinmonthfl, DBI32Type* d_date__d_holidayfl, DBI32Type* d_date__d_weekdayfl, size_t date_size, DBI32Type* d_customer__c_custkey, DBStringType* d_customer__c_name, DBStringType* d_customer__c_address, DBStringType* d_customer__c_city, DBStringType* d_customer__c_nation, DBStringType* d_customer__c_region, DBStringType* d_customer__c_phone, DBStringType* d_customer__c_mktsegment, size_t customer_size, DBI32Type* d_region__r_regionkey, DBStringType* d_region__r_name, DBStringType* d_region__r_comment, size_t region_size, DBI16Type* d_part__p_brand1_encoded, DBI16Type* d_supplier__s_nation_encoded, DBI16Type* d_customer__c_city_encoded, DBI16Type* d_supplier__s_city_encoded, DBI16Type* d_customer__c_nation_encoded, DBI16Type* d_part__p_category_encoded, std::unordered_map<DBI16Type, std::string>& part__p_brand1_map, std::unordered_map<DBI16Type, std::string>& supplier__s_nation_map, std::unordered_map<DBI16Type, std::string>& customer__c_city_map, std::unordered_map<DBI16Type, std::string>& supplier__s_city_map, std::unordered_map<DBI16Type, std::string>& customer__c_nation_map, std::unordered_map<DBI16Type, std::string>& part__p_category_map) {\n";
}

std::unique_ptr<mlir::Pass> relalg::createCudaCodeGenPass() {
   return std::make_unique<CudaCodeGen>(gCompilingSSB);
}

void relalg::addCudaCodeGenPass(mlir::OpPassManager& pm) {
   if (gCudaCodeGenEnabled) {
      pm.addNestedPass<mlir::func::FuncOp>(createCudaCodeGenPass());
   } else if (gCudaCodeGenNoCountEnabled) {
      pm.addNestedPass<mlir::func::FuncOp>(createCudaCodeGenNoCountPass());
   } else if (gCudaCrystalCodeGenNoCountEnabled) {
      pm.addNestedPass<mlir::func::FuncOp>(createCudaCrystalCodeGenNoCountPass());
   } else if (gCudaCrystalCodeGenEnabled) {
      pm.addNestedPass<mlir::func::FuncOp>(createCudaCrystalCodeGenPass());
   }
}

void removeCodeGenSwitch(int& argc, char** argv, int i) {
   // Remove --gen-cuda-code from the argument list
   for (int j = i; j < argc - 1; j++) {
      argv[j] = argv[j + 1];
   }
   argc--;
}

void checkForBenchmarkSwitch(int& argc, char** argv) {
   for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]) == "--ssb") {
         gCompilingSSB = true;
         std::clog << "Compiling for SSB benchmark\n";
         removeCodeGenSwitch(argc, argv, i);
         break;
      } else if (std::string(argv[i]) == "--tpch") {
         gCompilingSSB = false;
         removeCodeGenSwitch(argc, argv, i);
         break;
      }
   }
}

void relalg::conditionallyEnableCudaCodeGen(int& argc, char** argv) {
   for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]) == "--gen-cuda-code") {
         gCudaCodeGenEnabled = true;
         removeCodeGenSwitch(argc, argv, i);
         break;
      } else if (std::string(argv[i]) == "--gen-cuda-code-no-count") {
         gCudaCodeGenNoCountEnabled = true;
         removeCodeGenSwitch(argc, argv, i);
         break;
      } else if (std::string(argv[i]) == "--gen-cuda-crystal-code-no-count") {
         gCudaCrystalCodeGenNoCountEnabled = true;
         removeCodeGenSwitch(argc, argv, i);
         break;
      } else if (std::string(argv[i]) == "--gen-cuda-crystal-code") {
         gCudaCrystalCodeGenEnabled = true;
         removeCodeGenSwitch(argc, argv, i);
         break;
      }
   }
   checkForBenchmarkSwitch(argc, argv);
}