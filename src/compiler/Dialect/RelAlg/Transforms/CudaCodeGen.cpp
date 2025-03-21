
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
std::string ToHex(void* op) {
   std::stringstream sstream;
   sstream << std::hex << (unsigned long long) (void*) op;
   std::string result = sstream.str();
   return result;
}

static std::string HT(void* op) {
   return "HT_" + ToHex(op);
}
static std::string KEY(void* op) {
   return "KEY_" + ToHex(op);
}
static std::string SLOT(void* op) {
   return "SLOT_" + ToHex(op);
}
static std::string BUF(void* op) {
   return "BUF_" + ToHex(op);
}
static std::string BUF_IDX(void* op) {
   return "BUF_IDX_" + ToHex(op);
}
static std::string buf_idx(void* op) {
   return "buf_idx_" + ToHex(op);
}
static std::string COUNT(void* op) {
   return "COUNT" + ToHex(op);
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
   std::vector<tuples::ColumnRefAttr> dependencies; // valid if type is Mapped
   int streamId;
   ColumnMetadata(const std::string& le, ColumnType ty, int streamId) : loadExpression(le), type(ty), streamId(streamId) {}
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
   std::vector<TupleStreamCode*> joinedStreams;

   std::map<std::string, ColumnMetadata*> columnData;
   std::set<std::string> loadedColumns;
   std::set<std::string> loadedCountColumns;

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
         args += fmt::format("{1}d_{0}", p.first, sep);
         sep = ", ";
      }
      return fmt::format("{0}_{1}<<<std::ceil((float){2}/32.), 32>>>({3})", _kernelName, ToHex((void*) this), size, args);
   }

   public:
   TupleStreamCode(relalg::BaseTableOp baseTableOp) {
      static int StreamId = 0;
      std::string tableName = baseTableOp.getTableIdentifier().data();
      std::string tableSize = tableName + "_size";
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only

      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;", KernelType::Main);
      appendKernel(fmt::format("if (tid >= {}) return;", tableSize), KernelType::Main);
      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;", KernelType::Count);
      appendKernel(fmt::format("if (tid >= {}) return;", tableSize), KernelType::Count);
      for (auto namedAttr : baseTableOp.getColumns().getValue()) {
         auto columnName = namedAttr.getName().str();
         ColumnDetail detail(mlir::cast<tuples::ColumnDefAttr>(namedAttr.getValue()));
         auto globalSymbol = fmt::format("{0}__{1}", tableName, columnName);
         auto mlirSymbol = detail.getMlirSymbol();
         mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
         ColumnMetadata* metadata = new ColumnMetadata(fmt::format("{0}", mlirSymbol), ColumnType::Direct, StreamId);
         metadata->rid = "tid";
         columnData[mlirSymbol] = metadata;
      }
      id = StreamId;
      StreamId++;
      return;
   }
   ~TupleStreamCode() {
      for (auto p : columnData) delete p.second;
   }
   std::string LoadColumn(const tuples::ColumnRefAttr& attr, KernelType ty) {
      ColumnDetail detail(attr);
      auto mlirSymbol = detail.getMlirSymbol();

      if (columnData.find(mlirSymbol) == columnData.end()) {
         assert(false && "Column ref not in tuple stream");
      }
      auto cudaId = fmt::format("reg_{0}", mlirSymbol);
      if (ty == KernelType::Main && loadedColumns.find(mlirSymbol) == loadedColumns.end()) {
         loadedColumns.insert(mlirSymbol);
      } else if (ty == KernelType::Count && loadedCountColumns.find(mlirSymbol) == loadedCountColumns.end()) {
         loadedCountColumns.insert(mlirSymbol);
      }
      auto colData = columnData[mlirSymbol];
      if (colData->type == ColumnType::Mapped) {
         for (auto dep : colData->dependencies) {
            LoadColumn(dep, ty);
         }
      }
      appendKernel(fmt::format("auto {1} = {0};", colData->loadExpression + (colData->type == ColumnType::Direct ? "[" + colData->rid + "]" : ""), cudaId), ty);
      if (ty == KernelType::Main) {
         mainArgs[mlirSymbol] = mlirTypeToCudaType(detail.type) + "*"; // columns are always a 1d array
      } else {
         countArgs[mlirSymbol] = mlirTypeToCudaType(detail.type) + "*";
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
         assert(false && "TODO: handle runtime predicates\n");
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
      appendKernel(fmt::format("atomicAdd({0}, 1);", COUNT(op)), KernelType::Count);

      appendControl(fmt::format("uint64_t* d_{0};", COUNT(op)));
      appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(uint64_t));", COUNT(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", COUNT(op)));
      appendControl(launchKernel(KernelType::Count));
      appendControl(fmt::format("uint64_t {0};", COUNT(op)));
      appendControl(fmt::format("cudaMemcpy(&{0}, d_{0}, sizeof(uint64_t), cudaMemcpyDeviceToHost)", COUNT(op)));
      appendControl(fmt::format("cudaFree(d_{0});", COUNT(op)));
   }
   std::string MakeKeys(mlir::Operation* op, const mlir::ArrayAttr& keys, KernelType kernelType) {
      //TODO(avinash, p3): figure a way out for double keys
      appendKernel(fmt::format("uint64_t {0} = 0;", KEY(op)), kernelType);
      std::map<std::string, int> allowedKeysToSize;
      allowedKeysToSize["DBCharType"] = 1;
      allowedKeysToSize["DBI32Type"] = 4;
      allowedKeysToSize["DBDateType"] = 4;
      allowedKeysToSize["DBI64Type"] = 8;
      int totalKeySize = 0;
      for (auto i = 0ull; i < keys.size(); i++) {
         tuples::ColumnRefAttr key = mlir::cast<tuples::ColumnRefAttr>(keys[i]);
         auto baseType = mlirTypeToCudaType(key.getColumn().type);
         if (allowedKeysToSize.find(baseType) == allowedKeysToSize.end()) {
            keys.dump();
            assert(false && "Type is not hashable");
         }
         std::string cudaIdentifierKey = LoadColumn(key, kernelType);
         appendKernel(fmt::format("{0} <<= {1};\n", KEY(op), std::to_string(allowedKeysToSize[baseType] * 8)), kernelType);
         appendKernel(fmt::format("{0} |= {1};\n", KEY(op), cudaIdentifierKey), kernelType);
         totalKeySize += allowedKeysToSize[baseType];
         if (totalKeySize > 8) {
            assert(false && "Total hash key exceeded 8 bytes");
         }
      }
      return KEY(op);
   }

   std::vector<std::pair<int, std::string>> getBaseRelations(const std::map<std::string, ColumnMetadata*>& columnData) {
      std::set<std::pair<int, std::string>> temp;
      for (auto p : columnData) {
         auto metadata = p.second;
         if (metadata->type == ColumnType::Direct)
            temp.insert(std::make_pair(metadata->streamId, metadata->rid));
      }
      std::vector<std::pair<int, std::string>> baseRelations(temp.begin(), temp.end());
      std::sort(baseRelations.begin(), baseRelations.end());
      return baseRelations;
   }
   std::map<std::string, ColumnMetadata*> InsertHashTable(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Insert hash table accepts only inner join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel(fmt::format("auto {0} = atomicAdd({1}, 1);", buf_idx(op), BUF_IDX(op)), KernelType::Main);
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
      appendControl(fmt::format("uint64_t* d_{0};", BUF_IDX(op)));
      appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(uint64_t));", BUF_IDX(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", BUF_IDX(op)));
      appendControl(fmt::format("uint64_t* d_{0}", BUF(op)));
      appendControl(fmt::format("cudaMalloc(&d_{0}, sizeof(uint64_t) * {1} * {2});", BUF(op), COUNT(op), baseRelations.size()));
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ {1}*2, cuco::empty_key{{(int64_t)-1}},cuco::empty_value{{(int64_t)-1}},thrust::equal_to<int64_t>{{}},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>() }};",
                                HT(op), COUNT(op)));
      appendControl(launchKernel(KernelType::Main));
      appendControl(fmt::format("cudaFree(d_{0});", BUF_IDX(op)));
      return columnData;
   }
   void ProbeHashTable(mlir::Operation* op, const std::map<std::string, ColumnMetadata*>& leftColumnData) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Probe hash table accepts only inner join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      MakeKeys(op, keys, KernelType::Count);
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Main);
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key), KernelType::Count);
      appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)), KernelType::Main);
      appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)), KernelType::Count);

      // add all leftColumn data to this data
      auto baseRelations = getBaseRelations(leftColumnData);
      std::map<int, int> streamIdToBufId;
      int i = 0;
      for (auto br : baseRelations) {
         streamIdToBufId[br.first] = i;
         i++;
      }
      for (auto colData : leftColumnData) {
         colData.second->streamId = id;
         if (colData.second->type == ColumnType::Direct) {
            colData.second->rid = fmt::format("{3}[{0}->second * {1} + {2}]",
                                              SLOT(op),
                                              std::to_string(baseRelations.size()),
                                              streamIdToBufId[colData.second->streamId],
                                              BUF(op));
            columnData[colData.first] = colData.second;
         }
         columnData[colData.first] = colData.second;
      }
      mainArgs[HT(op)] = "HASHTABLE_FIND";
      mainArgs[BUF(op)] = "uint64_t*";
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
            streamCode->AddSelectionPredicate(predicate);
            streamCodeMap[op] = streamCode;
         } else if (auto joinOp = llvm::dyn_cast<relalg::InnerJoinOp>(op)) {
            auto leftStream = joinOp.getLeftMutable().get().getDefiningOp();
            auto rightStream = joinOp.getRightMutable().get().getDefiningOp();
            auto leftStreamCode = streamCodeMap[leftStream];
            auto rightStreamCode = streamCodeMap[rightStream];
            if (!leftStreamCode) assert(false && "No downstream operation build side of hash join found");

            leftStreamCode->MaterializeCount(op);
            auto leftCols = leftStreamCode->InsertHashTable(op);
            rightStreamCode->ProbeHashTable(op, leftCols);

            streamCodeMap[op] = rightStreamCode;
         } else if (auto aggregationOp = llvm::dyn_cast<relalg::AggregationOp>(op)) {
         } else if (auto scanOp = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
            std::string tableName = scanOp.getTableIdentifier().data();
            TupleStreamCode* streamCode = new TupleStreamCode(scanOp);

            streamCodeMap[op] = streamCode;
         } else if (auto sortOp = llvm::dyn_cast<relalg::SortOp>(op)) {
         } else if (auto materializeOp = llvm::dyn_cast<relalg::MaterializeOp>(op)) {
         }
      });
   }
};
}

std::unique_ptr<mlir::Pass>
relalg::createCudaCodeGenPass() { return std::make_unique<CudaCodeGen>(); }
