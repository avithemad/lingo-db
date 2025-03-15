
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

namespace {
using namespace lingodb::compiler::dialect;
enum class KernelType {
   Main,
   Count
};

std::string join(std::vector<std::string>& v, std::string separator) {
   // below is quadratic, can be made linear with some effort, but arguments and parameters should not be that huge to matter.
   std::string res = "", sep = "";
   for (auto e : v) {
      res += (sep + e);
      sep = separator;
   }
   return res;
}
// TODO(avinash, p1): Check if StringColumn defined as char* is sufficient for direct comparisons, especially with null terminated c style strings within cuda device memory.
void printAllTpchSchema(std::ostream& stream) {
   stream << "#include <cuco/static_map.cuh>\n";
   stream << "\n";
}
// for all the cudaidentifier that create a state for example join, aggregation, use the operation address
// instead of the stream address, which ensures that uniqueness for the data structure used by the operation
// is maintained
std::string convertToHex(void* op) {
   std::stringstream sstream;
   sstream << std::hex << (unsigned long long) (void*) op;
   std::string result = sstream.str();
   return result;
}
typedef std::map<std::string, std::string> RIDMAP;
typedef std::set<std::string> LOADEDCOLUMNS;
static std::string getBaseCudaType(mlir::Type ty) {
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
   ty.dump();
   assert(false && "unhandled type");
   return "";
}
struct TupleStreamCode {
   std::vector<std::string> baseRelation; // in data centric code gen, each stream will have exactly one base relation where it scans from
   std::string kernelCode;
   std::string kernelCountCode;
   std::string controlCode;
   std::string controlCountCode;
   int threadBlockSize = 32;
   // argument represents type and name of kernel arguments
   // using map to preserve the order of arguments, kernelArgs[argument] = type
   std::map<std::string, mlir::Type> kernelArgs; // for storing database columns
   std::map<std::string, std::string> stateArgs; // for storing out custom data structures

   std::map<std::string, mlir::Type> kernelCountArgs; // for storing database columns
   std::map<std::string, std::string> stateCountArgs; // for storing out custom data structures
   RIDMAP ridMap; // row identifier map. maps table to cuda identifier containing the RID of the table

   // enforce the invariant that code for each loaded column withing the belows sets are present in the kernel codes respectively.
   // Right now this is mental model, but as the project grows this would be useful TODO(avinash)
   LOADEDCOLUMNS loadedColumns; // loaded registers within the stream kernel
   LOADEDCOLUMNS loadedCountColumns;
   TupleStreamCode() : kernelCode("") {}

   void appendKernel(std::string code) {
      kernelCode += code + "\n";
   }
   void appendCountKernel(std::string code) {
      kernelCountCode += code + "\n";
   }

   void appendCountControl(std::string code) {
      controlCountCode += code + "\n";
   }
   void appendControl(std::string code) {
      controlCode += code + "\n";
   }
   std::string launchKernel(KernelType ty) {
      std::string _kernelName;
      std::map<std::string, std::string> _stateArgs;
      std::map<std::string, mlir::Type> _kernelArgs;
      if (ty == KernelType::Main) {
         _kernelName = "main";
         _stateArgs = stateArgs;
         _kernelArgs = kernelArgs;
      } else {
         _stateArgs = stateCountArgs;
         _kernelArgs = kernelCountArgs;
         _kernelName = "count";
      }
      std::string res = _kernelName + "_pipeline_" + convertToHex((void*) this);
      res += "<<<std::ceil((float)" + *(baseRelation.end() - 1) + "_size/(float)" + std::to_string(threadBlockSize) + "), " + std::to_string(threadBlockSize) + ">>>(";
      for (auto p : _kernelArgs) {
         res += "d_" + p.first + ", ";
      }
      std::vector<std::string> args;
      for (auto p : _stateArgs) {
         std::string arg = "";
         if (p.second == "HASHTABLE_FIND") {
            arg = p.first + ".ref(cuco::find)";
         } else if (p.second == "HASHTABLE_INSERT") {
            arg = p.first + ".ref(cuco::insert)";
         } else if (p.second == "size_t") {
            arg = p.first;
         } else {
            arg = "d_" + p.first;
         }
         args.push_back(arg);
      }
      res += join(args, ", ") + ");";
      return res;
   }
   void printKernel(KernelType ty, std::ostream& stream) {
      std::map<std::string, std::string> _stateArgs;
      std::map<std::string, mlir::Type> _kernelArgs;
      std::string _kernelName;
      std::string _kernelCode;
      if (ty == KernelType::Main) {
         _stateArgs = stateArgs;
         _kernelArgs = kernelArgs;
         _kernelName = "main";
         _kernelCode = kernelCode;
      } else {
         _stateArgs = stateCountArgs;
         _kernelArgs = kernelCountArgs;
         _kernelName = "count";
         _kernelCode = kernelCountCode;
      }

      std::set<std::string> hashTableTypes;
      for (auto p : _stateArgs) {
         if (p.second == "HASHTABLE_FIND" || p.second == "HASHTABLE_INSERT") hashTableTypes.insert(p.second);
      }
      std::vector<std::string> args;
      if (hashTableTypes.size() > 0) {
         stream << "template<";
         for (auto ty : hashTableTypes) {
            args.push_back("typename " + ty);
         }
         stream << join(args, ", ") << ">" << std::endl;
         args.clear();
      }
      stream << "__global__ void " + _kernelName + "_pipeline_" + convertToHex((void*) this) + "(";

      for (auto p : _kernelArgs) {
         args.push_back(getBaseCudaType(p.second) + " *" + p.first);
      }
      stream << join(args, ",\n");
      args.clear();
      if (_stateArgs.size() > 0) stream << ",\n";
      for (auto p : _stateArgs) {
         args.push_back(p.second + " " + p.first);
      }
      stream << join(args, ",\n") << std::endl;
      args.clear();
      stream << ") {" << std::endl;
      stream << _kernelCode << "}\n";
   }
   void printControl(KernelType ty, std::ostream& stream) {
      if (ty == KernelType::Main)
         stream << controlCode + "\n";
      else
         stream << controlCountCode + "\n";
   }
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
struct ColumnDetail {
   std::string name;
   std::string relation;
   mlir::Type type;
   ColumnDetail(std::string name, std::string relation, mlir::Type type) : name(name), relation(relation), type(type) {}
   void print() {
      std::clog << relation << "->" << name << " : ";
      type.dump();
      std::clog << std::endl;
   }

   ColumnDetail(const tuples::ColumnRefAttr& colAttr) {
      relation = getTableName<tuples::ColumnRefAttr>(colAttr);
      name = getColumnName<tuples::ColumnRefAttr>(colAttr);
      type = colAttr.getColumn().type;
   }
};

std::string LoadColumnIntoStream(TupleStreamCode* streamCode, const tuples::ColumnRefAttr& colAttr, KernelType type) {
   // add to the kernel argument, get the name and type from colAttr
   ColumnDetail detail(colAttr);
   std::string cudaIdentifier = "reg__" + detail.relation + "__" + detail.name;
   if (type == KernelType::Main) {
      if (streamCode->loadedColumns.find(cudaIdentifier) == streamCode->loadedColumns.end()) {
         if (streamCode->ridMap.find(detail.relation) == streamCode->ridMap.end()) assert(false && "No record identifier for the table found");
         // load the column into register
         streamCode->loadedColumns.insert(cudaIdentifier);
         streamCode->appendKernel("auto reg__" + detail.relation + "__" + detail.name + " = " + detail.relation + "__" + detail.name + "[" + streamCode->ridMap[detail.relation] + "];");
         streamCode->kernelArgs[detail.relation + "__" + detail.name] = detail.type; // add information to the arguments
      }
      assert(streamCode->loadedColumns.find(cudaIdentifier) != streamCode->loadedColumns.end());
   } else {
      if (streamCode->loadedCountColumns.find(cudaIdentifier) == streamCode->loadedCountColumns.end()) {
         if (streamCode->ridMap.find(detail.relation) == streamCode->ridMap.end()) assert(false && "No record identifier for the table found");
         streamCode->loadedCountColumns.insert(cudaIdentifier);
         streamCode->appendCountKernel("auto reg__" + detail.relation + "__" + detail.name + " = " + detail.relation + "__" + detail.name + "[" + streamCode->ridMap[detail.relation] + "];");
         streamCode->kernelCountArgs[detail.relation + "__" + detail.name] = detail.type; // add information to the arguments
      }
      assert(streamCode->loadedCountColumns.find(cudaIdentifier) != streamCode->loadedCountColumns.end());
   }
   return cudaIdentifier;
}
int daysSinceEpoch(const std::string& dateStr) {
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

   return duration.count() / 24; // Convert hours to days
}

std::string translateConstantOp(mlir::Operation* operand) {
   std::string result = "";
   if (auto constantOp = mlir::dyn_cast_or_null<db::ConstantOp>(operand)) {
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
      }
   }
   operand->dump();
   assert(false && "Unable to translate the operand");
   return "";
}

static std::string translateSelection(mlir::Region& predicate, TupleStreamCode* streamCode) {
   auto terminator = mlir::cast<tuples::ReturnOp>(predicate.front().getTerminator());
   if (!terminator.getResults().empty()) {
      auto& predicateBlock = predicate.front();
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predicateBlock.getTerminator())) {
         mlir::Value matched = returnOp.getResults()[0];
         std::vector<std::pair<int, mlir::Value>> conditions;
         // hoping that we always have a compare in selection
         if (auto compareOp = mlir::dyn_cast_or_null<db::CmpOp>(matched.getDefiningOp())) {
            auto left = compareOp.getLeft();
            std::string leftOperand;
            if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(left.getDefiningOp())) {
               LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Count); // selection always needs to be done in count code as well
               leftOperand = LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Main);
            } else {
               leftOperand = translateConstantOp(left.getDefiningOp());
            }

            auto right = compareOp.getRight();
            std::string rightOperand;
            if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(right.getDefiningOp())) {
               LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Count); // selection always needs to be done in count code as well
               rightOperand = LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Main);
            } else {
               rightOperand = translateConstantOp(right.getDefiningOp());
            }

            auto cmp = compareOp.getPredicate();
            switch (cmp) {
               case db::DBCmpPredicate::eq:
                  /* code */
                  {
                     return "evaluatePredicate(" + leftOperand + ", " + rightOperand + ", Predicate::eq)";
                  }
                  break;
               case db::DBCmpPredicate::neq: {
                  return "evaluatePredicate(" + leftOperand + ", " + rightOperand + ", Predicate::neq)";
               } break;
               case db::DBCmpPredicate::lt: {
                  return "evaluatePredicate(" + leftOperand + ", " + rightOperand + ", Predicate::lt)";
               } break;
               case db::DBCmpPredicate::gt: {
                  return "evaluatePredicate(" + leftOperand + ", " + rightOperand + ", Predicate::gt)";
               } break;
               case db::DBCmpPredicate::lte: {
                  return "evaluatePredicate(" + leftOperand + ", " + rightOperand + ", Predicate::lte)";
               } break;
               case db::DBCmpPredicate::gte: {
                  return "evaluatePredicate(" + leftOperand + ", " + rightOperand + ", Predicate::gte)";
               } break;
               case db::DBCmpPredicate::isa: {
                  assert(false && "should not happen");
               }
               default:
                  break;
            }
         } else if (auto compareOp = mlir::dyn_cast_or_null<db::RuntimeCall>(matched.getDefiningOp())) { // or a like operation
            // TODO(avinash, p1): handle runtime predicate like operator
            assert(false && "TODO: handle runtime predicates\n");
         }
      } else {
         assert(false && "invalid");
      }
   }
   return "";
}

static std::string HT(void* op) {
   return "HT_" + convertToHex(op);
}
static std::string KEY(void* op) {
   return "KEY_" + convertToHex(op);
}
static std::string SLOT(void* op) {
   return "SLOT_" + convertToHex(op);
}
static std::string d_BUF(void* op) {
   return "d_BUF_" + convertToHex(op);
}
static std::string d_BUF_IDX(void* op) {
   return "d_BUF_IDX_" + convertToHex(op);
}
static std::string BUF(void* op) {
   return "BUF_" + convertToHex(op);
}
static std::string BUF_IDX(void* op) {
   return "BUF_IDX_" + convertToHex(op);
}
static std::string buf_idx(void* op) {
   return "buf_idx_" + convertToHex(op);
}

static std::string MakeKeysInStream(mlir::Operation* op, TupleStreamCode* stream, const mlir::ArrayAttr& keys, KernelType kernelType) {
   //TODO(avinash, p3): figure a way out for double keys
   std::string keyMaker = "int64_t " + KEY(op) + " = 0;\n";
   std::map<std::string, int> allowedKeysToSize;
   allowedKeysToSize["DBCharType"] = 1;
   allowedKeysToSize["DBI32Type"] = 4;
   allowedKeysToSize["DBDateType"] = 4;
   allowedKeysToSize["DBI64Type"] = 8;
   int totalKeySize = 0;
   for (auto i = 0ull; i < keys.size(); i++) {
      tuples::ColumnRefAttr key = mlir::cast<tuples::ColumnRefAttr>(keys[i]);
      auto baseType = getBaseCudaType(key.getColumn().type);
      if (allowedKeysToSize.find(baseType) == allowedKeysToSize.end()) {
         assert(false && "Type is not hashable");
      }
      std::string cudaIdentifierKey = LoadColumnIntoStream(stream, key, kernelType);
      keyMaker += KEY(op) + " <<= " + std::to_string(allowedKeysToSize[baseType] * 8) + ";\n";
      keyMaker += KEY(op) + "  |= " + cudaIdentifierKey + ";\n";
      totalKeySize += allowedKeysToSize[baseType];
      if (totalKeySize > 8) {
         assert(false && "Total hash key exceeded 8 bytes");
      }
   }
   if (kernelType == KernelType::Main)
      stream->appendKernel(keyMaker);
   else
      stream->appendCountKernel(keyMaker);
   return KEY(op);
}
void mapOpDfs(mlir::Operation* op, TupleStreamCode* streamCode, std::ostream& expr) {
   // leaf condition
   if (auto constOp = mlir::dyn_cast_or_null<db::ConstantOp>(op)) {
      expr << translateConstantOp(constOp);
      return;
   }
   if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(op)) {
      expr << LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Main);
      return;
   }
   if (auto binaryOp = mlir::dyn_cast_or_null<db::MulOp>(op)) {
      expr << "(";
      mapOpDfs(binaryOp.getLeft().getDefiningOp(), streamCode, expr);
      expr << ")";
      expr << " * ";
      expr << "(";
      mapOpDfs(binaryOp.getRight().getDefiningOp(), streamCode, expr);
      expr << ")";
      return;
   } else if (auto binaryOp = mlir::dyn_cast_or_null<db::AddOp>(op)) {
      expr << "(";
      mapOpDfs(binaryOp.getLeft().getDefiningOp(), streamCode, expr);
      expr << ")";
      expr << " + ";
      expr << "(";
      mapOpDfs(binaryOp.getRight().getDefiningOp(), streamCode, expr);
      expr << ")";
      return;
   } else if (auto binaryOp = mlir::dyn_cast_or_null<db::SubOp>(op)) {
      expr << "(";
      mapOpDfs(binaryOp.getLeft().getDefiningOp(), streamCode, expr);
      expr << ")";
      expr << " - ";
      expr << "(";
      mapOpDfs(binaryOp.getRight().getDefiningOp(), streamCode, expr);
      expr << ")";
      return;
   } else if (auto binaryOp = mlir::dyn_cast_or_null<db::DivOp>(op)) {
      expr << "(";
      mapOpDfs(binaryOp.getLeft().getDefiningOp(), streamCode, expr);
      expr << ")";
      expr << " / ";
      expr << "(";
      mapOpDfs(binaryOp.getRight().getDefiningOp(), streamCode, expr);
      expr << ")";
      return;
   } else if (auto castOp = mlir::dyn_cast_or_null<db::CastOp>(op)) {
      expr << "(castOp)(";
      mapOpDfs(castOp.getVal().getDefiningOp(), streamCode, expr);
      expr << ")";
      return;
   }
   op->dump();
   assert(false && "Unexpected compute graph");
}
class PythonCodeGen : public mlir::PassWrapper<PythonCodeGen, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-python-code-gen"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PythonCodeGen)

   // this seems good enough, re-evaluate the design later
   std::map<mlir::Operation*, std::string> streamMap;
   std::map<mlir::Operation*, TupleStreamCode*> streamCodeMap;
   std::vector<TupleStreamCode*> kernelSchedule;

   std::string getNextPythonVariable() {
      static int python_variable_id = 0;
      return "v" + std::to_string(python_variable_id++);
   }

   // TODO(avinash, p1) : potentially all operations could itself start a new pipeline
   // 1. Check if streamCodeMap[stream] is a pipeline ender. (Aggregation/materialization)
   // 2. if yes, then start a new tuple stream. (This may happen for any operation except the base_table_scan)
   // TODO(avinash, p1) : handle when there might be 2 base table scans of the same name
   // here the global name for the table columns are the same, but the identifiers within the kernel will change (needed for q2)
   void runOnOperation() override {
      getOperation().walk([&](mlir::Operation* op) {
         if (auto selection = llvm::dyn_cast<relalg::SelectionOp>(op)) {
            ::mlir::Operation* stream = selection.getRelMutable().get().getDefiningOp();

            TupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) assert(false && "No downstream operation for selection found");
            // Get the predicate region
            ::mlir::Region& predicateRegion = selection.getPredicate();
            auto condition = translateSelection(predicateRegion, streamCode);

            streamCode->appendKernel("if (!(" + condition + ")) return;");
            streamCode->appendCountKernel("if (!(" + condition + ")) return;");
            streamCodeMap[op] = streamCode;

            //this is basically produce code for the scan
         } else if (auto aggregation = llvm::dyn_cast<relalg::AggregationOp>(op)) {
            /**
						* This is a materializing operation.
						* Get the keys for aggregation and the tuplestream
						*/
            mlir::Operation* stream = aggregation.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) assert(false && "No downstream operation for aggregation found");

            mlir::ArrayAttr groupByKeys = aggregation.getGroupByCols(); // these are columnrefs
            mlir::ArrayAttr computedCols = aggregation.getComputedCols(); // these are columndefs

            // create hash table for the stream for this aggregation
            streamCode->stateCountArgs[HT(op)] = "HASHTABLE_INSERT";

            // compute the keys
            auto cudaIdentifierKey = MakeKeysInStream(op, streamCode, groupByKeys, KernelType::Count);
            MakeKeysInStream(op, streamCode, groupByKeys, KernelType::Main); // make keys in main as well as count

            streamCode->appendCountKernel(HT(op) + ".insert(cuco::pair{" + cudaIdentifierKey + ", 1});");
            // end the count kernel
            streamCode->appendCountKernel("return;");
            // add algo
            // - to count the keys in HT,
            // - initialize buffers of size size(HT)*computedcols.size
            // - create the host buffers and copy the data from gpu to cpu
            // note: use the rows estimate for this operation for sizing the hashtable.
            //
            // TODO(avinash, p2): make sure this estimate is an overestimate (it is complete, doesnt give less value is actually we have more)
            // also check if use-db is enabled, otherwise all query optimization costs are wrong

            std::string ht_size = "0";
            // TODO(avinash, p2): this is a hacky way, actually check if --use-db flag is enabled and query optimization is performed
            if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(op->getAttr("rows"))) {
               if (std::floor(floatAttr.getValueAsDouble()) != 0)
                  ht_size = std::to_string((size_t) std::ceil(floatAttr.getValueAsDouble()));
            }
            if (ht_size == "0") {
               // take the base relation's size
               ht_size = *(streamCode->baseRelation.end() - 1) + "_size";
            }

            streamCode->appendCountControl("auto " + HT(op) + " = cuco::static_map{ " + ht_size + "* 2,cuco::empty_key{(int64_t)-1},cuco::empty_value{(int64_t)-1},thrust::equal_to<int64_t>{},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>()};");
            streamCode->appendCountControl(streamCode->launchKernel(KernelType::Count));
            // TODO(avinash, p1): add thrust code to assign unique identifier for each key slot

            for (auto& col : computedCols) {
               std::string colName = getColumnName<tuples::ColumnDefAttr>(mlir::cast<tuples::ColumnDefAttr>(col));
               std::string tableName = getTableName<tuples::ColumnDefAttr>(mlir::cast<tuples::ColumnDefAttr>(col));
               // create buffers of aggregation length obtained in the count kernel, and create new buffers in the control code
               streamCode->kernelArgs[tableName + "__" + colName] = (mlir::cast<tuples::ColumnDefAttr>(col)).getColumn().type;
               // create a new buffer in control side with size d_HT.size()
               auto cudaType = getBaseCudaType((mlir::cast<tuples::ColumnDefAttr>(col)).getColumn().type) + "* ";
               streamCode->appendControl(cudaType + " d_" + tableName + "__" + colName + ";");
               // remove star from cudaType
               auto baseCudaType = getBaseCudaType((mlir::cast<tuples::ColumnDefAttr>(col)).getColumn().type);
               streamCode->appendControl("cudaMalloc(&d_" + tableName + "__" + colName + ", sizeof(" + baseCudaType + ") * " + HT(op) + ".size());");
               streamCode->appendControl("cudaMemset(d_" + tableName + "__" + colName + ",0 , sizeof(" + baseCudaType + ") * " + HT(op) + ".size());");
            }
            streamCode->stateArgs[HT(op)] = "HASHTABLE_FIND";
            streamCode->appendKernel("auto " + buf_idx(op) + " = " + HT(op) + ".find(" + cudaIdentifierKey + ")->second;");
            // walk through the region
            auto& aggRgn = aggregation.getAggrFunc();
            std::map<mlir::Operation*, std::string> newColumnMap;
            if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(aggRgn.front().getTerminator())) {
               int i = 0;
               for (mlir::Value col : returnOp.getResults()) {
                  // map each aggrfunc which is col.getDefiningOp to computedColName
                  auto newcol = mlir::cast<tuples::ColumnDefAttr>(computedCols[i]);
                  auto newbuffername = getTableName<tuples::ColumnDefAttr>(newcol) + "__" + getColumnName<tuples::ColumnDefAttr>(newcol);
                  newColumnMap[col.getDefiningOp()] = newbuffername;
                  i++;
               }
            } else {
               assert(false && "nothing to aggregate!!");
            }
            for (auto& regionOp : aggRgn.front()) {
               // now materialize all the computed columns here
               if (auto aggrFunc = llvm::dyn_cast<relalg::AggrFuncOp>(regionOp)) {
                  auto fn = aggrFunc.getFn();
                  tuples::ColumnRefAttr col = aggrFunc.getAttr(); // we dont need the tuplestream that is getRel here for now.
                  auto colName = getColumnName<tuples::ColumnRefAttr>(col);
                  auto tableName = getTableName<tuples::ColumnRefAttr>(col);

                  auto cudaRegIdentifier = LoadColumnIntoStream(streamCode, col, KernelType::Main);

                  // below assertions already handled in loadcolumninto stream no need of it here
                  // assert(streamCode->kernelArgs.find(tableName + "__" + colName) != streamCode->kernelArgs.end() && "existing column (input to aggregation) not found in kernel args.");

                  auto slot = newColumnMap[&regionOp];
                  assert(streamCode->kernelArgs.find(slot) != streamCode->kernelArgs.end() && "the new column is not in the kernel args.");

                  slot += "[" + buf_idx(op) + "]";

                  // the return values have one to one corr to computed cold
                  switch (fn) {
                     case relalg::AggrFunc::sum: {
                        streamCode->appendKernel("aggregate_sum(&" + slot + ", " + cudaRegIdentifier + ");");
                     } break;
                     case relalg::AggrFunc::count: {
                        streamCode->appendKernel("aggregate_count(&" + slot + ", " + cudaRegIdentifier + ");");
                     } break;
                     case relalg::AggrFunc::any: {
                        streamCode->appendKernel("aggregate_any(&" + slot + ", " + cudaRegIdentifier + ");");
                     } break;
                     case relalg::AggrFunc::avg: {
                        assert(false && "average should be split into sum and divide");
                     } break;
                     case relalg::AggrFunc::min: {
                        streamCode->appendKernel("aggregate_min(&" + slot + ", " + cudaRegIdentifier + ");");
                     } break;
                     case relalg::AggrFunc::max: {
                        streamCode->appendKernel("aggregate_max(&" + slot + ", " + cudaRegIdentifier + ");");
                     } break;
                     default:
                        assert(false && "this aggregation is not handled");
                        break;
                  }
               }
            }

            streamCode->appendKernel("return;");
            streamCode->appendControl(streamCode->launchKernel(KernelType::Main));
            kernelSchedule.push_back(streamCode);
            // any upstream op should start a new kernel.
            // for example the topk.
            streamCodeMap[op] = streamCode;
         } else if (auto table_scan = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
            std::string tableIdentifier = table_scan.getTableIdentifier().data();
            TupleStreamCode* streamCode = new TupleStreamCode(); // TODO(avinash, p3): clean up all allocated streams after printing them

            streamCode->stateCountArgs[tableIdentifier + "_size"] = "size_t";
            streamCode->stateArgs[tableIdentifier + "_size"] = "size_t";

            streamCode->baseRelation.push_back(tableIdentifier);
            streamCode->ridMap[tableIdentifier] = "tid";

            streamCode->appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
            streamCode->appendKernel("if (tid >= " + tableIdentifier + "_size) return;");

            streamCode->appendCountKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
            streamCode->appendCountKernel("if (tid >= " + tableIdentifier + "_size) return;");
            streamCodeMap[op] = streamCode;
         } else if (auto mapOp = llvm::dyn_cast<relalg::MapOp>(op)) {
            mlir::Operation* stream = mapOp.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) assert(false && "No downstream operation for map operation found");

            // TODO(avinash, p1/now): for all operations that do not start a tuplestream, check if parent is aggregation, and start a new kernel accordingly
            if (auto parentAgg = mlir::dyn_cast_or_null<relalg::AggregationOp>(stream)) {
							assert(false && "Implement this part");
            } else {
               auto computedCols = mapOp.getComputedCols();
               auto& predRegion = mapOp.getPredicate();
               if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predRegion.front().getTerminator())) {
                  assert(returnOp.getResults().size() == computedCols.size() && "Computed cols size not equal to result size");
                  auto i = 0ull;
                  // TODO(avinash): traverse the operation graph, until you find a getcolop or a constant, for each result
                  for (auto col : computedCols) {
                     auto colAttr = mlir::cast<tuples::ColumnDefAttr>(col);
                     auto table_name = getTableName<tuples::ColumnDefAttr>(colAttr);
                     auto column_name = getColumnName<tuples::ColumnDefAttr>(colAttr);
                     std::stringstream expr;
                     mapOpDfs(returnOp.getResults()[i].getDefiningOp(), streamCode, expr);
                     auto cudaIdentifier = "reg__" + table_name + "__" + column_name;
                     streamCode->appendKernel("auto " + cudaIdentifier + " = " + expr.str());
                     streamCode->loadedColumns.insert(cudaIdentifier);

                     streamCode->appendCountKernel("auto " + cudaIdentifier + " = " + expr.str());
                     streamCode->loadedCountColumns.insert(cudaIdentifier);
                     i++;
                  }
               } else {
                  assert(false && "No return op found for the map operation region");
               }
							streamCodeMap[op] = streamCode;
            }

         } else if (auto topKOp = llvm::dyn_cast<relalg::TopKOp>(op)) {
            unsigned int maxRows = topKOp.getMaxRows();
            auto sortSpecs = topKOp.getSortspecs();
            std::clog << "Max rows in topkop: " << maxRows << "\n";
            std::clog << "SORTSPECS\n";
            for (auto attr : sortSpecs) {
               auto sortspecAttr = mlir::cast<relalg::SortSpecificationAttr>(attr);
               tuples::ColumnRefAttr col = mlir::cast<tuples::ColumnRefAttr>(sortspecAttr.getAttr());

               std::clog << getTableName<tuples::ColumnRefAttr>(col) << " :: " << getColumnName<tuples::ColumnRefAttr>(col) << " ";
               if (sortspecAttr.getSortSpec() == relalg::SortSpec::desc) {
                  std::clog << "descending\n";
               } else {
                  std::clog << "ascending\n";
               }
            }
            // add thrust code to sort based on key and value, no kernel operations here
            // this is neither a pipeline starter or an ender, when to sort?, can we do a max
         } else if (auto joinOp = llvm::dyn_cast<relalg::InnerJoinOp>(op)) {
            // left side is a materialization point, so end the kernel and push it to the pipelineschedules
            // Generate 2 kernels one to get the count, and another to fill in the buffers
            mlir::Operation* leftStream = joinOp.getLeftMutable().get().getDefiningOp();
            TupleStreamCode* leftStreamCode = streamCodeMap[leftStream];
            if (!leftStreamCode) assert(false && "No downstream operation build side of hash join found");

            leftStreamCode->appendCountKernel("atomicAdd(" + BUF_IDX(op) + ", 1);");
            leftStreamCode->appendCountKernel("return;");
            //TODO(avinash): BUF_IDX ideally should be uint64_t, we are using int for atomic support for now
            leftStreamCode->stateCountArgs[BUF_IDX(op)] = "int *";

            // create the control code for count kernel
            leftStreamCode->appendCountControl("int *" + d_BUF_IDX(op) + ";");
            leftStreamCode->appendCountControl("cudaMalloc(&" + d_BUF_IDX(op) + ", sizeof(int));");
            leftStreamCode->appendCountControl("cudaMemset(" + d_BUF_IDX(op) + ",0 , sizeof(int));");
            leftStreamCode->appendCountControl(leftStreamCode->launchKernel(KernelType::Count));

            // increment the buffer idx, and insert tid into the buffer
            auto leftHash = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
            auto cudaIdentifierLeftKey = MakeKeysInStream(op, leftStreamCode, leftHash, KernelType::Main);
            leftStreamCode->stateArgs[BUF_IDX(op)] = "int *";
            leftStreamCode->stateArgs[BUF(op)] = "uint64_t *";
            leftStreamCode->stateArgs[HT(op)] = "HASHTABLE_INSERT";
            leftStreamCode->appendKernel("auto " + buf_idx(op) + " = atomicAdd(" + BUF_IDX(op) + ", 1);");
            leftStreamCode->appendKernel(HT(op) + ".insert(cuco::pair{" + cudaIdentifierLeftKey + ", " + buf_idx(op) + "});");
            int i = 0;
            for (auto br : leftStreamCode->baseRelation) {
               leftStreamCode->appendKernel(BUF(op) + "[" + buf_idx(op) + " * " + std::to_string(leftStreamCode->baseRelation.size()) + " + " + std::to_string(i) + "] = " + leftStreamCode->ridMap[br] + ";");
               i++;
            }
            leftStreamCode->appendKernel("return;"); // end the kernel
            // control for actual kernel
            // create the data structure for the actual kernel in control code
            // 1. copy d_BUF_ID to COUNT and clear it
            // 2. create a hash table of size COUNT
            // 3. create buffers of size COUNT*leftBaseRelations.size()
            // 4. launch kernel

            leftStreamCode->appendControl("int " + BUF_IDX(op) + ";");
            leftStreamCode->appendControl("cudaMemcpy(&" + BUF_IDX(op) + ", " + d_BUF_IDX(op) + ", sizeof(int), cudaMemcpyDeviceToHost);");
            leftStreamCode->appendControl("cudaMemset(" + d_BUF_IDX(op) + ",0 , sizeof(int));");
            leftStreamCode->appendControl("auto " + HT(op) + " = cuco::static_map{ " + BUF_IDX(op) + "* 2,cuco::empty_key{(int64_t)-1},cuco::empty_value{(int64_t)-1},thrust::equal_to<int64_t>{},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>()};");
            leftStreamCode->appendControl("uint64_t *" + d_BUF(op) + ";");
            leftStreamCode->appendControl("cudaMalloc(&" + d_BUF(op) + ", sizeof(uint64_t) * " + std::to_string(leftStreamCode->baseRelation.size()) + " * " + BUF_IDX(op) + ");");
            leftStreamCode->appendControl("cudaMemset(" + d_BUF(op) + ",0 , sizeof(int));");
            leftStreamCode->appendControl(leftStreamCode->launchKernel(KernelType::Main));

            kernelSchedule.push_back(leftStreamCode);

            // continue the right stream code gen
            // TODO(avinash): Add predicate region handling for the probe side

            mlir::Operation* rightStream = joinOp.getRightMutable().get().getDefiningOp();
            TupleStreamCode* rightStreamCode = streamCodeMap[rightStream];
            if (!rightStreamCode) assert(false && "No downstream operation probe side of hash join found");

            auto rightHash = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
            MakeKeysInStream(op, rightStreamCode, rightHash, KernelType::Count);
            auto cudaIdentifierRightKey = MakeKeysInStream(op, rightStreamCode, rightHash, KernelType::Main);
            rightStreamCode->stateCountArgs[BUF(op)] = "uint64_t*";
            rightStreamCode->stateCountArgs[HT(op)] = "HASHTABLE_FIND";
            rightStreamCode->stateArgs[BUF(op)] = "uint64_t*";
            rightStreamCode->stateArgs[HT(op)] = "HASHTABLE_FIND";
            rightStreamCode->appendCountKernel("auto " + SLOT(op) + " = " + HT(op) + ".find(" + cudaIdentifierRightKey + ");");
            rightStreamCode->appendCountKernel("auto " + buf_idx(op) + " = " + SLOT(op) + "->second;");
            rightStreamCode->appendKernel("auto " + SLOT(op) + " = " + HT(op) + ".find(" + cudaIdentifierRightKey + ");");
            rightStreamCode->appendKernel("auto " + buf_idx(op) + " = " + SLOT(op) + "->second;");
            i = 0;
            // emplace
            for (auto br : leftStreamCode->baseRelation) {
               auto rbr_beg = rightStreamCode->baseRelation.begin();
               rightStreamCode->baseRelation.emplace(rbr_beg + i, br);
               rightStreamCode->ridMap[br] = BUF(op) + "[" + buf_idx(op) + "*" + std::to_string(leftStreamCode->baseRelation.size()) + " + " + std::to_string(i) + "]";
               i++;
            }

            // upstream operator would use the probe side of the hashjoin
            streamCodeMap[op] = rightStreamCode;
         }
      });

      std::ofstream outputFile("output.cu");

      printAllTpchSchema(outputFile);
      for (auto code : kernelSchedule) {
         code->printKernel(KernelType::Count, outputFile);
         code->printKernel(KernelType::Main, outputFile);
      }

      outputFile << "void control() {\n";
      for (auto code : kernelSchedule) {
         code->printControl(KernelType::Count, outputFile);
         code->printControl(KernelType::Main, outputFile);
      }
      outputFile << "}\n";
      outputFile.close();
   }
};

/**
 * 1 = customer
 * 2 = orders 
 * 5 = lineitem
 * 3 = selection(customer, mktsegment = building)
 * 4 = selection(orders, orderdate < 1995-03-15)
 * 6 = selection(lineitem, shipd/ate > 1995-03-15)
 * 7 = join(orders, lineitem)
 * 8 = join
 */
}

std::unique_ptr<mlir::Pass> relalg::createPythonCodeGenPass() { return std::make_unique<PythonCodeGen>(); }
