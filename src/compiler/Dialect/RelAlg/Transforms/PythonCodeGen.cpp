
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <map>
#include <set>
#include <vector>
namespace {
using namespace lingodb::compiler::dialect;

typedef std::map<std::string, std::string> RIDMAP;
typedef std::set<std::string> LOADEDCOLUMNS;
struct TupleStreamCode {
   std::string baseRelation; // in data centric code gen, each stream will have exactly one base relation where it scans from
   std::string kernelCode;
   RIDMAP ridMap; // row identifier map. maps table to cuda identifier containing the RID of the table
   // TODO(avinash) : insert any of the columns that have been loaded into the registers in the below set
   LOADEDCOLUMNS loadedColumns;
   TupleStreamCode() : kernelCode("") {}

   void appendKernel(std::string code) {
      kernelCode += code + "\n";
   }
   void print() {
      std::cout << kernelCode << "\n";
   }
};

struct ColumnDetail {
   std::string name;
   std::string relation;
   mlir::Type type;
   ColumnDetail(std::string name, std::string relation, mlir::Type type) : name(name), relation(relation), type(type) {}
   void print() {
      std::clog << relation << "->" << name << " : ";
      type.dump();
      std::cout << std::endl;
   }
   static std::string getTableName(const tuples::ColumnRefAttr &colAttr)
   {
      return colAttr.getName().getRootReference().str();
   
   }
   static std::string getColumnName(const tuples::ColumnRefAttr &colAttr)
   {
      for (auto n: colAttr.getName().getNestedReferences())
      {
         return n.getAttr().str();
      }
      // should never reach here ideally
      assert(false && "No column for columnrefattr found");
      return "";
   }
   ColumnDetail(const tuples::ColumnRefAttr &colAttr) {
      relation = getTableName(colAttr);
      name = getColumnName(colAttr);
      type = colAttr.getColumn().type;
   }
};

std::string LoadColumnIntoStream(TupleStreamCode *streamCode, const tuples::ColumnRefAttr &colAttr) {
   ColumnDetail detail(colAttr);
   std::string cudaIdentifier = "reg__" + detail.relation + "__" + detail.name; 
   if (streamCode->loadedColumns.find(cudaIdentifier) == streamCode->loadedColumns.end()) {
      // load the column into register 
      streamCode->loadedColumns.insert(cudaIdentifier);
      streamCode->appendKernel("auto reg__" + detail.relation + "__" + detail.name + " = " + detail.relation + "__" + detail.name + "[" + streamCode->ridMap[detail.relation] + "];");
   }
   assert(streamCode->loadedColumns.find(cudaIdentifier) != streamCode->loadedColumns.end());
   return cudaIdentifier;
}
std::string operandToString(mlir::Operation* operand) {
   std::string result = "";
   if (auto constantOp = mlir::dyn_cast_or_null<db::ConstantOp>(operand)) {
      if (auto integerAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(constantOp.getValue())) {
         result = std::to_string(integerAttr.getInt());
      } else if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(constantOp.getValue())) {
         result = std::to_string(floatAttr.getValueAsDouble());
      } else if (auto stringAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(constantOp.getValue())) {
         result = "\"" + stringAttr.str() + "\"";
      } else {
         assert(false && "Unknown constant type");
      }
   } else {
      assert(false && "Unhandled operand for selection translation");
   }
   return result;
}

static std::string translateSelection(mlir::Region& predicate, TupleStreamCode *streamCode) {
   auto terminator = mlir::cast<tuples::ReturnOp>(predicate.front().getTerminator());
   if (!terminator.getResults().empty()) {
      auto& predicateBlock = predicate.front();
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predicateBlock.getTerminator())) {
         mlir::Value matched = returnOp.getResults()[0];
         std::vector<std::pair<int, mlir::Value>> conditions;
         /**
          * TODO(avinash) : Handle other conditions
          */
         // hoping that we always have a compare in selection
         if (auto compareOp = mlir::dyn_cast_or_null<db::CmpOp>(matched.getDefiningOp())) {
            // TODO(avinash): convert the string to py arrow date integer, if the typeof column is datetime (the other operand)
            auto left = compareOp.getLeft();
            std::string leftOperand; 
            if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(left.getDefiningOp())) {
               leftOperand = LoadColumnIntoStream(streamCode, getColOp.getAttr());
            } else {
               leftOperand = operandToString(left.getDefiningOp());
            }

            auto right = compareOp.getRight();
            std::string rightOperand;
            if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(right.getDefiningOp())) {
               rightOperand = LoadColumnIntoStream(streamCode, getColOp.getAttr());
            } else {
               rightOperand = operandToString(right.getDefiningOp());
            }

            auto cmp = compareOp.getPredicate();
            switch (cmp) {
               case db::DBCmpPredicate::eq:
                  /* code */
                  {
                     return leftOperand + " == " + rightOperand;
                  }
                  break;
               case db::DBCmpPredicate::neq: {
                  return leftOperand + " != " + rightOperand;
               } break;
               case db::DBCmpPredicate::lt: {
                  return leftOperand + " < " + rightOperand;
               } break;
               case db::DBCmpPredicate::gt: {
                  return leftOperand + " > " + rightOperand;

               } break;
               case db::DBCmpPredicate::lte: {
                  return leftOperand + " <= " + rightOperand;
               } break;
               case db::DBCmpPredicate::gte: {
                  return leftOperand + " >= " + rightOperand;

               } break;
               case db::DBCmpPredicate::isa: {
                  assert(false && "should not happen");
               }
               default:
                  break;
            }
         } else if (auto compareOp = mlir::dyn_cast_or_null<db::RuntimeCall>(matched.getDefiningOp())) { // or a like operation
            std::clog << "TODO: handle runtime predicates\n";
         }
      } else {
         assert(false && "invalid");
      }
   }
   return "";
}

std::string convertToHex(TupleStreamCode* stream)
{
   std::stringstream sstream;
   sstream << std::hex << (unsigned long long)(void*)stream;
   std::string result = sstream.str();
   return result;
}
static std::string HT(TupleStreamCode* stream)
{
   return "HT_" + convertToHex(stream);
}
static std::string KEY(TupleStreamCode* stream)
{
   return "KEY_" + convertToHex(stream);
}
static std::string SLOT(TupleStreamCode* stream)
{
   return "SLOT_" + convertToHex(stream);
}
static std::string MakeKeysInStream(TupleStreamCode* stream, const mlir::ArrayAttr &keys) {
   std::string keyMakerString = ("int64_t " + KEY(stream) + " = make_keys(");
   for (auto i = 0ull; i<keys.size(); i++) {
      tuples::ColumnRefAttr key = mlir::cast<tuples::ColumnRefAttr>(keys[i]);
      std::string cudaIdentifierKey = LoadColumnIntoStream(stream, key);
      keyMakerString += cudaIdentifierKey;
      if (i != keys.size() - 1)
      {
         keyMakerString += ", ";
      }
      else {
         keyMakerString += ")";
      }
   }
   stream->appendKernel(keyMakerString);
   return KEY(stream);
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

   void runOnOperation() override {
      getOperation().walk([&](mlir::Operation* op) {
         if (auto selection = llvm::dyn_cast<relalg::SelectionOp>(op)) {
            ::mlir::Operation* stream = selection.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];

            // Get the predicate region
            ::mlir::Region& predicateRegion = selection.getPredicate();
            auto condition = translateSelection(predicateRegion, streamCode);

            streamCode->appendKernel("if (!(" + condition + ")) return;");
            streamCodeMap[op] = streamCode;
            /**
                 * TODO(avinash): check if the implemented predicate in python is good
                 *     after re-evaluating its design, handle the predicate by the region: getPredicate
                 */

            //this is basically produce code for the scan
         }
         if (auto aggregation = llvm::dyn_cast<relalg::AggregationOp>(op)) {
            // TODO(avinash): scheduled on 10th march
            // this would be the consume method for the aggregation
            // since it is a materializing operator, we need to
            // end the kernel of the operand,
            /**
                 * This is a materializing operation.
                 * Get the keys for aggregation and the tuplestream
                 */
            mlir::Operation* stream = aggregation.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];

            mlir::ArrayAttr groupByKeys = aggregation.getGroupByCols();
            MakeKeysInStream(streamCode, groupByKeys);

            //TODO(avinash): work on the aggregation functions;

            // auto Hash = joinOp->getAttrOfType<mlir::ArrayAttr>("Hash");
            // auto cudaIdentifierKey = MakeKeysInStream(streamCode, Hash);

            streamCode->appendKernel("ht_id.insert(keys, aggregate on the slot);");
            streamCode->appendKernel("return;");
            kernelSchedule.push_back(streamCode);
            streamCodeMap[op] = streamCode;
         }
         if (auto table_scan = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
            std::string tableIdentifier = table_scan.getTableIdentifier().data();
            TupleStreamCode* streamCode = new TupleStreamCode();
            streamCode->baseRelation = tableIdentifier;
            streamCode->ridMap[tableIdentifier] = "tid";
            streamCode->appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
            streamCode->appendKernel("if (tid >= " + tableIdentifier + "_size) return;");
            streamCodeMap[op] = streamCode;
         }
         if (auto mapOp = llvm::dyn_cast<relalg::MapOp>(op)) {
            // TODO(avinash): Scheduled on 9th march
            mlir::Operation* stream = mapOp.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];
            streamCode->appendKernel("auto mapped_register = expression(attributes, constants, operations);");
            streamCodeMap[op] = streamCode;
         }
         if (auto joinOp = llvm::dyn_cast<relalg::InnerJoinOp>(op)) {
            // left side is a materialization point, so end the kernel and push it to the pipelineschedules
            // Generate 2 kernels one to get the count, and another to fill in the buffers
            mlir::Operation* leftStream = joinOp.getLeftMutable().get().getDefiningOp();
            TupleStreamCode* leftStreamCode = streamCodeMap[leftStream];
            auto leftHash = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
            auto cudaIdentifierLeftKey = MakeKeysInStream(leftStreamCode, leftHash);
            // load keys into the register
            leftStreamCode->appendKernel(HT(leftStreamCode) + ".insert(cuco::pair(" + cudaIdentifierLeftKey + ", tid));");
            leftStreamCode->appendKernel("return;"); // end the kernel
            kernelSchedule.push_back(leftStreamCode);
            
            // continue the right stream code gen
            mlir::Operation* rightStream = joinOp.getRightMutable().get().getDefiningOp();
            TupleStreamCode* rightStreamCode = streamCodeMap[rightStream];
            auto rightHash = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
            auto cudaIdentifierRightKey = MakeKeysInStream(rightStreamCode, rightHash);
            rightStreamCode->appendKernel("auto " + SLOT(rightStreamCode) + " = " + HT(leftStreamCode) + ".find(" + cudaIdentifierRightKey +");");
            // add the left table's rid to the rid map which is SLOT(rightStreamCode)->second
            rightStreamCode->ridMap[leftStreamCode->baseRelation] = SLOT(rightStreamCode) + "->second";
            // upstream operator would use the probe side of the hashjoin
            streamCodeMap[op] = rightStreamCode;
         }
      });
      for (auto code : kernelSchedule) {
         code->print();
      }
   }
};

/**
 * 1 = customer
 * 2 = orders 
 * 5 = lineitem
 * 3 = selection(customer, mktsegment = building)
 * 4 = selection(orders, orderdate < 1995-03-15)
 * 6 = selection(lineitem, shipdate > 1995-03-15)
 * 7 = join(orders, lineitem)
 * 8 = join
 */
}

std::unique_ptr<mlir::Pass> relalg::createPythonCodeGenPass() { return std::make_unique<PythonCodeGen>(); }
