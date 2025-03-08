
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
#include <vector>
namespace {
using namespace lingodb::compiler::dialect;

struct TupleStreamCode {
   std::string kernelCode;
   TupleStreamCode() : kernelCode("") {}

   void appendKernel(std::string code) {
      kernelCode += code + "\n";
   }
   void print() {
      std::cout << kernelCode << "\n";
   }
};

static std::vector<std::pair<int, mlir::Value>> translateSelection(mlir::Region& predicate) {
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
         if (auto compareOp = mlir::dyn_cast_or_null<db::CmpOp>(matched.getDefiningOp())){
            auto cmp = compareOp.getPredicate();
            auto left = compareOp.getLeft();
            auto right = compareOp.getRight();
            
            switch (cmp)
            {
            case db::DBCmpPredicate::eq:
               /* code */
            {

            }
            break;
            case db::DBCmpPredicate::neq: 
            {

            }
            case db::DBCmpPredicate::lt: 
            {

            }
            case db::DBCmpPredicate::gt: 
            {

            }
            case db::DBCmpPredicate::lte: 
            {

            }
            case db::DBCmpPredicate::gte: 
            {

            }
            case db::DBCmpPredicate::isa: 
            {
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
         return std::vector<std::pair<int, mlir::Value>>();
      }
   }
   return std::vector<std::pair<int, mlir::Value>>();
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
            auto conditions = translateSelection(predicateRegion);
            for (auto &c: conditions) {
               std::cout << "condition: " << c.first << std::endl;
               c.second.dump();
            }
            std::cout << std::endl;

               streamCode->appendKernel("if (!predicate) return;");
            streamCodeMap[op] = streamCode;
            /**
                 * TODO(avinash): check if the implemented predicate in python is good
                 *     after re-evaluating its design, handle the predicate by the region: getPredicate
                 */

            //this is basically produce code for the scan
         }
         if (auto aggregation = llvm::dyn_cast<relalg::AggregationOp>(op)) {
            // this would be the consume method for the aggregation
            // since it is a materializing operator, we need to
            // end the kernel of the operand,
            /**
                 * This is a materializing operation.
                 * Get the keys for aggregation and the tuplestream
                 */
            mlir::Operation* stream = aggregation.getRelMutable().get().getDefiningOp();
            TupleStreamCode* streamCode = streamCodeMap[stream];

            streamCode->appendKernel("ht_id.insert(keys, aggregate on the slot);");
            streamCode->appendKernel("return;");
            kernelSchedule.push_back(streamCode);
            streamCodeMap[op] = streamCode;
         }
         if (auto table_scan = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
            std::string tableIdentifier = table_scan.getTableIdentifier().data();
            TupleStreamCode* streamCode = new TupleStreamCode();
            streamCode->appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
            streamCode->appendKernel("if (tid >= " + tableIdentifier + "_size) return;");
            streamCodeMap[op] = streamCode;
         }
         if (auto mapOp = llvm::dyn_cast<relalg::MapOp>(op)) {
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
            leftStreamCode->appendKernel("ht_id.insert(keys, tid);");
            leftStreamCode->appendKernel("return;"); // end the kernel
            kernelSchedule.push_back(leftStreamCode);

            // continue the right stream code gen
            mlir::Operation* rightStream = joinOp.getRightMutable().get().getDefiningOp();
            TupleStreamCode* rightStreamCode = streamCodeMap[rightStream];
            rightStreamCode->appendKernel("auto slot_id = ht_id.probe(keys);");

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
