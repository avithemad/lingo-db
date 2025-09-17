#ifndef LINGODB_COMPILER_DIALECT_RELALG_CUDA_CODE_GEN_HELPER_H
#define LINGODB_COMPILER_DIALECT_RELALG_CUDA_CODE_GEN_HELPER_H

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
void emitTimingEventCreation(std::ostream& outputFile);
bool isPrimaryKey(const std::set<std::string>& keysSet);
std::vector<std::string> split(std::string s, std::string delimiter);
extern bool gPrintHashTableSizes;

// -- [start] kernel timing code generation --

bool generateKernelTimingCode();
bool generatePerOperationProfile();
bool isProfiling();
bool usePartitionHashJoin();

// -- [end] kernel timing code generation --

// --- [start] different sized hash tables helpers ---

bool shouldGenerateSmallerHashTables();
std::string getHTKeyType(mlir::ArrayAttr keys);
std::string getHTValueType();
std::string getBufEltType();
std::string getBufIdxType();
std::string getBufIdxPtrType();
std::string getBufPtrType();

// --- [end] different sized hash tables helpers ---

namespace cudacodegen {
using namespace lingodb::compiler::dialect;

enum class KernelType {
   Main,
   Count,
   Main_And_Count
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

std::string GetId(const void* op);

std::string HT(const void* op);
std::string KEY(const void* op);
std::string SLOT(const void* op);
std::string BUF(const void* op);
std::string BUF_IDX(const void* op);
std::string buf_idx(const void* op);
std::string COUNT(const void* op);
std::string MAT(const void* op);
std::string MAT_IDX(const void* op);
std::string mat_idx(const void* op);
std::string slot_first(const void* op);
std::string slot_second(const void* op);
std::string BF(const void* op);
std::string SHUF_BUF_NAME(const void* op);
std::string SHUF_BUF_EXPR(const void* op);
std::string SHUF_BUF_VAL(const void* op);

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
   std::string tableName;
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

int daysSinceEpoch(const std::string& dateStr);
std::string mlirTypeToCudaType(const mlir::Type& ty);
std::string translateConstantOp(db::ConstantOp& constantOp);

typedef struct  {
   size_t cur_shuffle_id = 0;
   std::set<const mlir::Operation*> shuffleBufOps; // the expressions that need to be saved until now
   // for an op, we see if this op was already saved to the shuffle buffer. 
   // If saved, we just use the retrieved value for subsequent saves.
   // Else, we use the SLOT(op)->second value
   std::set<const mlir::Operation*> savedOps;
} ShuffleData;

// -- [Start] TupleStreamCode ---

class TupleStreamCode {
protected:
   std::vector<std::string> mainCode;
   std::vector<std::string> countCode;
   std::vector<std::string> controlCode;
   std::vector<std::string> controlDeclarations;
   int forEachScopes = 0;
   std::map<std::string, ColumnMetadata*> columnData;
   std::set<std::string> loadedColumns;
   std::set<std::string> loadedCountColumns;
   std::set<std::string> deviceFrees;
   std::vector<std::pair<mlir::Operation*, std::string>> profileInfo;
   std::set<std::string> hostFrees;

   std::map<std::string, std::string> mlirToGlobalSymbol; // used when launching the kernel.

   std::map<std::string, std::string> mainArgs;
   std::map<std::string, std::string> countArgs;
   bool m_hasInsertedSelection = false;
   bool m_genSelectionCheckUniversally = true;
   ShuffleData m_shuffleData;
   int id;

   void appendKernel(std::string stmt, KernelType ty = KernelType::Main_And_Count) {
      if (ty == KernelType::Main)
         mainCode.push_back(stmt);
      else if (ty == KernelType::Count)
         countCode.push_back(stmt);
      else if (ty == KernelType::Main_And_Count) {
         mainCode.push_back(stmt);
         countCode.push_back(stmt);
      } else {
         assert(false && "Unknown kernel type");
      }
   }

   void appendControlDecl(std::string stmt) {
      controlDeclarations.push_back(stmt);
   }

   void appendControl(std::string stmt) {
      controlCode.push_back(stmt);
   }

   std::string getKernelName(KernelType ty) {
      if (ty == KernelType::Main)
         return "main";
      else
         return "count";
   }

   virtual std::string launchKernel(KernelType ty) = 0;

   void genLaunchKernel(KernelType ty) {
      static bool shouldGeneratePerKernelTiming = false;
      if (generateKernelTimingCode() && shouldGeneratePerKernelTiming)
         appendControl("cudaEventRecord(start);");
      appendControl(launchKernel(ty));
      if (generateKernelTimingCode() && shouldGeneratePerKernelTiming) {
         appendControl("cudaEventRecord(stop);");
         auto kernelName = getKernelName(ty) + "_" + GetId((void*) this);
         auto kernelTimeVarName = kernelName + "_time";
         appendControl("float " + kernelTimeVarName + ";");
         appendControl(fmt::format("cudaEventSynchronize(stop);"));
         appendControl(fmt::format("cudaEventElapsedTime(&{0}, start, stop);", kernelTimeVarName));
         appendControl(fmt::format("std::cout << \"{0}\" << \", \" << {1} << std::endl;", kernelName, kernelTimeVarName));
      }
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

   std::string getKernelSizeVariable() {
      for (auto it : mainArgs)
         if (it.second == "size_t") return it.first;
      assert(false && "this kernel is supposed to have a size parameter");
      return "";
   }

   void printHashTableSize(const std::string& count_var, const std::string& key_size, const std::string& value_size, const std::string& load_factor, mlir::Operation *op) {
      if (gPrintHashTableSizes) {
         auto rightHashAttr = op->getAttrOfType<mlir::ArrayAttr>("rightHash");
         auto leftHashAttr = op->getAttrOfType<mlir::ArrayAttr>("leftHash");
         
         // Convert ArrayAttr to string representation
         std::string rightHashStr = "null";
         std::string leftHashStr = "null";
         
         if (rightHashAttr) {
            rightHashStr = "[";
            for (size_t i = 0; i < rightHashAttr.size(); ++i) {
               if (auto colRefAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(rightHashAttr[i])) {
                  ColumnDetail detail(colRefAttr);
                  rightHashStr += detail.getMlirSymbol();
               } else {
                  rightHashStr += "unknown";
               }
               if (i < rightHashAttr.size() - 1) rightHashStr += ", ";
            }
            rightHashStr += "]";
         }
         
         if (leftHashAttr) {
            leftHashStr = "[";
            for (size_t i = 0; i < leftHashAttr.size(); ++i) {
               if (auto colRefAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(leftHashAttr[i])) {
                  ColumnDetail detail(colRefAttr);
                  leftHashStr += detail.getMlirSymbol();
               } else {
                  leftHashStr += "unknown";
               }
               if (i < leftHashAttr.size() - 1) leftHashStr += ", ";
            }
            leftHashStr += "]";
         }

         auto kernel_name = "main_" +  GetId((void*) this);
         appendControl(fmt::format("if (runCountKernel) {{ std::cout << \"-- HT Size: \" << (uint32_t)({0} * {1}) * (sizeof({2}) + sizeof({3})) << \" bytes, Count: \" << {0} << \", ID: {9}, Op: {6}, OpId : {7}, Left: {4}, Right: {5}, Kernel: {8} --\" << std::endl;", count_var, load_factor, key_size, value_size, leftHashStr, rightHashStr, op->getName().getStringRef().str(), GetId((void*) op), kernel_name, HT((void*) op)));
         appendControl(fmt::format("std::cout << \"-- HT_Build: {0}, Op: {1}, OpId: {2}, Kernel: {3} --\" << std::endl; }}", HT((void*) op), op->getName().getStringRef().str(), GetId((void*) op), kernel_name));
      }
   }

   void printProbeHashTableEntry(mlir::Operation* op) {
      if (!gPrintHashTableSizes)
         return;
      auto kernel_name = "main_" +  GetId((void*) this);
      appendControl(fmt::format("if (runCountKernel) std::cout << \"-- HT_Probe: {0}, Op:  {1}, OpId : {2}, Kernel: {3} --\" << std::endl;", HT((void*) op), op->getName().getStringRef().str(), GetId((void*) op), kernel_name));
   }

   void printBufferSize(const std::string& size_str, mlir::Operation* op)
   {
      if (!gPrintHashTableSizes)
         return;
      auto kernel_name = "main_" +  GetId((void*) this);
      appendControl(fmt::format("if (runCountKernel) std::cout << \"-- Buffer Size: \" << {0} << \" bytes, Op:  {1}, OpId : {2}, Kernel: {3} --\" << std::endl;", size_str, op->getName().getStringRef().str(), GetId((void*) op), kernel_name));
   }
public:
   void printControlDeclarations(std::ostream& stream) {
      for (auto line : controlDeclarations) {
         stream << line << std::endl;
      }
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

}

// --- [start] code generation switches helpers ---

void removeCodeGenSwitch(int& argc, char** argv, int i);
void checkForBenchmarkSwitch(int& argc, char** argv);
void checkForCodeGenSwitches(int& argc, char** argv);

// --- [end] code generation switches helpers ---

// --- [start] Pyper ---
extern bool gPyperShuffle; // TODO: Move to a getter
extern bool gThreadsAlwaysAlive; // TODO: Move to a getter
// -- [end] Pyper ---
extern bool gUseBloomFiltersForJoin; // TODO: Move to a getter
extern bool gShuffleAllOps; // TODO: Move to a getter

bool isPrimaryKey(const std::set<std::string>& keysSet);
bool invertJoinIfPossible(std::set<std::string>& rightkeysSet, bool left_pk);
void emitTimingEventCreation(std::ostream& outputFile);

#endif // LINGODB_COMPILER_DIALECT_RELALG_CUDA_CODE_GEN_HELPER_H