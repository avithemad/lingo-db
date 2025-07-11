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

// -- [start] kernel timing code generation --

bool generateKernelTimingCode();
bool generatePerOperationProfile();

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

int daysSinceEpoch(const std::string& dateStr);
std::string mlirTypeToCudaType(const mlir::Type& ty);
std::string translateConstantOp(db::ConstantOp& constantOp);

}

// --- [start] code generation switches helpers ---

void removeCodeGenSwitch(int& argc, char** argv, int i);
void checkForBenchmarkSwitch(int& argc, char** argv);
void checkForCodeGenSwitches(int& argc, char** argv);

// --- [end] code generation switches helpers ---

// --- [start] Pyper ---
extern bool gGeneratingShuffles; // TODO: Move to a getter
extern bool gGeneratingNestedCode; // TODO: Move to a getter
// -- [end] Pyper ---
extern bool gUseBloomFiltersForJoin; // TODO: Move to a getter
extern bool gCompilingSSB; // TODO: Move to a getter

bool isPrimaryKey(const std::set<std::string>& keysSet);
bool invertJoinIfPossible(std::set<std::string>& rightkeysSet, bool left_pk);
void emitTimingEventCreation(std::ostream& outputFile);

#endif // LINGODB_COMPILER_DIALECT_RELALG_CUDA_CODE_GEN_HELPER_H