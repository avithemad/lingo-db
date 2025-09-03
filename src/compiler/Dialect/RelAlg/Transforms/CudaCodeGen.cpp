#include "lingodb/compiler/Dialect/RelAlg/CudaCodeGenHelper.h"
using namespace lingodb::compiler::dialect;

static bool gCudaCodeGenEnabled = false;
static bool gCudaCodeGenNoCountEnabled = false;
static bool gCudaCrystalCodeGenEnabled = false;
static bool gCudaCrystalCodeGenNoCountEnabled = false;

using NameTypePairs = std::vector<std::pair<std::string, std::string>>;

namespace cudacodegen {

static int StreamId = 0;

struct PartitionHashJoinResultInfo {
   std::set<std::string> joinKeySet;
   std::string resultVar;
   std::string resultType;
   std::string leftTableName;
   std::string rightTableName;
   std::string leftRowId;
   std::string  rightRowId;
};

std::string getGlobalSymbolName(const std::string& tableName, const std::string& columnName) {
   return fmt::format("d_{0}__{1}", tableName, columnName);
}

class HyperTupleStreamCode : public TupleStreamCode {
   std::map<KernelType, size_t> m_threadActiveScopeCount;
   std::map<mlir::Operation*, std::string> m_joinResultVariables;
   PartitionHashJoinResultInfo m_joinInfo;
   std::string launchKernel(KernelType ty) override {
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
   HyperTupleStreamCode(relalg::BaseTableOp& baseTableOp) {
      std::string tableName = baseTableOp.getTableIdentifier().data();
      std::string tableSize = tableName + "_size";
      mlirToGlobalSymbol[tableSize] = tableSize;
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only

      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
      appendKernel(fmt::format("if (tid >= {}) return;", tableSize));

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
   HyperTupleStreamCode(relalg::AggregationOp aggOp) {
      mlir::Operation *op = aggOp.getOperation();
      std::string tableSize = COUNT(op);

      mlirToGlobalSymbol[tableSize] = tableSize;
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only

      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
      appendKernel(fmt::format("if (tid >= {0}) return;", tableSize));

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
   HyperTupleStreamCode(relalg::InnerJoinOp joinOp, const PartitionHashJoinResultInfo& resultInfo, const HyperTupleStreamCode* leftStreamCode, const HyperTupleStreamCode* rightStreamCode) {
      assert(usePartitionHashJoin() && "Should be called only when partition hash join is used.");
      mlir::Operation *op = joinOp.getOperation();
      std::string tableSize = COUNT(op);
      m_joinInfo = resultInfo;

      mlirToGlobalSymbol[tableSize] = tableSize;
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only
      mlirToGlobalSymbol[resultInfo.leftRowId] = fmt::format("{0}.get_typed_ptr<1>()", resultInfo.resultVar);
      mlirToGlobalSymbol[resultInfo.rightRowId] = fmt::format("{0}.get_typed_ptr<2>()", resultInfo.resultVar);
      mainArgs[resultInfo.leftRowId] = mainArgs[resultInfo.rightRowId] = countArgs[resultInfo.leftRowId] = countArgs[resultInfo.rightRowId] = "uint64_t*";

      appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
      appendKernel(fmt::format("if (tid >= {0}) return;", tableSize));

      auto groupByKeys = leftStreamCode->columnData;
      for (auto& symbolDataPair : leftStreamCode->columnData) {
         auto mlirSymbol = symbolDataPair.first;
         auto globalSymbol = fmt::format("d_{0}", mlirSymbol);
         mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
         ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
         metadata->rid = fmt::format("{}[tid]", resultInfo.leftRowId);
         columnData[mlirSymbol] = metadata;
      }
      for (auto& symbolDataPair : rightStreamCode->columnData) {
         auto mlirSymbol = symbolDataPair.first;
         auto globalSymbol = fmt::format("d_{0}", mlirSymbol);
         mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
         ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
         metadata->rid = fmt::format("{}[tid]", resultInfo.rightRowId);
         columnData[mlirSymbol] = metadata;
      }
      id = StreamId;
      StreamId++;
      return;
   }
   ~HyperTupleStreamCode() {
      for (auto p : columnData) delete p.second;
   }
   void RenamingOp(relalg::RenamingOp renamingOp) {
      for (mlir::Attribute attr : renamingOp.getColumns()) {
         auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
         mlir::Attribute from = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting())[0];
         auto relationRefAttr = mlir::dyn_cast_or_null<tuples::ColumnRefAttr>(from);
         ColumnDetail detailRef(relationRefAttr), detailDef(relationDefAttr);
         auto colData = columnData[detailRef.getMlirSymbol()];
         std::string ty = mlirTypeToCudaType(detailRef.type);
         if (colData == nullptr && mlirTypeToCudaType(detailRef.type) == "DBStringType") {
            colData = columnData[detailRef.getMlirSymbol() + "_encoded"];
            ty = "DBI16Type";
         }
         if (colData == nullptr) {
            assert(false && "Renaming op: column ref not in tuple stream");
         }
         mainArgs[detailRef.getMlirSymbol()] = ty + "*";
         countArgs[detailRef.getMlirSymbol()] = ty + "*";
         auto newSymbol = mlirTypeToCudaType(detailDef.type) == "DBStringType" ? detailDef.getMlirSymbol() + "_encoded" : detailDef.getMlirSymbol();
         columnData[newSymbol] = new ColumnMetadata(colData);
         columnData[detailDef.getMlirSymbol()] = new ColumnMetadata(colData);
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
      if (mlirSymbol != colData->loadExpression) {
         return cudaId;
      }
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
         int i = 0;
         for (auto v : runtimeOp.getArgs()) {
            if ((i == 1) && (function == "Like")) {
               // remove first and last character from the string,
               std::string likeArg = SelectionOpDfs(v.getDefiningOp());
               if (likeArg[0] == '\"' && likeArg[likeArg.size() - 1] == '\"') {
                  likeArg = likeArg.substr(1, likeArg.size() - 2);
               }
               std::vector<std::string> tokens = split(likeArg, "%");
               std::string patternArray = "", sizeArray = "";
               std::clog << "TOKENS: ";
               for (auto t : tokens) std::clog << t << "|";
               std::clog << std::endl;
               int midpatterns = 0;
               if (tokens.size() <= 2) {
                  patternArray = "nullptr";
                  sizeArray = "nullptr";
               } else {
                  std::string t1 = "";
                  for (size_t i = 1; i < tokens.size() - 1; i++) {
                     patternArray += t1 + fmt::format("\"{}\"", tokens[i]);
                     sizeArray += t1 + std::to_string(tokens[i].size());
                     t1 = ", ";
                     midpatterns++;
                  }
               }
               std::string patarr = patternArray == "nullptr" ? "nullptr" : fmt::format("(const char*[]){{ {0} }}", patternArray);
               std::string sizearr = sizeArray == "nullptr" ? "nullptr" : fmt::format("(const int[]){{ {0} }}", sizeArray);
               args += sep + fmt::format("\"{0}\", \"{1}\", {2}, {3}, {4}", tokens[0], tokens[tokens.size() - 1], patarr, sizearr, midpatterns);
               break;
            } else {
               args += sep + SelectionOpDfs(v.getDefiningOp());
               sep = ", ";
            }
            i++;
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
      } else if (auto castOp = mlir::dyn_cast_or_null<db::CastOp>(op)) {
         auto val = castOp.getVal();
         return fmt::format("(({1}){0})", SelectionOpDfs(val.getDefiningOp()), mlirTypeToCudaType(castOp.getRes().getType()));
      }
      op->dump();
      assert(false && "Selection predicate not handled");
      return "";
   }
   bool shouldUseThreadsAliveCodeGen() {
      return gThreadsAlwaysAlive && forEachScopes == 0; // we are not inside a forEach lambda of a multimap
   }

   // --- [start] pyper shuffle helpers ---
   bool shouldGenerateShuffle() {
      // we can only generate shuffle when threads always alive is enabled
      // AND we are not inside a forEach lambda of a multimap
      return gGeneratingShuffles && shouldUseThreadsAliveCodeGen();
   }
   void saveOpToShuffleBuffer(const mlir::Operation *op) {
      assert(shouldGenerateShuffle() && "saveOpToShuffleBuffer can only be called when shuffles are enabled");
      m_shuffleData.shuffleBufOps.insert(op);
   }
   void saveCurStateToShuffleBuffer() {
      assert(shouldGenerateShuffle() && "saveCurStateToShuffleBuffer can only be called when shuffles are enabled");      
      for (auto& op : m_shuffleData.shuffleBufOps) {
         if (m_shuffleData.savedOps.find(op) == m_shuffleData.savedOps.end())
            appendKernel(fmt::format("{0} = {1};", SHUF_BUF_VAL(op), SHUF_BUF_EXPR(op)));
      }
      appendKernel("threadActive = true;");
      closeThreadActiveScopes();
      startThreadActiveScope("ShouldShuffle(threadActive)");
      startThreadActiveScope("threadActive");
      appendKernel("// Save current state to shuffle buffer");      
      appendKernel(fmt::format("auto shuffle_slot = atomicAdd_block(&shuffle_buf_idx[{0}], 1);", m_shuffleData.cur_shuffle_id));
      appendKernel(fmt::format("shuffle_buf_tid[shuffle_slot] = tid;"));
      for (auto& op : m_shuffleData.shuffleBufOps) {
         auto slotValue = (m_shuffleData.savedOps.find(op) != m_shuffleData.savedOps.end()) ? slot_or_shuf_val(op) : SHUF_BUF_EXPR(op);
         appendKernel(fmt::format("{0}[shuffle_slot] = {1};", SHUF_BUF_NAME(op), slot_or_shuf_val(op)));
         m_shuffleData.savedOps.insert(op); // mark this op as saved
      }
      closeThreadActiveScopes(1); // close the threadActive scope
   }
   void retrieveCurStateFromShuffleBuffer() {
      assert(shouldGenerateShuffle() && "retrieveCurStateFromShuffleBuffer can only be called when shuffles are enabled");
      appendKernel(fmt::format("INVALIDATE_IF_THREAD_BEYOND_SHUFFLE({0});", m_shuffleData.cur_shuffle_id++));
      appendKernel("// Retrieve current state from shuffle buffer");
      startThreadActiveScope("shuffle_valid");
      loadedColumns.clear(); // The tid has changed. The columns need to be reloaded.
      loadedCountColumns.clear();
      appendKernel("threadActive = true;");
      appendKernel(fmt::format("tid = shuffle_buf_tid[threadIdx.x];"));
      for (auto& op : m_shuffleData.shuffleBufOps) {
         appendKernel(fmt::format("{0} = {1}[threadIdx.x];", SHUF_BUF_VAL(op), SHUF_BUF_NAME(op)));
      }      
      closeThreadActiveScopes(-1);
      appendKernel("__syncthreads();");
      closeThreadActiveScopes();
      startThreadActiveScope("threadActive");
      appendKernel("threadActive = false;");
   }
   void genShuffleThreads() {
      assert(shouldGenerateShuffle() && "genShuffleThreads can only be called when shuffles are enabled");
      saveCurStateToShuffleBuffer();
      retrieveCurStateFromShuffleBuffer();      
   }
   std::string slot_or_shuf_val(const mlir::Operation* op) {
      if (shouldGenerateShuffle()) {
         return SHUF_BUF_VAL(op); // also check for the attribute on the op if we are selectively inserting shuffles.
      } else {
         return fmt::format("{0}->second", SLOT(op));
      }
   }
   // --- [end] pyper shuffle helpers ---   
   void AddSelectionPredicate(mlir::Region& predicate, float selectivity = 0.5f) {
      auto terminator = mlir::cast<tuples::ReturnOp>(predicate.front().getTerminator());
      if (!terminator.getResults().empty()) {
         auto& predicateBlock = predicate.front();
         if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predicateBlock.getTerminator())) {
            mlir::Value matched = returnOp.getResults()[0];
            std::string condition = SelectionOpDfs(matched.getDefiningOp());
            if (condition == "!(false)" || condition == "true")
               return; // This is a null check op. No-op for now
            if (shouldUseThreadsAliveCodeGen())
            {
               startThreadActiveScope(condition);
               if (shouldGenerateShuffle() && selectivity <= 1.0f)
                  return genShuffleThreads();
               else
                  return;
            } else {
               appendKernel(fmt::format("if (!({0})) return;", condition));
               return;
            }
         } else {
            assert(false && "expected return op to be in the end of the predicate region");
         }
      }
      predicate.front().dump();
      assert(false && "Predicate is not implemented");
      return;
   }
   void MaterializeCount(mlir::Operation* op, const std::string& suffix="") {
      if (usePartitionHashJoin() && mlir::isa<relalg::InnerJoinOp>(op)) return; // partition hash-join materializes count at the end.

      std::string countVarName = COUNT(op) + suffix;
      countArgs[countVarName] = "uint64_t*";
      mlirToGlobalSymbol[countVarName] = fmt::format("d_{}", countVarName);
      appendKernel("// Materialize count", KernelType::Count);
      
      appendKernel(fmt::format("atomicAdd((int*){0}, 1);", countVarName), KernelType::Count);

      appendControl("// Materialize count");
      appendControlDecl(fmt::format("uint64_t* d_{0} = nullptr;", countVarName));
      if (!isProfiling())
         appendControl("if (runCountKernel){\n");
      appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof(uint64_t));", countVarName));
      deviceFrees.insert(fmt::format("d_{0}", countVarName));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", countVarName));
      genLaunchKernel(KernelType::Count);
      appendControlDecl(fmt::format("uint64_t {0};", countVarName));
      appendControl(fmt::format("cudaMemcpy(&{0}, d_{0}, sizeof(uint64_t), cudaMemcpyDeviceToHost);", countVarName));
      if (!isProfiling())
         appendControl("}\n");
   }
   std::string MakeKeys(mlir::Operation* op, const mlir::ArrayAttr& keys, KernelType kernelType) {
      //TODO(avinash, p3): figure a way out for double keys
      auto keyType = getHTKeyType(keys);
      appendKernel(fmt::format("{1} {0} = 0;", KEY(op), keyType), kernelType);
      std::map<std::string, int> allowedKeysToSize;
      allowedKeysToSize["DBCharType"] = 1;
      allowedKeysToSize["DBStringType"] = 2;
      allowedKeysToSize["DBI32Type"] = 4;
      allowedKeysToSize["DBDateType"] = 4;
      allowedKeysToSize["DBI64Type"] = 4; // TODO(avinash): This is a temporary fix for date grouping.
      std::string sep = "";
      int totalKeySize = 0;
      for (auto i = 0ull; i < keys.size(); i++) {
         // check if key[i] is cpp string, then continue
         if (mlir::isa<mlir::StringAttr>(keys[i])) {
            continue;
         }
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

   void BuildHashTableSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::SemiJoinOp>(op);
      if (!joinOp) assert(false && "Build hash table accepts only semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Insert hash table kernel - SemiJoin", KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}, 1}});", HT(op), key), KernelType::Main);

      mainArgs[HT(op)] = "HASHTABLE_INSERT_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      appendControl("// Insert hash table control;");
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{({2})-1}},cuco::empty_value{{({3})-1}},thrust::equal_to<{2}>{{}},cuco::linear_probing<1, cuco::default_hash_function<{2}>>() }};",
                                HT(op), COUNT(op), getHTKeyType(keys), getHTValueType()));
      genLaunchKernel(KernelType::Main);
   }
   void BuildHashTableAntiSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::AntiSemiJoinOp>(op);
      if (!joinOp) assert(false && "Build hash table accepts only anti semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Insert hash table kernel - AntiJoin", KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}, 1}});", HT(op), key), KernelType::Main);

      mainArgs[HT(op)] = "HASHTABLE_INSERT_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      appendControl("// Insert hash table control;");
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{({2})-1}},cuco::empty_value{{({3})-1}},thrust::equal_to<{2}>{{}},cuco::linear_probing<1, cuco::default_hash_function<{2}>>() }};",
                                HT(op), COUNT(op), getHTKeyType(keys), getHTValueType()));
      genLaunchKernel(KernelType::Main);
   }
   void ProbeHashTableSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::SemiJoinOp>(op);
      if (!joinOp) assert(false && "Probe hash table accepts only semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      MakeKeys(op, keys, KernelType::Count);
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Probe Hash table");
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key));
      if (shouldUseThreadsAliveCodeGen()) {
         startThreadActiveScope(fmt::format("{0} != {1}.end()", SLOT(op), HT(op)));
         if (shouldGenerateShuffle()) {
            saveOpToShuffleBuffer(op);
            genShuffleThreads();
         } 
      } else {
         appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)));
      }

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

      appendKernel("// Probe Hash table");
      appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key));
      if (shouldUseThreadsAliveCodeGen()) {
         startThreadActiveScope(fmt::format("{0} == {1}.end()", SLOT(op), HT(op)));
         if (shouldGenerateShuffle()) {
            // We don't really need to save this op to shuffle buffer 
            // as the pipeline only progresses when there's no hash table
            // match.
            genShuffleThreads();
         } 
      } else {
         // Anti-Semi join. We should only output non-matching rows of the build size
         appendKernel(fmt::format("if ({0} != {1}.end()) return;", SLOT(op), HT(op))); 
      }

      mainArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      countArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
   }
   std::map<std::string, ColumnMetadata*> BuildHashTable(mlir::Operation* op, bool pk, bool right) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Insert hash table accepts only inner join operation.");
      std::string hash = right ? "rightHash" : "leftHash";
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>(hash);
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

      mainArgs[BUF_IDX(op)] = getBufIdxPtrType();
      if (pk)
         mainArgs[HT(op)] = "HASHTABLE_INSERT_PK";
      else
         mainArgs[HT(op)] = "HASHTABLE_INSERT";
      mainArgs[BUF(op)] = getBufPtrType();
      mlirToGlobalSymbol[BUF_IDX(op)] = fmt::format("d_{}", BUF_IDX(op));
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      mlirToGlobalSymbol[BUF(op)] = fmt::format("d_{}", BUF(op));
      appendControl("// Insert hash table control;");
      appendControlDecl(fmt::format("{1} d_{0} = nullptr;", BUF_IDX(op), getBufIdxPtrType()));
      appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof({1}));", BUF_IDX(op), getBufIdxType()));
      deviceFrees.insert(fmt::format("d_{0}", BUF_IDX(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof({1}));", BUF_IDX(op), getBufIdxType()));
      appendControlDecl(fmt::format("{1} d_{0} = nullptr;", BUF(op), getBufPtrType()));
      appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof({3}) * {1} * {2});", BUF(op), COUNT(op), baseRelations.size(), getBufEltType()));
      deviceFrees.insert(fmt::format("d_{0}", BUF(op)));
      // #ifdef MULTIMAP
      if (!pk)
         appendControl(fmt::format("auto d_{0} = cuco::experimental::static_multimap{{ (int){1}*2, cuco::empty_key{{({2})-1}},cuco::empty_value{{({3})-1}},thrust::equal_to<{2}>{{}},cuco::linear_probing<1, cuco::default_hash_function<{2}>>() }};",
                                   HT(op), COUNT(op), getHTKeyType(keys), getHTValueType()));
      // #else
      else
         appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{({2})-1}},cuco::empty_value{{({3})-1}},thrust::equal_to<{2}>{{}},cuco::linear_probing<1, cuco::default_hash_function<{2}>>() }};",
                                   HT(op), COUNT(op), getHTKeyType(keys), getHTValueType()));
      // #endif
      genLaunchKernel(KernelType::Main);
      
      if (gUseBloomFiltersForJoin) {
         appendControl(fmt::format("thrust::device_vector<{0}> keys_{1}(d_{2}.size()), vals_{1}(d_{2}.size());", getHTKeyType(keys), GetId(op), HT(op)));
         appendControl(fmt::format("d_{0}.retrieve_all(keys_{1}.begin(), vals_{1}.begin());", HT(op), GetId(op))); // retrieve all the keys from the hash table into the keys vector

         // create a bloom filter
         appendControl(fmt::format("auto d_{0} = cuco::bloom_filter<{1}>(max(d_{2}.size()/32, 1));", BF(op), getHTKeyType(keys), HT(op))); // 32 is an arbitrary constant. We need to fix this.
         appendControl(fmt::format("d_{0}.add(keys_{1}.begin(), keys_{1}.end());", BF(op), GetId(op))); // insert all the keys into the bloom filter         
      }

      // appendControl(fmt::format("cudaFree(d_{0});", BUF_IDX(op)));
      return columnData;
   }

   void ProbeBloomFilter(mlir::Operation* op, std::string key, bool pk) {
      if (!gUseBloomFiltersForJoin) 
         return;
      
      if (pk) {
         appendKernel("// Probe Bloom filter");
         if (shouldUseThreadsAliveCodeGen()) {
            auto threadActiveCondition = fmt::format("{0}.contains({1});", BF(op), key);
            startThreadActiveScope(threadActiveCondition);
         }
         else
            appendKernel(fmt::format("if (!{0}.contains({1})) return;", BF(op), key));

         mainArgs[BF(op)] = "BLOOM_FILTER_CONTAINS";
         countArgs[BF(op)] = "BLOOM_FILTER_CONTAINS";
         mlirToGlobalSymbol[BF(op)] = fmt::format("d_{}.ref()", BF(op));
      } else {
         // assert(false && "Bloom filter for multi-map not implemented yet.");
      }
   }

   void ProbeHashTable(mlir::Operation* op, const std::map<std::string, ColumnMetadata*>& leftColumnData, bool pk, bool right) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Probe hash table accepts only inner join operation.");
      // bool shouldShuffleAtThisOp = true; // TODO: Get this from the join operator
      
      std::string hash = right ? "leftHash" : "rightHash";
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>(hash);
      MakeKeys(op, keys, KernelType::Count);
      auto key = MakeKeys(op, keys, KernelType::Main);

      // check the bloom filter first, before probing the hash table, if bloom filters are enabled
      ProbeBloomFilter(op, key, pk);

      appendKernel("// Probe Hash table");

      if (!pk) {
         appendKernel(fmt::format("{0}.for_each({1}, [&] __device__ (auto const {2}) {{", HT(op), key, SLOT(op)));
         appendKernel(fmt::format("auto const [{0}, {1}] = {2};", slot_first(op), slot_second(op), SLOT(op)));
            forEachScopes++;

      } else {
         if (shouldUseThreadsAliveCodeGen()) { 
            // we are not inside a forEach lambda function, so we can use the threadActive variable
            appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key));
            auto threadActiveCondition = fmt::format("{0} != {1}.end()", SLOT(op), HT(op));
            startThreadActiveScope(threadActiveCondition);
            if (shouldGenerateShuffle()) {
               saveOpToShuffleBuffer(joinOp);
               genShuffleThreads();
            }            
         } else {
            // we are inside a forEach lambda function, so we cannot use the threadActive variable
            appendKernel(fmt::format("auto {0} = {1}.find({2});", SLOT(op), HT(op), key));
            appendKernel(fmt::format("if ({0} == {1}.end()) return;", SLOT(op), HT(op)));
         }
      }
      
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
            // #ifdef MULTIMAP
            if (!pk) {
               colData.second->rid = fmt::format("{3}[{0} * {1} + {2}]",
                                                 slot_second(op),
                                                 std::to_string(baseRelations.size()),
                                                 streamIdToBufId[colData.second->streamId],
                                                 BUF(op));
            }
            // #else
            else {
               colData.second->rid = fmt::format("{3}[{0} * {1} + {2}]",
                                                 slot_or_shuf_val(op),
                                                 std::to_string(baseRelations.size()),
                                                 streamIdToBufId[colData.second->streamId],
                                                 BUF(op));
            }
            // #endif
            // colData.second->streamId = id;
            columnData[colData.first] = colData.second;
            mlirToGlobalSymbol[colData.second->loadExpression] = colData.second->globalId;
         }
         columnData[colData.first] = colData.second;
      }
      if (pk) {
         mainArgs[HT(op)] = "HASHTABLE_PROBE_PK";
         mainArgs[BUF(op)] = getBufPtrType();
         countArgs[HT(op)] = "HASHTABLE_PROBE_PK";
         countArgs[BUF(op)] = getBufPtrType();
         
      } else {
         mainArgs[HT(op)] = "HASHTABLE_PROBE";
         mainArgs[BUF(op)] = getBufPtrType();
         countArgs[HT(op)] = "HASHTABLE_PROBE";
         countArgs[BUF(op)] = getBufPtrType();
      }
      // #ifdef MULTIMAP
      if (!pk)
         mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::for_each)", HT(op));
      // #else
      else
         mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
      // #endif
      mlirToGlobalSymbol[BUF(op)] = fmt::format("d_{}", BUF(op));
   }
   void CreateAggregationHashTable(mlir::Operation* op) {
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "CreateAggregationHashTable expects aggregation op as a parameter!");
      mlir::ArrayAttr groupByKeys = aggOp.getGroupByCols();
      if (groupByKeys.empty()){ // we are doing a global aggregation
         appendControl(fmt::format("size_t {0} = 1;", COUNT(op))); // just create a count variable of 1
         return;
      }
      auto key = MakeKeys(op, groupByKeys, KernelType::Count);
      appendKernel("// Create aggregation hash table", KernelType::Count);
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
      appendControl("// Create aggregation hash table");
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{({2})-1}},\
cuco::empty_value{{({3})-1}},\
thrust::equal_to<{2}>{{}},\
cuco::linear_probing<1, cuco::default_hash_function<{2}>>() }};",
                                HT(op), ht_size, getHTKeyType(groupByKeys), getHTValueType()));
      genLaunchKernel(KernelType::Count);
      appendControl(fmt::format("size_t {0} = d_{1}.size();", COUNT(op), HT(op)));
      // TODO(avinash): deallocate the old hash table and create a new one to save space in gpu when estimations are way off
      appendControl(fmt::format("thrust::device_vector<{3}> keys_{0}({2}), vals_{0}({2});\n\
d_{1}.retrieve_all(keys_{0}.begin(), vals_{0}.begin());\n\
d_{1}.clear();\n\
{3}* raw_keys{0} = thrust::raw_pointer_cast(keys_{0}.data());\n\
insertKeys<<<std::ceil((float){2}/128.), 128>>>(raw_keys{0}, d_{1}.ref(cuco::insert), {2});",
                                GetId(op), HT(op), COUNT(op), getHTKeyType(groupByKeys)));
   }
   void AggregateInHashTable(mlir::Operation* op) {
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "CreateAggregationHashTable expects aggregation op as a parameter!");
      mlir::ArrayAttr groupByKeys = aggOp.getGroupByCols();
      if (!groupByKeys.empty()) {
         auto key = MakeKeys(op, groupByKeys, KernelType::Main);
         mainArgs[HT(op)] = "HASHTABLE_FIND";
         mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
         appendKernel("// Aggregate in hashtable", KernelType::Main);
         appendKernel(fmt::format("auto {0} = {1}.find({2})->second;", buf_idx(op), HT(op), key), KernelType::Main);
      } else {
         appendKernel(fmt::format("auto {0} = 0;", buf_idx(op)), KernelType::Main);
      }
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
            appendControlDecl(fmt::format("{0}* d_{1} = nullptr;", bufferColType, newbuffername));
            appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof({1}) * {2});", newbuffername, bufferColType, COUNT(op)));
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
            appendControlDecl(fmt::format("DBI16Type* d_{0} = nullptr;", keyColumnName));
            appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof(DBI16Type) * {1});", keyColumnName, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", keyColumnName));
            appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(DBI16Type) * {1});", keyColumnName, COUNT(op)));
            auto key = LoadColumn<1>(mlir::cast<tuples::ColumnRefAttr>(col), KernelType::Main);
            appendKernel(fmt::format("{0}[{1}] = {2};", keyColumnName, buf_idx(op), key), KernelType::Main);
         } else {
            std::string keyColumnName = KEY(op) + mlirSymbol;
            mainArgs[keyColumnName] = keyColumnType + "*";
            mlirToGlobalSymbol[keyColumnName] = fmt::format("d_{}", keyColumnName);
            appendControlDecl(fmt::format("{0}* d_{1} = nullptr;", keyColumnType, keyColumnName));
            appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof({1}) * {2});", keyColumnName, keyColumnType, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", keyColumnName));
            appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof({1}) * {2});", keyColumnName, keyColumnType, COUNT(op)));
            auto key = LoadColumn(mlir::cast<tuples::ColumnRefAttr>(col), KernelType::Main);
            appendKernel(fmt::format("{0}[{1}] = {2};", keyColumnName, buf_idx(op), key), KernelType::Main);
         }
      }
      genLaunchKernel(KernelType::Main);
   }
   void MaterializeBuffers(mlir::Operation* op) {
      auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(op);
      if (!materializeOp) assert(false && "Materialize buffer needs materialize op as argument.");

      appendControl("// Materialize buffers");
      appendControlDecl(fmt::format("uint64_t* d_{0} = nullptr;", MAT_IDX(op)));
      appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof(uint64_t));", MAT_IDX(op)));
      deviceFrees.insert(fmt::format("d_{0}", MAT_IDX(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", MAT_IDX(op)));
      mainArgs[MAT_IDX(op)] = "uint64_t*";
      mlirToGlobalSymbol[MAT_IDX(op)] = "d_" + MAT_IDX(op);
      appendKernel("// Materialize buffers", KernelType::Main);
      appendKernel(fmt::format("auto {0} = atomicAdd((int*){1}, 1);", mat_idx(op), MAT_IDX(op)), KernelType::Main);
      for (auto col : materializeOp.getCols()) {
         auto columnAttr = mlir::cast<tuples::ColumnRefAttr>(col);
         auto detail = ColumnDetail(columnAttr);

         std::string mlirSymbol = detail.getMlirSymbol();
         std::string type = mlirTypeToCudaType(detail.type);

         if (type == "DBStringType") {
            std::string newBuffer = MAT(op) + mlirSymbol + "_encoded";
            appendControlDecl(fmt::format("DBI16Type* {0} = nullptr;", newBuffer));
            appendControl(fmt::format("mallocExt(&{0}, sizeof(DBI16Type) * {1});", newBuffer, COUNT(op)));
            hostFrees.insert(newBuffer);
            appendControlDecl(fmt::format("DBI16Type* d_{0} = nullptr;", newBuffer));
            appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof(DBI16Type) * {1});", newBuffer, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", newBuffer));
            mainArgs[newBuffer] = "DBI16Type*";
            mlirToGlobalSymbol[newBuffer] = "d_" + newBuffer;
            auto key = LoadColumn<1>(columnAttr, KernelType::Main);
            appendKernel(fmt::format("{0}[{2}] = {1};", newBuffer, key, mat_idx(op)), KernelType::Main);
         } else {
            std::string newBuffer = MAT(op) + mlirSymbol;
            appendControlDecl(fmt::format("{1}* {0} = nullptr;", newBuffer, type));
            appendControl(fmt::format("mallocExt(&{0}, sizeof({1}) * {2});", newBuffer, type, COUNT(op)));
            hostFrees.insert(newBuffer);
            appendControlDecl(fmt::format("{1}* d_{0} = nullptr;", newBuffer, type));
            appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof({1}) * {2});", newBuffer, type, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", newBuffer));
            mainArgs[newBuffer] = type + "*";
            mlirToGlobalSymbol[newBuffer] = "d_" + newBuffer;
            auto key = LoadColumn(columnAttr, KernelType::Main);
            appendKernel(fmt::format("{0}[{2}] = {1};", newBuffer, key, mat_idx(op)), KernelType::Main);
         }
      }
      genLaunchKernel(KernelType::Main);
      // appendControl(fmt::format("cudaFree(d_{0});", MAT_IDX(op)));
      std::string printStmts;
      std::string delimiter = "|";
      bool first = true;
      if (!isProfiling())
         appendControl("if (iter == numIterations - 1) {"); // only print at the last iteration
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
            if (type == "DBDateType")
               printStmts += fmt::format("std::cout << \"{0}\" << Date32ScalarToString({1}[i]);\n", first ? "" : delimiter, newBuffer);
            else
               printStmts += fmt::format("std::cout << \"{0}\" << {1}[i];\n", first ? "" : delimiter, newBuffer);
         }
         first = false;
      }
      // Only append the print statements if we are not generating kernel timing code
      // We want to be able to parse the timing info and don't want unnecessary print statements
      // when we're timing kernels
      if (!generateKernelTimingCode()) {         
         appendControl(fmt::format("for (auto i=0ull; i < {0}; i++) {{ {1}std::cout << std::endl; }}",
                                   COUNT(op), printStmts));
      }
      if (!isProfiling())
         appendControl("}");
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
         int i = 0;
         for (auto v : runtimeOp.getArgs()) {
            if ((i == 1) && (function == "Like")) {
               // remove first and last character from the string,
               std::string likeArg = SelectionOpDfs(v.getDefiningOp());
               if (likeArg[0] == '\"' && likeArg[likeArg.size() - 1] == '\"') {
                  likeArg = likeArg.substr(1, likeArg.size() - 2);
               }
               std::vector<std::string> tokens = split(likeArg, "%");
               std::string patternArray = "", sizeArray = "";
               std::clog << "TOKENS: ";
               for (auto t : tokens) std::clog << t << "|";
               std::clog << std::endl;
               int midpatterns = 0;
               if (tokens.size() <= 2) {
                  patternArray = "nullptr";
                  sizeArray = "nullptr";
               } else {
                  std::string t1 = "";
                  for (size_t i = 1; i < tokens.size() - 1; i++) {
                     patternArray += t1 + fmt::format("\"{}\"", tokens[i]);
                     sizeArray += t1 + std::to_string(tokens[i].size());
                     t1 = ", ";
                     midpatterns++;
                  }
               }
               std::string patarr = patternArray == "nullptr" ? "nullptr" : fmt::format("(const char*[]){{ {0} }}", patternArray);
               std::string sizearr = sizeArray == "nullptr" ? "nullptr" : fmt::format("(const int[]){{ {0} }}", sizeArray);
               args += sep + fmt::format("\"{0}\", \"{1}\", {2}, {3}, {4}", tokens[0], tokens[tokens.size() - 1], patarr, sizearr, midpatterns);
               break;
            } else {
               args += sep + SelectionOpDfs(v.getDefiningOp());
               sep = ", ";
            }
            i++;
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
         std::string leftOperand = mapOpDfs(left.getDefiningOp(), dep);

         auto right = compareOp.getRight();
         std::string rightOperand = mapOpDfs(right.getDefiningOp(), dep);

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
      } else if (auto extuiOp = mlir::dyn_cast_or_null<mlir::arith::ExtUIOp>(op)) {
         auto val = extuiOp.getIn();
         return fmt::format("({0})", mapOpDfs(val.getDefiningOp(), dep));
      } else if (auto andOp = mlir::dyn_cast_or_null<db::AndOp>(op)) {
         std::string res = "(";
         std::string sep = "";
         for (auto v : andOp.getVals()) {
            res += sep + mapOpDfs(v.getDefiningOp(), dep);
            sep = " && (";
            res += ")";
         }
         return res;
         // return "true";
      } else if (auto orOp = mlir::dyn_cast_or_null<db::OrOp>(op)) {
         std::string res = "(";
         std::string sep = "";
         for (auto v : orOp.getVals()) {
            res += sep + mapOpDfs(v.getDefiningOp(), dep);
            sep = " || (";
            res += ")";
         }
         return res;
      } else if (auto selectOp = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(op)) {
         auto cond = selectOp.getCondition();
         auto trueVal = selectOp.getTrueValue();
         auto falseVal = selectOp.getFalseValue();
         return fmt::format("({0}) ? ({1}) : ({2})", mapOpDfs(cond.getDefiningOp(), dep),
                            mapOpDfs(trueVal.getDefiningOp(), dep),
                            mapOpDfs(falseVal.getDefiningOp(), dep));
      } else if (auto castOp = mlir::dyn_cast_or_null<db::CastOp>(op)) {
         auto val = castOp.getVal();
         return fmt::format("(({1}){0})", mapOpDfs(val.getDefiningOp(), dep), mlirTypeToCudaType(castOp.getRes().getType()));
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

   void startThreadActiveScope(std::string predicate, KernelType kernelType = KernelType::Main_And_Count) {
      auto validKernelTypes = kernelType == KernelType::Main_And_Count  ? std::vector<KernelType>{KernelType::Main, KernelType::Count} : std::vector<KernelType>{kernelType};
      for (auto kt: validKernelTypes) {
         appendKernel(fmt::format("if ({0}) {{", predicate), kt);
         if (m_threadActiveScopeCount.find(kt) == m_threadActiveScopeCount.end()) {
            m_threadActiveScopeCount[kt] = 1;
         } else {
            m_threadActiveScopeCount[kt]++;
         }
      }
   }

   void writeThreadActiveScopeEndingBraces(KernelType kernelType, std::ostream& stream) {
      if (m_threadActiveScopeCount.find(kernelType) == m_threadActiveScopeCount.end()) {
         return; // no active scope to end
      }
      auto count = m_threadActiveScopeCount[kernelType];
      for (size_t i = 0; i < count; i++) {
         stream << "}\n"; // end the if threadActive scope
      }
      m_threadActiveScopeCount[kernelType] = 0;
   }

   void closeThreadActiveScopes(int k = -2, KernelType kernelType = KernelType::Main_And_Count) {
      auto validKernelTypes = kernelType == KernelType::Main_And_Count  ? std::vector<KernelType>{KernelType::Main, KernelType::Count} : std::vector<KernelType>{kernelType};
      for (auto kt: validKernelTypes) {
         if (m_threadActiveScopeCount.find(kt) == m_threadActiveScopeCount.end())
             continue; // no active scope to close

         auto count = m_threadActiveScopeCount[kt];
         assert((k >= -2 || k < (int)count) && "Invalid number of closing braces requested");
         auto numClosingBraces = 0;
         if (k >= 1)
            numClosingBraces = k;
         else if (k == -1)            
            numClosingBraces = count - 1; // close all but one active scope         
         else if (k == -2)
            numClosingBraces = count; // close all active scopes
         for (auto i = 0; i < numClosingBraces; i++) {
            appendKernel("}\n", kt);
         }
         m_threadActiveScopeCount[kt] -= numClosingBraces;
      }
   }

   void writeShuffleBufferDefinitions(std::ostream& stream) {
      if (m_shuffleData.cur_shuffle_id == 0)
         return;
      stream << fmt::format("SHUFFLE_IDX_INIT({0})\n", m_shuffleData.cur_shuffle_id);
      stream << "__shared__ int shuffle_buf_tid[128];\n";
      for (auto &op : m_shuffleData.shuffleBufOps) {
         stream << fmt::format("__shared__ int {0}[128];\n", SHUF_BUF_NAME(op)); // TODO: Should this be int? What's the datatype of tid?
         stream << fmt::format("size_t {0};", SHUF_BUF_VAL(op));
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
      for (auto p : _args) hasHash |= (p.second == "HASHTABLE_FIND" || p.second == "HASHTABLE_INSERT" || p.second == "HASHTABLE_PROBE" || p.second == "HASHTABLE_INSERT_SJ" || p.second == "HASHTABLE_PROBE_SJ" || p.second == "HASHTABLE_INSERT_PK" || p.second == "HASHTABLE_PROBE_PK" || p.second == "BLOOM_FILTER_CONTAINS");
      if (hasHash) {
         if (shouldGenerateSmallerHashTables()) {
            // The hash tables can be different sized (e.g., one hash table can have a 32-bit key and another can have a 64-bit key)
            // In this case, we just get a different template typename for each hash table
            stream << "template<";
            auto id = 0;
            std::string sep = "";
            for (auto p : _args) {
               if (p.second == "HASHTABLE_FIND" || p.second == "HASHTABLE_INSERT" || p.second == "HASHTABLE_PROBE" || p.second == "HASHTABLE_INSERT_SJ" || p.second == "HASHTABLE_PROBE_SJ" || p.second == "HASHTABLE_INSERT_PK" || p.second == "HASHTABLE_PROBE_PK" || p.second == "BLOOM_FILTER_CONTAINS") {
                  p.second = fmt::format("{}_{}", p.second, id++);
                  stream << fmt::format("{}typename {}", sep, p.second);
                  _args[p.first] = p.second;
                  sep = ", ";
               }
            }
            stream << ">\n";
         } else {
            if (gUseBloomFiltersForJoin)
               stream << "#error \"Bloom filter is not yet implemented for this case! Disable gUseBloomFiltersForJoin in code generation.\"";
            stream << "template<";
            bool find = false, insert = false, probe = false;
            bool insertSJ = false, probeSJ = false;
            bool insertPK = false, probePK = false;
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
               } else if (p.second == "HASHTABLE_INSERT_PK" && !insertPK) {
                  insertPK = true;
                  stream << sep + "typename " + p.second;
                  sep = ", ";
               } else if (p.second == "HASHTABLE_PROBE_PK" && !probePK) {
                  probePK = true;
                  stream << sep + "typename " + p.second;
                  sep = ", ";
               }
            }
            stream << ">\n";
         }
      }
      stream << fmt::format("__global__ void {0}_{1}(", _kernelName, GetId((void*) this));
      std::string sep = "";
      for (auto p : _args) {
         stream << fmt::format("{0}{1} {2}", sep, p.second, p.first);
         sep = ", ";
      }
      stream << ") {\n";
      writeShuffleBufferDefinitions(stream);
      if (KernelType::Main == ty) {
         for (auto line : mainCode) { stream << line << std::endl; }
      } else {
         for (auto line : countCode) { stream << line << std::endl; }
      }
      for (int i = 0; i < forEachScopes; i++) {
         stream << "});\n";
      }
      writeThreadActiveScopeEndingBraces(ty, stream);
      stream << "}\n";
   }

   NameTypePairs MaterializeColumns(mlir::Operation* op, mlir::ArrayAttr& columns, const std::string& countSuffix)
   {
      std::string countVarName = COUNT(op) + countSuffix;
      NameTypePairs materializedColNameAndTypes;
      if (columns.size() == 0) return materializedColNameAndTypes;

      bool isPartitionJoin = mlir::isa<relalg::InnerJoinOp>(op) && usePartitionHashJoin();
      if (isPartitionJoin) {
        assert(columns.size() == 1 && "Join columns size should be one");
        tuples::ColumnRefAttr keyAttr = mlir::cast<tuples::ColumnRefAttr>(columns[0]);
        ColumnDetail detail(keyAttr);
        assert((detail.table == m_joinInfo.leftTableName || detail.table == m_joinInfo.rightTableName) && "Table should match either left or right table name");
        std::string rowId = detail.table == m_joinInfo.leftTableName ? m_joinInfo.leftRowId : m_joinInfo.rightRowId;
        int ptrId = detail.table == m_joinInfo.leftTableName ? 1 : 2;
        std::string rowIdVarName = fmt::format("{0}_{1}", rowId, GetId(op));
        appendControl(fmt::format("uint64_t* {0} = {1}.get_typed_ptr<{2}>();", rowIdVarName, m_joinInfo.resultVar, ptrId));
        if (m_joinInfo.joinKeySet.contains(detail.column)) {
            materializedColNameAndTypes.push_back(std::make_pair(m_joinInfo.resultVar, mlirTypeToCudaType(detail.type)));
            materializedColNameAndTypes.push_back(std::make_pair(rowIdVarName, "uint64_t"));
        }
        else {
           auto filteredParamName = fmt::format("{0}_col_filtered", detail.column);
           auto filteredArgName = fmt::format("d_{0}_col_filtered_{1}", detail.column, GetId(op));
           auto inputColName = detail.column;
           auto cudaType = mlirTypeToCudaType(detail.type);
           appendControlDecl(fmt::format("{1}* {0} = nullptr;", filteredArgName, cudaType));
           appendControl(fmt::format("cudaMalloc(&{0}, sizeof({1}) * {2});", filteredArgName, cudaType, countVarName));
           deviceFrees.insert(filteredArgName);
           mainArgs[filteredParamName] = cudaType + "*";
           mainArgs[inputColName] = cudaType + "*";
           mlirToGlobalSymbol[filteredParamName] = filteredArgName;
           mlirToGlobalSymbol[inputColName] = getGlobalSymbolName(detail.table, detail.column);
           // copy the value
           appendKernel("// Materialize join columns", KernelType::Main);
           appendKernel(fmt::format("{0}[tid] = {2}[{1}[tid]];", filteredParamName, rowId, detail.column), KernelType::Main);
           materializedColNameAndTypes.push_back(std::make_pair(filteredArgName, cudaType));
           materializedColNameAndTypes.push_back(std::make_pair(rowIdVarName, "uint64_t"));
        }
        genLaunchKernel(KernelType::Main);
        return materializedColNameAndTypes;
      }
      // if it's in key set, use the key. Else, materialize.
      // Materialize -> Add column to args, add output column arg, use output_column[tid] = idx_column[tid];
      appendKernel("// Materialize columns", KernelType::Main);
      appendKernel(fmt::format("auto filter_idx = atomicAdd((int*)filter_count, 1);"), KernelType::Main);

      std::string keyColumnNamesConcat = "";
      for (auto col : columns) {
         if (keyColumnNamesConcat.size() > 0) {
            keyColumnNamesConcat += "_";
         }
         tuples::ColumnRefAttr keyAttr = mlir::cast<tuples::ColumnRefAttr>(col);
         keyColumnNamesConcat += getColumnName<tuples::ColumnRefAttr>(keyAttr);
      }

      auto filteredParamName = fmt::format("{0}_col_filtered", keyColumnNamesConcat);
      auto filteredIdxParamName = fmt::format("{0}_col_filtered_idx", keyColumnNamesConcat);
      auto filteredArgName = fmt::format("d_{0}_col_filtered_{1}", keyColumnNamesConcat, GetId(op));
      auto filteredIdxArgName = fmt::format("d_{0}_col_filtered_idx_{1}", keyColumnNamesConcat, GetId(op));
      auto filteredCountArgName = fmt::format("d_{0}_filtered_count", GetId(op));
      auto cudaType = getHTKeyType(columns);

      deviceFrees.insert(filteredArgName);
      deviceFrees.insert(filteredIdxArgName);
      materializedColNameAndTypes.push_back(std::make_pair(filteredArgName, cudaType));
      materializedColNameAndTypes.push_back(std::make_pair(filteredIdxArgName, "uint64_t"));

      appendControlDecl(fmt::format("{1}* {0} = nullptr;", filteredArgName, cudaType));
      appendControlDecl(fmt::format("uint64_t* {0} = nullptr;", filteredIdxArgName));
      appendControlDecl(fmt::format("uint64_t* {0} = nullptr;", filteredCountArgName));
      appendControl(fmt::format("cudaMallocExt(&{0}, sizeof({1}) * {2});", filteredArgName, cudaType, countVarName));
      appendControl(fmt::format("cudaMallocExt(&{0}, sizeof(uint64_t) * {1});", filteredIdxArgName, countVarName));
      appendControl(fmt::format("cudaMallocExt(&{0}, sizeof(uint64_t));", filteredCountArgName));
      deviceFrees.insert(filteredArgName);
      deviceFrees.insert(filteredIdxArgName);
      deviceFrees.insert(filteredCountArgName);

      // parameters
      mainArgs[filteredParamName] = cudaType + "*";
      mainArgs[filteredIdxParamName] = "uint64_t*";
      
      // arguments
      mlirToGlobalSymbol[filteredParamName] = filteredArgName;
      mlirToGlobalSymbol[filteredIdxParamName] = filteredIdxArgName;

      auto mergedKey = MakeKeys(op, columns, KernelType::Main);
      // copy the value
      appendKernel(fmt::format("{0}[filter_idx] = {1};", filteredParamName, mergedKey), KernelType::Main);
      appendKernel(fmt::format("{0}[filter_idx] = tid;", filteredIdxParamName), KernelType::Main);

      // Allocate memory for filtered columns
      for (auto col : columns) {
         tuples::ColumnRefAttr keyAttr = mlir::cast<tuples::ColumnRefAttr>(col);
         ColumnDetail detail(keyAttr);

         mainArgs[detail.column] = mlirTypeToCudaType(detail.type) + "*";
         mlirToGlobalSymbol[detail.column] = getGlobalSymbolName(detail.table, detail.column);
      }

      // memset size to 0
      appendControl(fmt::format("cudaMemset({0}, 0, sizeof(uint64_t));", filteredCountArgName));

      mainArgs["filter_count"] = "uint64_t*";
      mlirToGlobalSymbol["filter_count"] = filteredCountArgName;
      genLaunchKernel(KernelType::Main);
      return materializedColNameAndTypes;
   }

   std::string BuildPartitionHashJoinChunkType(const NameTypePairs& keys) {
      std::string chunkType = "Chunk<";
      std::string sep = "";
      for (auto key : keys) {
         chunkType += sep + key.second;
         sep = ", ";
      }
      chunkType += ">";
      return chunkType;
   }

   NameTypePairs MaterializeJoinResult(
      mlir::Operation* joinOp,
      mlir::ArrayAttr& leftKeys,
      mlir::ArrayAttr& rightKeys,
      const NameTypePairs& materializedLeftKeys,
      const NameTypePairs& materializedRightKeys,
      const std::string& leftCount,
      const std::string& rightCount) {
      std::string leftTupleVar = fmt::format("left_{0}_tuple", cudacodegen::GetId(joinOp));
      std::string rightTupleVar = fmt::format("right_{0}_tuple", cudacodegen::GetId(joinOp));
      std::string leftResultType = fmt::format("Left_{0}_result_t", cudacodegen::GetId(joinOp));
      std::string rightResultType = fmt::format("Right_{0}_result_t", cudacodegen::GetId(joinOp));

      std::string resultVar = fmt::format("{0}_{1}_result", leftTupleVar, rightTupleVar);
      m_joinResultVariables[joinOp] = resultVar;

      std::string leftChunkType = BuildPartitionHashJoinChunkType(materializedLeftKeys);
      std::string rightChunkType = BuildPartitionHashJoinChunkType(materializedRightKeys);
      NameTypePairs resultColumns =
      {
            std::make_pair(resultVar, materializedLeftKeys[0].second),
            materializedLeftKeys[1],
            materializedRightKeys[1]
      };
      std::string resultChunkType = BuildPartitionHashJoinChunkType(resultColumns);

      appendControl(fmt::format("// Partition Hash Join for {0}", cudacodegen::GetId(joinOp)));
      // Determine result tuple type based on input tuples
      std::string resultType = fmt::format("{0}_{1}_result_t", leftTupleVar, rightTupleVar);
      appendControl(fmt::format("using {0} = {1};", leftResultType, leftChunkType));
      appendControl(fmt::format("using {0} = {1};", rightResultType, rightChunkType));
      appendControl(fmt::format("using {0} = {1};", resultType, resultChunkType));

      // Create input tuples
      appendControl(fmt::format("{0} {1};", leftChunkType, leftTupleVar));
      appendControl(fmt::format("{0} {1};", rightChunkType, rightTupleVar));

      // set number of items for each tuple.
      appendControl(fmt::format("{0}.set_num_items({1});", leftTupleVar, leftCount));
      appendControl(fmt::format("{0}.set_num_items({1});", rightTupleVar, rightCount));

      // Set column data pointers for each tuple
      for (size_t i = 0; i < materializedLeftKeys.size(); i++) {
         auto columnExpression = mlirToGlobalSymbol.contains(materializedLeftKeys[i].first) ? mlirToGlobalSymbol[materializedLeftKeys[i].first] : materializedLeftKeys[i].first;
         appendControl(fmt::format("{0}.add_column({1});", leftTupleVar, columnExpression));
      }

      for (size_t i = 0; i < materializedRightKeys.size(); i++) {
         auto columnExpression = mlirToGlobalSymbol.contains(materializedRightKeys[i].first) ? mlirToGlobalSymbol[materializedRightKeys[i].first] : materializedRightKeys[i].first;
         appendControl(fmt::format("{0}.add_column({1});", rightTupleVar, columnExpression));
      }
      
      appendControl(fmt::format("auto {0} = partitionHashJoinHelper<{1}>({2}, {3});", resultVar, resultType, leftTupleVar, rightTupleVar));

      // Materialize join count.
      appendControlDecl(fmt::format("uint64_t {0};", COUNT(joinOp)));
      appendControlDecl(fmt::format("uint64_t* d_{0} = nullptr;", COUNT(joinOp)));
      appendControl(fmt::format("{0} = {1}.num_items;", COUNT(joinOp), resultVar));
      return resultColumns;
   }

   PartitionHashJoinResultInfo PartitionHashJoin(mlir::Operation* joinOp,
                          mlir::Operation* leftStream,
                          mlir::Operation* rightStream,
                          HyperTupleStreamCode* leftStreamCode,
                          HyperTupleStreamCode* rightStreamCode) {
      auto innerJoinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(joinOp);
      
      if (!innerJoinOp) {
         assert(false && "Only InnerJoinOp is supported for partition hash join");
      }

      // Get join keys
      mlir::ArrayAttr leftKeys, rightKeys;

      leftKeys = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
      rightKeys = innerJoinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");

      // Filter out keys that are not tuples::ColumnRefAttr
      std::vector<mlir::Attribute> filteredLeftKeys;
      for (auto key : leftKeys) {
         if (key.isa<tuples::ColumnRefAttr>()) {
         filteredLeftKeys.push_back(key);
         }
      }
      leftKeys = mlir::ArrayAttr::get(leftKeys.getContext(), filteredLeftKeys);

      std::vector<mlir::Attribute> filteredRightKeys;
      for (auto key : rightKeys) {
         if (key.isa<tuples::ColumnRefAttr>()) {
         filteredRightKeys.push_back(key);
         }
      }
      rightKeys = mlir::ArrayAttr::get(rightKeys.getContext(), filteredRightKeys);

      auto countVarSuffix =  "_" + GetId(joinOp);
      auto leftSuffix = mlir::isa<relalg::InnerJoinOp>(leftStream) ? "" : countVarSuffix;
      auto rightSuffix = mlir::isa<relalg::InnerJoinOp>(rightStream) ? "" : countVarSuffix;
      leftStreamCode->MaterializeCount(leftStream, leftSuffix);
      rightStreamCode->MaterializeCount(rightStream, rightSuffix);

      auto materializedLeftCols = leftStreamCode->MaterializeColumns(leftStream, leftKeys, leftSuffix);
      auto materializedRightCols = rightStreamCode->MaterializeColumns(rightStream, rightKeys, rightSuffix);

      auto resultColumns = MaterializeJoinResult(joinOp, leftKeys, rightKeys, materializedLeftCols, materializedRightCols, COUNT(leftStream) + leftSuffix, COUNT(rightStream) + rightSuffix);
      PartitionHashJoinResultInfo resultInfo;
      resultInfo.resultVar = resultColumns[0].first;
      resultInfo.resultType = resultColumns[0].second;
      for (auto key : leftKeys) {
         tuples::ColumnRefAttr keyAttr = mlir::cast<tuples::ColumnRefAttr>(key);

         resultInfo.joinKeySet.insert(getColumnName<tuples::ColumnRefAttr>(keyAttr));
         resultInfo.leftTableName = getTableName<tuples::ColumnRefAttr>(keyAttr);
         resultInfo.leftRowId = resultColumns[1].first;
      }
      for (auto key : rightKeys) {
         tuples::ColumnRefAttr keyAttr = mlir::cast<tuples::ColumnRefAttr>(key);

         resultInfo.joinKeySet.insert(getColumnName<tuples::ColumnRefAttr>(keyAttr));
         resultInfo.rightTableName = getTableName<tuples::ColumnRefAttr>(keyAttr);
         resultInfo.rightRowId = resultColumns[2].first;
      }
      return resultInfo;
   }
};

class CudaCodeGen : public mlir::PassWrapper<CudaCodeGen, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-cuda-code-gen"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaCodeGen)

   std::map<mlir::Operation*, HyperTupleStreamCode*> streamCodeMap;
   std::vector<HyperTupleStreamCode*> kernelSchedule;

   CudaCodeGen() {}

   void runOnOperation() override {
      getOperation().walk([&](mlir::Operation* op) {
         if (auto selection = llvm::dyn_cast<relalg::SelectionOp>(op)) {
            mlir::Operation* stream = selection.getRelMutable().get().getDefiningOp();
            HyperTupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation found for selection.");
            }

            mlir::Region& predicate = selection.getPredicate();
            // Get the selectivity attribute if available
            float selectivity = 0.5f; // Default selectivity
            if (auto selectivityAttr = selection->getAttrOfType<mlir::FloatAttr>("selectivity")) 
               selectivity = selectivityAttr.getValueAsDouble();
            streamCode->AddSelectionPredicate(predicate, selectivity);
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

            if (usePartitionHashJoin()) {
               // Perform partition hash join
               auto joinResultInfo = rightStreamCode->PartitionHashJoin(op, leftStream, rightStream, leftStreamCode, rightStreamCode);
               kernelSchedule.push_back(leftStreamCode);
               kernelSchedule.push_back(rightStreamCode);

               auto newJoinStream = new HyperTupleStreamCode(joinOp, joinResultInfo, leftStreamCode, rightStreamCode);
               mlir::Region& predicate = joinOp.getPredicate();
               newJoinStream->AddSelectionPredicate(predicate);
               streamCodeMap[op] = newJoinStream;
            } else {
               auto leftkeys = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");

               std::set<std::string> leftkeysSet;
               for (auto key : leftkeys) {
                  if (mlir::isa<mlir::StringAttr>(key)) {
                     continue;
                  }
                  tuples::ColumnRefAttr key1 = mlir::cast<tuples::ColumnRefAttr>(key);
                  ColumnDetail detail(key1);
                  leftkeysSet.insert(detail.column);
               }
               bool is_pk = false;
               bool is_left_pk = isPrimaryKey(leftkeysSet);

               is_pk |= is_left_pk;
               std::set<std::string> rightkeysSet;
               auto rightKeys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
               for (auto key : rightKeys) {
                  if (mlir::isa<mlir::StringAttr>(key)) {
                     continue;
                  }
                  tuples::ColumnRefAttr key1 = mlir::cast<tuples::ColumnRefAttr>(key);
                  ColumnDetail detail(key1);
                  rightkeysSet.insert(detail.column);
               }
               bool is_right_pk = invertJoinIfPossible(rightkeysSet, is_left_pk);
               if (is_right_pk) {
                  std::swap(leftStreamCode, rightStreamCode);
               }
               is_pk |= is_right_pk;

               leftStreamCode->MaterializeCount(op); // count of left
               auto leftCols = leftStreamCode->BuildHashTable(op, is_pk, is_right_pk); // main of left
               kernelSchedule.push_back(leftStreamCode);
               rightStreamCode->ProbeHashTable(op, leftCols, is_pk, is_right_pk);
               mlir::Region& predicate = joinOp.getPredicate();
               rightStreamCode->AddSelectionPredicate(predicate);

               streamCodeMap[op] = rightStreamCode;
            }
         } else if (auto aggregationOp = llvm::dyn_cast<relalg::AggregationOp>(op)) {
            mlir::Operation* stream = aggregationOp.getRelMutable().get().getDefiningOp();
            HyperTupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for aggregation found");
            }

            streamCode->CreateAggregationHashTable(op); // count part
            streamCode->AggregateInHashTable(op); // main part
            kernelSchedule.push_back(streamCode);

            auto newStreamCode = new HyperTupleStreamCode(aggregationOp);
            streamCodeMap[op] = newStreamCode;
         } else if (auto scanOp = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
            std::string tableName = scanOp.getTableIdentifier().data();
            HyperTupleStreamCode* streamCode = new HyperTupleStreamCode(scanOp);

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
            HyperTupleStreamCode* streamCode = streamCodeMap[stream];
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
            #include <thrust/host_vector.h>\n \
            #include \"cudautils.cuh\"\n\
            #include \"db_types.h\"\n\
            #include \"dbruntime.h\"\n\
            #include <chrono>\n";

      if (generateKernelTimingCode()) {
         outputFile << "#include <cuda_runtime.h>\n";
      }
      if (gUseBloomFiltersForJoin) outputFile << "#include <cuco/bloom_filter.cuh>\n";

      if (usePartitionHashJoin()) {
         outputFile << "#include \"../db-utils/phj/partitioned_hash_join.cuh\"\n";
      }

      for (auto code : kernelSchedule) {
         code->printKernel(KernelType::Count, outputFile);
         code->printKernel(KernelType::Main, outputFile);
      }

      emitControlFunctionSignature(outputFile);
      emitTimingEventCreation(outputFile);
      for (auto code : kernelSchedule) {
         code->printControlDeclarations(outputFile);
      }
      outputFile << "size_t used_mem = usedGpuMem();\n";

      // generate timing start
      if (!isProfiling()) {
         outputFile << "auto startTime = std::chrono::high_resolution_clock::now();\n";
         outputFile << fmt::format("size_t numIterations = {0};\n", generateKernelTimingCode() ? 10 : 2);
         outputFile << "for (size_t iter = 0; iter < numIterations; iter++) {\n";
         outputFile << "bool runCountKernel = (iter == 0);\n";
         outputFile << "if (iter == 1) startTime = std::chrono::high_resolution_clock::now();\n"; // start the timer after the warp up iteration
      }
      for (auto code : kernelSchedule) {
         code->printControl(outputFile);
      }
      if (!isProfiling()) {
         outputFile << "}\n";
         outputFile << "auto endTime = std::chrono::high_resolution_clock::now();\n";
         outputFile << fmt::format("auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime)/(numIterations - 1);\n");
         if (generateKernelTimingCode())
            outputFile << "std::cout << \"total_query, \" << duration.count() / 1000. << std::endl;\n";
      }
      outputFile << "std::clog << \"Used memory: \" << used_mem / (1024 * 1024) << \" MB\" << std::endl; \n\
size_t aux_mem = usedGpuMem() - used_mem;\n\
std::clog << \"Auxiliary memory: \" << aux_mem / (1024) << \" KB\" << std::endl;\n";

      for (auto code : kernelSchedule) {
         code->printFrees(outputFile);
      }
      outputFile << "}";
      outputFile.close();
   }
};
}

std::unique_ptr<mlir::Pass> relalg::createCudaCodeGenPass() {
   return std::make_unique<cudacodegen::CudaCodeGen>();
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
   checkForCodeGenSwitches(argc, argv);
}