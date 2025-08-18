#include "lingodb/compiler/Dialect/RelAlg/CudaCodeGenHelper.h"
using namespace lingodb::compiler::dialect;

namespace cudacodegen {

static int StreamId = 0;

class CrystalTupleStreamCode : public TupleStreamCode {
   
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
      return fmt::format("{0}_{1}<<<std::ceil((float){2}/(float)TILE_SIZE), TILE_SIZE/ITEMS_PER_THREAD>>>({3});", _kernelName, GetId((void*) this), size, args);
   }

   public:
   CrystalTupleStreamCode(relalg::BaseTableOp& baseTableOp) {
      std::string tableName = baseTableOp.getTableIdentifier().data();
      std::string tableSize = tableName + "_size";
      mlirToGlobalSymbol[tableSize] = tableSize;
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only

      if (generatePerOperationProfile())
         appendKernel("int64_t start, stop, cycles_per_warp;", KernelType::Main);
      appendKernel("size_t tile_offset = blockIdx.x * TILE_SIZE;");
      appendKernel("size_t tid = tile_offset + threadIdx.x;");
      appendKernel("int selection_flags[ITEMS_PER_THREAD];");
      appendKernel("for (int i=0; i<ITEMS_PER_THREAD; i++) selection_flags[i] = 1;");

      for (auto namedAttr : baseTableOp.getColumns().getValue()) {
         auto columnName = namedAttr.getName().str();
         ColumnDetail detail(mlir::cast<tuples::ColumnDefAttr>(namedAttr.getValue()));
         auto globalSymbol = fmt::format("d_{0}__{1}", tableName, columnName);
         auto mlirSymbol = detail.getMlirSymbol();
         mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
         ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
         metadata->rid = "ITEM*TB + tid";
         columnData[mlirSymbol] = metadata;

         if (mlirTypeToCudaType(detail.type) == "DBStringType") {
            ColumnMetadata* encoded_metadata = new ColumnMetadata(mlirSymbol + "_encoded", ColumnType::Direct, StreamId, globalSymbol + "_encoded");
            encoded_metadata->rid = "ITEM*TB + tid";
            columnData[mlirSymbol + "_encoded"] = encoded_metadata;
            mlirToGlobalSymbol[mlirSymbol + "_encoded"] = globalSymbol + "_encoded";
         }
      }
      id = StreamId;
      StreamId++;
      return;
   }
   CrystalTupleStreamCode(mlir::Operation* op) {
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "Expected aggregation operation");
      std::string tableSize = COUNT(op);

      mlirToGlobalSymbol[tableSize] = tableSize;
      mainArgs[tableSize] = "size_t";
      countArgs[tableSize] = "size_t"; // make sure this type is reserved for kernel size only

      if (generatePerOperationProfile())
         appendKernel("int64_t start, stop, cycles_per_warp;", KernelType::Main);
      appendKernel("size_t tile_offset = blockIdx.x * TILE_SIZE;");
      appendKernel("size_t tid = tile_offset + threadIdx.x;");
      appendKernel("int selection_flags[ITEMS_PER_THREAD];");
      appendKernel("for (int i=0; i<ITEMS_PER_THREAD; i++) selection_flags[i] = 1;");

      auto groupByKeys = aggOp.getGroupByCols();
      auto computedCols = aggOp.getComputedCols();
      for (auto& col : groupByKeys) {
         ColumnDetail detail(mlir::cast<tuples::ColumnRefAttr>(col));
         if (mlirTypeToCudaType(detail.type) == "DBStringType") {
            auto mlirSymbol = detail.getMlirSymbol() + "_encoded";
            auto globalSymbol = fmt::format("d_{0}", KEY(op) + mlirSymbol);
            mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
            ColumnMetadata* encoded_metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
            encoded_metadata->rid = "ITEM*TB + tid";
            columnData[mlirSymbol] = encoded_metadata;
         } else {
            auto mlirSymbol = detail.getMlirSymbol();
            auto globalSymbol = fmt::format("d_{0}", KEY(op) + mlirSymbol);
            mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
            ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
            metadata->rid = "ITEM*TB + tid";
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
            encoded_metadata->rid = "ITEM*TB + tid";
            columnData[mlirSymbol] = encoded_metadata;
         } else {
            auto mlirSymbol = detail.getMlirSymbol();
            auto globalSymbol = fmt::format("d_{0}", mlirSymbol);
            mlirToGlobalSymbol[mlirSymbol] = globalSymbol;
            ColumnMetadata* metadata = new ColumnMetadata(mlirSymbol, ColumnType::Direct, StreamId, globalSymbol);
            metadata->rid = "ITEM*TB + tid";
            columnData[mlirSymbol] = metadata;
         }
      }
      id = StreamId;
      StreamId++;
      return;
   }
   ~CrystalTupleStreamCode() {
      for (auto p : columnData) delete p.second;
   }
   void PushProfileInfo(mlir::Operation* op, std::string opName) {
      profileInfo.push_back(std::make_pair(op, opName));
   }
   void AddOperationProfile(mlir::Operation* op, std::string opName) {
      std::string profile_name = fmt::format("main_{0}_{1}_{2}", GetId((void*) this), opName, GetId((void*) op));
      appendControl(fmt::format("int64_t* d_cycles_per_warp_{0};", profile_name));
      appendControl(fmt::format("auto {0}_cpw_size = std::ceil((float){1}/(float)TILE_SIZE);", profile_name, getKernelSizeVariable()));
      appendControl(fmt::format("cudaMalloc(&d_cycles_per_warp_{0}, sizeof(int64_t) * {0}_cpw_size);", profile_name));
      appendControl(fmt::format("cudaMemset(d_cycles_per_warp_{0}, -1, sizeof(int64_t) * {0}_cpw_size);", profile_name));
      mainArgs["cycles_per_warp_" + profile_name] = "int64_t*";
      mlirToGlobalSymbol["cycles_per_warp_" + profile_name] = "d_cycles_per_warp_" + profile_name;
   }
   void BeginProfileKernel(mlir::Operation* op, std::string opName) {
      std::string profile_name = fmt::format("main_{0}_{1}_{2}", GetId((void*) this), opName, GetId((void*) op));
      appendKernel(fmt::format("if (threadIdx.x == 0) start = clock64();"), KernelType::Main);
   }
   void EndProfileKernel(mlir::Operation* op, std::string opName) {
      std::string profile_name = fmt::format("main_{0}_{1}_{2}", GetId((void*) this), opName, GetId((void*) op));
      appendKernel(fmt::format("if (threadIdx.x == 0) {{\
            stop = clock64();\
            cycles_per_warp = (stop - start);\
            cycles_per_warp_{0}[blockIdx.x] = cycles_per_warp;}}",
                               profile_name), KernelType::Main);
   }
   void PrintOperationProfile() {
      for (auto p : profileInfo) {
         mlir::Operation* op = p.first;
         std::string opName = p.second;
         std::string profile_name = fmt::format("main_{0}_{1}_{2}", GetId((void*) this), opName, GetId((void*) op));
         appendControl(fmt::format("int64_t* cycles_per_warp_{0} = (int64_t*)malloc(sizeof(int64_t) * {0}_cpw_size);", profile_name));
         appendControl(fmt::format("cudaMemcpy(cycles_per_warp_{0}, d_cycles_per_warp_{0}, sizeof(int64_t) * {0}_cpw_size, cudaMemcpyDeviceToHost);", profile_name));
         appendControl("std::cout << \"" + profile_name + " \";");
         appendControl(fmt::format("for (auto i=0ull; i < {0}_cpw_size; i++) std::cout << cycles_per_warp_{0}[i] << \" \";", profile_name));
         appendControl("std::cout << std::endl;");
      }
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
      std::string colType = mlirTypeToCudaType(detail.type);
      if (enc == 1) colType = "DBI16Type"; // use for string encoded columns
      appendKernel(fmt::format("{0} {1}[ITEMS_PER_THREAD];", colType, cudaId), ty);
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll", ty);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize), ty);
      if (!(colData->rid == "ITEM*TB + tid") || m_hasInsertedSelection) {
         appendKernel("if (!selection_flags[ITEM]) continue;", ty);
      }
      appendKernel(fmt::format("{0}[ITEM] = {1};", cudaId, colData->loadExpression + (colData->type == ColumnType::Direct ? "[" + colData->rid + "]" : "")), ty);
      appendKernel("}", ty);
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
         return LoadColumn(getColOp.getAttr(), KernelType::Main) + "[ITEM]";
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
         m_hasInsertedSelection = true;
         return fmt::format("evaluatePredicate({0}, {1}, {2})", leftOperand, rightOperand, predicate);
      } else if (auto runtimeOp = mlir::dyn_cast_or_null<db::RuntimeCall>(op)) { // or a like operation
         // TODO(avinash, p1): handle runtime predicate like operator
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
         m_hasInsertedSelection = true;
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
            m_hasInsertedSelection = true;
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
   void AddSelectionPredicate(mlir::Region& predicate) {
      auto terminator = mlir::cast<tuples::ReturnOp>(predicate.front().getTerminator());
      if (!terminator.getResults().empty()) {
         auto& predicateBlock = predicate.front();
         if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(predicateBlock.getTerminator())) {
            mlir::Value matched = returnOp.getResults()[0];
            std::string condition = SelectionOpDfs(matched.getDefiningOp());
            std::string kernelSize = getKernelSizeVariable();
            appendKernel("#pragma unroll");
            appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize));
            appendKernel("if (!selection_flags[ITEM]) continue;");
            appendKernel(fmt::format("selection_flags[ITEM] &= {0};", condition));
            appendKernel("}");
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
      appendKernel("// Materialize count", KernelType::Count);
      appendKernel("#pragma unroll", KernelType::Count);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", getKernelSizeVariable()), KernelType::Count);
      appendKernel("if (!selection_flags[ITEM]) continue;", KernelType::Count);
      appendKernel(fmt::format("atomicAdd((int*){0}, 1);", COUNT(op)), KernelType::Count);
      appendKernel("}", KernelType::Count);

      appendControl("// Materialize count");
      appendControlDecl(fmt::format("uint64_t* d_{0} = nullptr;", COUNT(op)));
      if (!isProfiling())
         appendControl("if (runCountKernel){\n");
      appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof(uint64_t));", COUNT(op)));
      deviceFrees.insert(fmt::format("d_{0}", COUNT(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", COUNT(op)));
      genLaunchKernel(KernelType::Count);
      appendControlDecl(fmt::format("uint64_t {0};", COUNT(op)));
      appendControl(fmt::format("cudaMemcpy(&{0}, d_{0}, sizeof(uint64_t), cudaMemcpyDeviceToHost);", COUNT(op)));
      if (!isProfiling())
         appendControl("}\n");
   }
   std::string MakeKeys(mlir::Operation* op, const mlir::ArrayAttr& keys, KernelType kernelType) {
      //TODO(avinash, p3): figure a way out for double keys
      appendKernel(fmt::format("uint64_t {0}[ITEMS_PER_THREAD];", KEY(op)), kernelType);
      std::map<std::string, int> allowedKeysToSize;
      allowedKeysToSize["DBCharType"] = 1;
      allowedKeysToSize["DBStringType"] = 2;
      allowedKeysToSize["DBI32Type"] = 4;
      allowedKeysToSize["DBDateType"] = 4;
      allowedKeysToSize["DBI64Type"] = 4; // TODO(avinash): This is a temporary fix for date grouping.
      std::string sep = "";
      std::vector<std::string> loadedColumnIds;
      for (auto i = 0ull; i < keys.size(); i++) {
         if (mlir::isa<mlir::StringAttr>(keys[i])) {
            continue;
         }
         tuples::ColumnRefAttr key = mlir::cast<tuples::ColumnRefAttr>(keys[i]);
         auto baseType = mlirTypeToCudaType(key.getColumn().type);
         if (baseType == "DBStringType") {
            loadedColumnIds.push_back(LoadColumn<1>(key, kernelType));
         } else {
            loadedColumnIds.push_back(LoadColumn(key, kernelType));
         }
      }
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll", kernelType);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize), kernelType);
      if (m_genSelectionCheckUniversally || m_hasInsertedSelection )
         appendKernel("if (!selection_flags[ITEM]) continue;", kernelType);
      appendKernel(fmt::format("{0}[ITEM] = 0;", KEY(op)), kernelType);
      int totalKeySize = 0;
      int j = 0;
      for (auto i = 0ull; i < keys.size(); i++) {
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
         std::string cudaIdentifierKey = loadedColumnIds[j++];
         if (sep != "") appendKernel(sep, kernelType);
         if (i < keys.size() - 1) {
            tuples::ColumnRefAttr next_key = mlir::cast<tuples::ColumnRefAttr>(keys[i + 1]);
            auto next_base_type = mlirTypeToCudaType(next_key.getColumn().type);

            sep = fmt::format("{0}[ITEM] <<= {1};", KEY(op), std::to_string(allowedKeysToSize[next_base_type] * 8));
         } else {
            sep = "";
         }
         if (baseType == "DBI64Type") {
            appendKernel(fmt::format("{0}[ITEM] |= (DBI32Type){1}[ITEM];", KEY(op), cudaIdentifierKey), kernelType);
         } else {
            appendKernel(fmt::format("{0}[ITEM] |= {1}[ITEM];", KEY(op), cudaIdentifierKey), kernelType);
         }
         totalKeySize += allowedKeysToSize[baseType];
         if (totalKeySize > 8) {
            std::clog << totalKeySize << std::endl;
            keys.dump();
            assert(false && "Total hash key exceeded 8 bytes");
         }
      }
      appendKernel("}", kernelType);
      return KEY(op);
   }

   void BuildHashTableSemiJoin(mlir::Operation* op) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::SemiJoinOp>(op);
      if (!joinOp) assert(false && "Build hash table accepts only semi join operation.");
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Insert hash table kernel;", KernelType::Main);
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll", KernelType::Main);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize), KernelType::Main);
      appendKernel(fmt::format("if (!selection_flags[ITEM]) continue;"), KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}[ITEM], 1}});", HT(op), key), KernelType::Main);
      appendKernel("}", KernelType::Main);

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
      appendKernel("// Insert hash table kernel;", KernelType::Main);
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll", KernelType::Main);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize), KernelType::Main);
      appendKernel(fmt::format("if (!selection_flags[ITEM]) continue;"), KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}[ITEM], 1}});", HT(op), key), KernelType::Main);
      appendKernel("}", KernelType::Main);

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
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll");
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize));
      if (m_genSelectionCheckUniversally || m_hasInsertedSelection )
         appendKernel("if (!selection_flags[ITEM]) continue;");
      appendKernel(fmt::format("auto {0} = {1}.find({2}[ITEM]);", SLOT(op), HT(op), key));
      appendKernel(fmt::format("if ({0} == {1}.end()) {{selection_flags[ITEM] = 0;}}", SLOT(op), HT(op)));
      appendKernel("}");

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
      appendKernel("//Probe Hash table");
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll");
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize));
      if (m_genSelectionCheckUniversally || m_hasInsertedSelection )
         appendKernel("if (!selection_flags[ITEM]) continue;");
      appendKernel(fmt::format("auto {0} = {1}.find({2}[ITEM]);", SLOT(op), HT(op), key));
      appendKernel(fmt::format("if (!({0} == {1}.end())) {{selection_flags[ITEM] = 0;}}", SLOT(op), HT(op)));
      appendKernel("}");

      mainArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      countArgs[HT(op)] = "HASHTABLE_PROBE_SJ";
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
   }
   std::map<std::string, ColumnMetadata*> BuildHashTable(mlir::Operation* op, bool right) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Insert hash table accepts only inner join operation.");
      std::string hash = right ? "rightHash" : "leftHash";
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>(hash);
      auto key = MakeKeys(op, keys, KernelType::Main);
      appendKernel("// Insert hash table kernel;", KernelType::Main);
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll", KernelType::Main);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize), KernelType::Main);
      appendKernel(fmt::format("if (!selection_flags[ITEM]) continue;"), KernelType::Main);
      appendKernel(fmt::format("auto {0} = atomicAdd((int*){1}, 1);", buf_idx(op), BUF_IDX(op)), KernelType::Main);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}[ITEM], {2}}});", HT(op), key, buf_idx(op)), KernelType::Main);
      auto baseRelations = getBaseRelations(columnData);
      int i = 0;
      for (auto br : baseRelations) {
         appendKernel(fmt::format("{0}[({1}) * {2} + {3}] = {4};",
                                  BUF(op),
                                  buf_idx(op),
                                  std::to_string(baseRelations.size()),
                                  i++,
                                  br.second),
                      KernelType::Main);
      }
      appendKernel("}", KernelType::Main);
      mainArgs[BUF_IDX(op)] = getBufIdxPtrType();
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
      appendControl(fmt::format("auto d_{0} = cuco::static_map{{ (int){1}*2, cuco::empty_key{{({2})-1}},cuco::empty_value{{(int64_t)-1}},thrust::equal_to<{2}>{{}},cuco::linear_probing<1, cuco::default_hash_function<{2}>>() }};",
                                HT(op), COUNT(op), getHTKeyType(keys), getHTValueType()));
      genLaunchKernel(KernelType::Main);
      // appendControl(fmt::format("cudaFree(d_{0});", BUF_IDX(op)));
      return columnData;
   }

   void ProbeHashTable(mlir::Operation* op, const std::map<std::string, ColumnMetadata*>& leftColumnData, bool right) {
      auto joinOp = mlir::dyn_cast_or_null<relalg::InnerJoinOp>(op);
      if (!joinOp) assert(false && "Probe hash table accepts only inner join operation.");
      std::string hash = right ? "leftHash" : "rightHash";
      auto keys = joinOp->getAttrOfType<mlir::ArrayAttr>(hash);
      MakeKeys(op, keys, KernelType::Count);
      auto key = MakeKeys(op, keys, KernelType::Main);

      std::string kernelSize = getKernelSizeVariable();
      appendKernel(fmt::format("int64_t {0}[ITEMS_PER_THREAD];", slot_second(op)));
      appendKernel("#pragma unroll");
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize));
      if (m_genSelectionCheckUniversally || m_hasInsertedSelection )
         appendKernel("if (!selection_flags[ITEM]) continue;");
      appendKernel(fmt::format("auto {0} = {1}.find({2}[ITEM]);", SLOT(op), HT(op), key));
      appendKernel(fmt::format("if ({0} == {1}.end()) {{selection_flags[ITEM] = 0; continue;}}", SLOT(op), HT(op)));
      appendKernel(fmt::format("{0}[ITEM] = {1}->second;", slot_second(op), SLOT(op)));
      appendKernel("}");

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
            colData.second->rid = fmt::format("{3}[{0}[ITEM] * {1} + {2}]",
                                              slot_second(op),
                                              std::to_string(baseRelations.size()),
                                              streamIdToBufId[colData.second->streamId],
                                              BUF(op));
            columnData[colData.first] = colData.second;
            mlirToGlobalSymbol[colData.second->loadExpression] = colData.second->globalId;
         }
         columnData[colData.first] = colData.second;
      }
      mainArgs[HT(op)] = "HASHTABLE_PROBE";
      mainArgs[BUF(op)] = getBufPtrType();
      countArgs[HT(op)] = "HASHTABLE_PROBE";
      countArgs[BUF(op)] = getBufPtrType();
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
      mlirToGlobalSymbol[BUF(op)] = fmt::format("d_{}", BUF(op));
   }
   void CreateAggregationHashTable(mlir::Operation* op) {
      // We'll run the count kernels twice to get a better estimate of the aggregation hash table size.
      // First pass: count the number of unique keys (with a bool that instructs to just count)
      // Second pass: insert the keys into the hash table
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "CreateAggregationHashTable expects aggregation op as a parameter!");
      mlir::ArrayAttr groupByKeys = aggOp.getGroupByCols();
      if (groupByKeys.empty()){ // we are doing a global aggregation
         appendControl(fmt::format("size_t {0} = 1;", COUNT(op))); // just create a count variable of 1
         return;
      }
      auto key = MakeKeys(op, groupByKeys, KernelType::Count);
      appendKernel("// Create aggregation hash table", KernelType::Count);
      appendKernel("#pragma unroll", KernelType::Count);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", getKernelSizeVariable()), KernelType::Count);
      appendKernel("if (!selection_flags[ITEM]) continue;", KernelType::Count);
      appendKernel(fmt::format("if (countKeys) estimator.add({0}[ITEM]); else ", key), KernelType::Count);
      appendKernel(fmt::format("{0}.insert(cuco::pair{{{1}[ITEM], 1}});", HT(op), key), KernelType::Count);
      appendKernel("}", KernelType::Count);
      countArgs[HT(op)] = "HASHTABLE_INSERT";
      
      mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
      
      std::string ht_size = "0";
      static bool useQueryOptimizerEstimate = false;
      if (useQueryOptimizerEstimate) {
         // TODO(avinash, p2): this is a hacky way, actually check if --use-db flag is enabled and query optimization is performed      
         if (auto floatAttr = mlir::dyn_cast_or_null<mlir::FloatAttr>(op->getAttr("rows"))) {
            if (std::floor(floatAttr.getValueAsDouble()) != 0) {            
               ht_size = std::to_string(std::min((int) std::ceil(floatAttr.getValueAsDouble()), INT32_MAX/2));
            }
            else {
               for (auto p : countArgs) {
                  if (p.second == "size_t")
                     ht_size = p.first;
               }
            }
         }      
         assert(ht_size != "0" && "hash table for aggregation is sizing to be 0!!");
      } else
         ht_size = "1";  // create an empty table for now

      appendControlDecl("// Create aggregation hash table");
      appendControlDecl(fmt::format("auto d_{0} = cuco::static_map{{ (int)({1}*1.25), cuco::empty_key{{({2})-1}},\
cuco::empty_value{{({3})-1}},\
thrust::equal_to<{2}>{{}},\
cuco::linear_probing<1, cuco::default_hash_function<{2}>>() }};",
                                HT(op), ht_size, getHTKeyType(groupByKeys), getHTValueType()));
      if (!isProfiling())
         appendControl("if (runCountKernel){\n");

      appendControlDecl(fmt::format("uint64_t *d_{0} = nullptr;", COUNT(op)));
      appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof(uint64_t));", COUNT(op)));
      appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof(uint64_t));", COUNT(op)));
      appendControlDecl(fmt::format("size_t {0} = 0;", COUNT(op)));

      countArgs[COUNT(op)] = "uint64_t*";
      countArgs["countKeys"] = "bool";
      mlirToGlobalSymbol[COUNT(op)] = fmt::format("d_{0}", COUNT(op));
      mlirToGlobalSymbol["countKeys"] = fmt::format("true", "countKeys");
      countArgs["estimator"] = "HLL_ESTIMATOR_REF";
      mlirToGlobalSymbol["estimator"] = fmt::format("estimator_{0}.ref()", GetId(op));

      // TODO: Check if we need a bigger sketch size than 32_KB
      appendControl(fmt::format("cuco::hyperloglog<{0}> estimator_{1}(32_KB);", getHTKeyType(groupByKeys), GetId(op)));
      genLaunchKernel(KernelType::Count);

      // Pass 2: Actually use the hash table
      // appendControl(fmt::format("cudaMemcpy(&{0}, d_{0}, sizeof(uint64_t), cudaMemcpyDeviceToHost);", COUNT(op)));      
      appendControl(fmt::format("{0} = estimator_{1}.estimate();", COUNT(op), GetId(op)));
      appendControl(fmt::format("d_{1}.rehash((int)({0} * 1.2));", COUNT(op), HT(op)));
      mlirToGlobalSymbol["countKeys"] = fmt::format("false", "countKeys");
      genLaunchKernel(KernelType::Count);
      appendControl(fmt::format("{0} = d_{1}.size();", COUNT(op), HT(op)));
      // TODO(avinash): deallocate the old hash table and create a new one to save space in gpu when estimations are way off
      appendControl(fmt::format("thrust::device_vector<{3}> keys_{0}({2}), vals_{0}({2});\n\
d_{1}.retrieve_all(keys_{0}.begin(), vals_{0}.begin());\n\
d_{1}.clear();\n\
{3}* raw_keys{0} = thrust::raw_pointer_cast(keys_{0}.data());\n\
insertKeys<<<std::ceil((float){2}/128.), 128>>>(raw_keys{0}, d_{1}.ref(cuco::insert), {2});",
                                GetId(op), HT(op), COUNT(op), getHTKeyType(groupByKeys)));
      if (!isProfiling())
         appendControl("}\n"); // runCountKernel
   }
   void AggregateInHashTable(mlir::Operation* op) {
      auto aggOp = mlir::dyn_cast_or_null<relalg::AggregationOp>(op);
      if (!aggOp) assert(false && "CreateAggregationHashTable expects aggregation op as a parameter!");
      mlir::ArrayAttr groupByKeys = aggOp.getGroupByCols();
      auto key = MakeKeys(op, groupByKeys, KernelType::Main);
      if (!groupByKeys.empty()) {
         mainArgs[HT(op)] = "HASHTABLE_FIND";
         mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::find)", HT(op));
         appendKernel("// Aggregate in hashtable", KernelType::Main);
      }else {
         appendKernel(fmt::format("auto {0} = 0;", buf_idx(op)), KernelType::Main);
      }
      auto& aggRgn = aggOp.getAggrFunc();
      mlir::ArrayAttr computedCols = aggOp.getComputedCols(); // these are columndefs
      appendControl("//Aggregate in hashtable");

      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(aggRgn.front().getTerminator())) {
         for (mlir::Value col : returnOp.getResults()) {
            if (auto aggrFunc = llvm::dyn_cast<relalg::AggrFuncOp>(col.getDefiningOp())) {
               // TODO(avinash): check if it is a string column
               ColumnDetail detail(aggrFunc.getAttr());
               if (mlirTypeToCudaType(detail.type) == "DBStringType") {
                  LoadColumn<1>(mlir::cast<tuples::ColumnRefAttr>(aggrFunc.getAttr()), KernelType::Main);
               } else {
                  LoadColumn(mlir::cast<tuples::ColumnRefAttr>(aggrFunc.getAttr()), KernelType::Main);
               }
            }
         }
      }
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll", KernelType::Main);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize), KernelType::Main);
      if (m_genSelectionCheckUniversally || m_hasInsertedSelection )
         appendKernel(fmt::format("if (!selection_flags[ITEM]) continue;"), KernelType::Main);
      if (groupByKeys.empty()) {
         appendKernel(fmt::format("auto {0} = 0;", buf_idx(op)), KernelType::Main); // all the keys are zero here
      } else { 
         appendKernel(fmt::format("auto {0} = {1}.find({2}[ITEM])->second;", buf_idx(op), HT(op), key), KernelType::Main);
      }
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
                     appendKernel(fmt::format("aggregate_sum(&{0}, {1}[ITEM]);", slot, val), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::count: {
                     appendKernel(fmt::format("aggregate_sum(&{0}, 1);", slot), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::any: {
                     appendKernel(fmt::format("aggregate_any(&{0}, {1}[ITEM]);", slot, val), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::avg: {
                     assert(false && "average should be split into sum and divide");
                  } break;
                  case relalg::AggrFunc::min: {
                     appendKernel(fmt::format("aggregate_min(&{0}, {1}[ITEM]);", slot, val), KernelType::Main);
                  } break;
                  case relalg::AggrFunc::max: {
                     appendKernel(fmt::format("aggregate_max(&{0}, {1}[ITEM]);", slot, val), KernelType::Main);
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
            // This load column is redundant as we do it in makekeys
            auto key = LoadColumn<1>(mlir::cast<tuples::ColumnRefAttr>(col), KernelType::Main);
            appendKernel(fmt::format("{0}[{1}] = {2}[ITEM];", keyColumnName, buf_idx(op), key), KernelType::Main);
         } else {
            std::string keyColumnName = KEY(op) + mlirSymbol;
            mainArgs[keyColumnName] = keyColumnType + "*";
            mlirToGlobalSymbol[keyColumnName] = fmt::format("d_{}", keyColumnName);
            appendControlDecl(fmt::format("{0}* d_{1} = nullptr;", keyColumnType, keyColumnName));
            appendControl(fmt::format("cudaMallocExt(&d_{0}, sizeof({1}) * {2});", keyColumnName, keyColumnType, COUNT(op)));
            deviceFrees.insert(fmt::format("d_{0}", keyColumnName));
            appendControl(fmt::format("cudaMemset(d_{0}, 0, sizeof({1}) * {2});", keyColumnName, keyColumnType, COUNT(op)));
            auto key = LoadColumn(mlir::cast<tuples::ColumnRefAttr>(col), KernelType::Main);
            appendKernel(fmt::format("{0}[{1}] = {2}[ITEM];", keyColumnName, buf_idx(op), key), KernelType::Main);
         }
      }
      appendKernel("}", KernelType::Main);
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
      for (auto col : materializeOp.getCols()) {
         auto columnAttr = mlir::cast<tuples::ColumnRefAttr>(col);
         auto detail = ColumnDetail(columnAttr);

         std::string type = mlirTypeToCudaType(detail.type);
         if (type == "DBStringType") {
            LoadColumn<1>(columnAttr, KernelType::Main);
         } else {
            LoadColumn(columnAttr, KernelType::Main);
         }
      }
      std::string kernelSize = getKernelSizeVariable();
      appendKernel("#pragma unroll", KernelType::Main);
      appendKernel(fmt::format("for (int ITEM = 0; ITEM < ITEMS_PER_THREAD && (ITEM*TB + tid < {0}); ++ITEM) {{", kernelSize), KernelType::Main);
      appendKernel(fmt::format("if (!selection_flags[ITEM]) continue;"), KernelType::Main);
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
            appendKernel(fmt::format("{0}[{2}] = {1}[ITEM];", newBuffer, key, mat_idx(op)), KernelType::Main);
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
            appendKernel(fmt::format("{0}[{2}] = {1}[ITEM];", newBuffer, key, mat_idx(op)), KernelType::Main);
         }
      }
      appendKernel("}", KernelType::Main);
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
      if(!generatePerOperationProfile() && !generateKernelTimingCode()) {
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
         return fmt::format("reg_{}[ITEM]", detail.getMlirSymbol());
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
         m_hasInsertedSelection = true;
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
      for (auto p : _args) hasHash |= (p.second == "HASHTABLE_FIND" || p.second == "HASHTABLE_INSERT" || p.second == "HASHTABLE_PROBE" || p.second == "HASHTABLE_INSERT_SJ" || p.second == "HASHTABLE_PROBE_SJ" || p.second == "HASHTABLE_INSERT_PK" || p.second == "HASHTABLE_PROBE_PK" || p.second == "BLOOM_FILTER_CONTAINS" || p.second == "HLL_ESTIMATOR_REF");
      if (hasHash) {
         if (shouldGenerateSmallerHashTables()) {
            // The hash tables can be different sized (e.g., one hash table can have a 32-bit key and another can have a 64-bit key)
            // In this case, we just get a different template typename for each hash table
            stream << "template<";
            auto id = 0;
            std::string sep = "";
            for (auto p : _args) {
               if (p.second == "HASHTABLE_FIND" || p.second == "HASHTABLE_INSERT" || p.second == "HASHTABLE_PROBE" || p.second == "HASHTABLE_INSERT_SJ" || p.second == "HASHTABLE_PROBE_SJ" || p.second == "HASHTABLE_INSERT_PK" || p.second == "HASHTABLE_PROBE_PK" || p.second == "BLOOM_FILTER_CONTAINS" || p.second == "HLL_ESTIMATOR_REF") {
                  p.second = fmt::format("{}_{}", p.second, id++);
                  stream << fmt::format("{}typename {}", sep, p.second);
                  _args[p.first] = p.second;
                  sep = ", ";
               }
            }
            stream << ">\n";
         } else {
            stream << "template<";
            bool find = false, insert = false, probe = false;
            bool insertSJ = false, probeSJ = false;
            bool insertPK = false, probePK = false;
            bool hllRef = false;
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
               } else if (p.second == "HLL_ESTIMATOR_REF" && !hllRef) {
                  hllRef = true;
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
      if (KernelType::Main == ty) {
         for (auto line : mainCode) { stream << line << std::endl; }
      } else {
         for (auto line : countCode) { stream << line << std::endl; }
      }
      stream << "}\n";
   }
};

class CudaCrystalCodeGen : public mlir::PassWrapper<CudaCrystalCodeGen, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-cuda-code-gen-crystal"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaCrystalCodeGen)

   std::map<mlir::Operation*, CrystalTupleStreamCode*> streamCodeMap;
   std::vector<CrystalTupleStreamCode*> kernelSchedule;

   void runOnOperation() override {
      getOperation().walk([&](mlir::Operation* op) {
         if (auto selection = llvm::dyn_cast<relalg::SelectionOp>(op)) {
            mlir::Operation* stream = selection.getRelMutable().get().getDefiningOp();
            CrystalTupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation found for selection.");
            }

            mlir::Region& predicate = selection.getPredicate();
            if (generatePerOperationProfile()) {
               std::string opname = "selection";
               streamCode->AddOperationProfile(op, opname);
               streamCode->BeginProfileKernel(op, opname);
               streamCode->AddSelectionPredicate(predicate);
               streamCode->EndProfileKernel(op, opname);
               streamCode->PushProfileInfo(op, opname);
            } else {
               streamCode->AddSelectionPredicate(predicate);
            }
            
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
            bool left_pk = isPrimaryKey(leftkeysSet);

            bool right = false;
            if (left_pk == false) {
               std::set<std::string> rightkeysSet;
               // check if right side is a pk
               auto rightKeys = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
               for (auto key : rightKeys) {
                  if (mlir::isa<mlir::StringAttr>(key)) {
                     continue;
                  }
                  tuples::ColumnRefAttr key1 = mlir::cast<tuples::ColumnRefAttr>(key);
                  ColumnDetail detail(key1);
                  rightkeysSet.insert(detail.column);
               }
               bool right_pk = isPrimaryKey(rightkeysSet);
               if (right_pk == false) {
                  op->dump();
                  assert(false && "This join is not possible without multimap, since both sides are not pk");
               } else {
                  std::swap(leftStreamCode, rightStreamCode);
                  right = true;
               }
            }
            leftStreamCode->MaterializeCount(op); // count of left
            std::map<std::string, ColumnMetadata*> leftCols;
            if (generatePerOperationProfile()) {

               std::string opname = "join_build";
               leftStreamCode->AddOperationProfile(op, opname);
               leftStreamCode->BeginProfileKernel(op, opname);
               leftCols = leftStreamCode->BuildHashTable(op, right); // main of left
               leftStreamCode->EndProfileKernel(op, opname);
               leftStreamCode->PushProfileInfo(op, opname);
               leftStreamCode->PrintOperationProfile();
            } else {
               leftCols = leftStreamCode->BuildHashTable(op, right);
            }
            
            kernelSchedule.push_back(leftStreamCode);

            if (generatePerOperationProfile()) {

               std::string opname = "join_probe";
               rightStreamCode->AddOperationProfile(op, opname);
               rightStreamCode->BeginProfileKernel(op, opname);
   
               rightStreamCode->ProbeHashTable(op, leftCols, right);
               mlir::Region& predicate = joinOp.getPredicate();
               rightStreamCode->AddSelectionPredicate(predicate);
   
               rightStreamCode->EndProfileKernel(op, opname);
               rightStreamCode->PushProfileInfo(op, opname);
            } else {
               rightStreamCode->ProbeHashTable(op, leftCols, right);
               mlir::Region& predicate = joinOp.getPredicate();
               rightStreamCode->AddSelectionPredicate(predicate);
            }

            streamCodeMap[op] = rightStreamCode;
         } else if (auto aggregationOp = llvm::dyn_cast<relalg::AggregationOp>(op)) {
            mlir::Operation* stream = aggregationOp.getRelMutable().get().getDefiningOp();
            CrystalTupleStreamCode* streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for aggregation found");
            }

            streamCode->CreateAggregationHashTable(op); // count part

            if (generatePerOperationProfile()) {

               std::string opname = "aggregation";
               streamCode->AddOperationProfile(op, opname);
               streamCode->BeginProfileKernel(op, opname);
   
               streamCode->AggregateInHashTable(op); // main part
               streamCode->EndProfileKernel(op, opname);
               streamCode->PushProfileInfo(op, opname);
               streamCode->PrintOperationProfile();
            } else {
               streamCode->AggregateInHashTable(op); // main part
            }
            kernelSchedule.push_back(streamCode);

            auto newStreamCode = new CrystalTupleStreamCode(op);
            streamCodeMap[op] = newStreamCode;
         } else if (auto scanOp = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
            std::string tableName = scanOp.getTableIdentifier().data();
            CrystalTupleStreamCode* streamCode = new CrystalTupleStreamCode(scanOp);

            streamCodeMap[op] = streamCode;
         } else if (auto mapOp = llvm::dyn_cast<relalg::MapOp>(op)) {
            auto stream = mapOp.getRelMutable().get().getDefiningOp();
            auto streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for map op found");
            }
            if (generatePerOperationProfile()) {

               std::string opname = "map";
               streamCode->AddOperationProfile(op, opname);
               streamCode->BeginProfileKernel(op, opname);
               streamCode->TranslateMapOp(op);
               streamCode->EndProfileKernel(op, opname);
               streamCode->PushProfileInfo(op, opname);
            } else {
               streamCode->TranslateMapOp(op);
            }
            streamCodeMap[op] = streamCode;
         } else if (auto materializeOp = llvm::dyn_cast<relalg::MaterializeOp>(op)) {
            auto stream = materializeOp.getRelMutable().get().getDefiningOp();
            auto streamCode = streamCodeMap[stream];
            if (!streamCode) {
               stream->dump();
               assert(false && "No downstream operation for materialize found");
            }

            streamCode->MaterializeCount(op);

            if (generatePerOperationProfile()) {

               std::string opname = "materialize";
               streamCode->AddOperationProfile(op, opname);
               streamCode->BeginProfileKernel(op, opname);
               streamCode->MaterializeBuffers(op);
               streamCode->EndProfileKernel(op, opname);
               streamCode->PushProfileInfo(op, opname);
               streamCode->PrintOperationProfile();
            } else {
               streamCode->MaterializeBuffers(op);
            }
            kernelSchedule.push_back(streamCode);
         } else if (auto sortOp = llvm::dyn_cast<relalg::SortOp>(op)) {
            std::clog << "WARNING: This operator has not been implemented, bypassing it.\n";
            op->dump();
            streamCodeMap[op] = streamCodeMap[sortOp.getRelMutable().get().getDefiningOp()];
         } else if (auto renamingOp = llvm::dyn_cast<relalg::RenamingOp>(op)) {
            mlir::Operation* stream = renamingOp.getRelMutable().get().getDefiningOp();
            CrystalTupleStreamCode* streamCode = streamCodeMap[stream];
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
            if (generatePerOperationProfile()) {
               std::string opname = "semi_join_build";
               rightStreamCode->AddOperationProfile(op, opname);
               rightStreamCode->BeginProfileKernel(op, opname);
               rightStreamCode->BuildHashTableSemiJoin(op); // main of left
               rightStreamCode->EndProfileKernel(op, opname);
               rightStreamCode->PushProfileInfo(op, opname);
            } else {
               rightStreamCode->BuildHashTableSemiJoin(op); // main of left
            }
            kernelSchedule.push_back(rightStreamCode);
            if (generatePerOperationProfile()) {
               std::string opname = "semi_join_probe";
               leftStreamCode->AddOperationProfile(op, opname);
               leftStreamCode->BeginProfileKernel(op, opname);
               leftStreamCode->ProbeHashTableSemiJoin(op);
               mlir::Region& predicate = semiJoinOp.getPredicate();
               leftStreamCode->AddSelectionPredicate(predicate);
               leftStreamCode->EndProfileKernel(op, opname);
               leftStreamCode->PushProfileInfo(op, opname);
            } else {
               leftStreamCode->ProbeHashTableSemiJoin(op);
               mlir::Region& predicate = semiJoinOp.getPredicate();
               leftStreamCode->AddSelectionPredicate(predicate);
            }

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
            if (generatePerOperationProfile()) {
               std::string opname = "anti_semi_join_build";
               rightStreamCode->AddOperationProfile(op, opname);
               rightStreamCode->BeginProfileKernel(op, opname);
               rightStreamCode->BuildHashTableAntiSemiJoin(op); // main of left
               rightStreamCode->EndProfileKernel(op, opname);
               rightStreamCode->PushProfileInfo(op, opname);
            } else {
               rightStreamCode->BuildHashTableAntiSemiJoin(op); // main of left
            }
            kernelSchedule.push_back(rightStreamCode);
            if (generatePerOperationProfile()) {
               std::string opname = "anti_semi_join_probe";
               leftStreamCode->AddOperationProfile(op, opname);
               leftStreamCode->BeginProfileKernel(op, opname);
               leftStreamCode->ProbeHashTableAntiSemiJoin(op);
               mlir::Region& predicate = antiSemiJoinOp.getPredicate();
               leftStreamCode->AddSelectionPredicate(predicate);
               leftStreamCode->EndProfileKernel(op, opname);
               leftStreamCode->PushProfileInfo(op, opname);
            } else {
               leftStreamCode->ProbeHashTableAntiSemiJoin(op);
               mlir::Region& predicate = antiSemiJoinOp.getPredicate();
               leftStreamCode->AddSelectionPredicate(predicate);
            }

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
#include \"dbruntime.h\"\n\
#include <chrono>\n\
#include <cuco/hyperloglog.cuh>\n\
#define ITEMS_PER_THREAD 4\n\
#define TILE_SIZE 512\n\
#define TB TILE_SIZE/ITEMS_PER_THREAD\n";
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
std::unique_ptr<mlir::Pass>
relalg::createCudaCrystalCodeGenPass() { return std::make_unique<cudacodegen::CudaCrystalCodeGen>(); }
