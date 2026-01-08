#include "lingodb/compiler/Dialect/RelAlg/CudaCodeGenHelper.h"

std::vector<std::string> split(std::string s, std::string delimiter) {
   size_t pos_start = 0, pos_end, delim_len = delimiter.length();
   std::string token;
   std::vector<std::string> res;

   while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
      token = s.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + delim_len;
      res.push_back(token);
   }

   res.push_back(s.substr(pos_start));
   return res;
}

// -- [start] kernel timing code generation --

static bool gGenKernelTimingCode = false;
static bool gGenPerOperationProfile = false;
static bool gGenIsProfiling = false; // Do not generate multiple iteration loops if we are profiling
static bool gPartitionHashJoinCodeGenEnabled = false;
bool gUseHTValForRowIdx = true; // Use hash table value as row index instead of storing row index separately in a buf
static std::string gProfileRangeName = "PROFILE_RANGE";

bool generateKernelTimingCode() { return gGenKernelTimingCode; }
bool generatePerOperationProfile() { return gGenPerOperationProfile; }
bool isProfiling() { return gGenIsProfiling; }
bool usePartitionHashJoin() { return gPartitionHashJoinCodeGenEnabled; }
std::string& getProfileRangeName() { return gProfileRangeName; }

// -- [end] kernel timing code generation --

// --- [start] different sized hash tables helpers ---

static bool gSmallerHashTables = false;

bool shouldGenerateSmallerHashTables() {
   return gSmallerHashTables;
}

std::string getHTKeyType(mlir::ArrayAttr keys) {
   if (!gSmallerHashTables || keys.size() > 1)
      return "int64_t";
   else
      return "int32_t"; 
}

std::string getHTValueType() {
   if (!gSmallerHashTables)
      return "int64_t";
   else
      return "int32_t";
}

std::string getBufEltType() {
   if (!gSmallerHashTables)
      return "uint64_t";
   else
      return "uint32_t";
}

std::string getBufIdxType() {
   return getBufEltType();
}

std::string getBufIdxPtrType() {
   return getBufIdxType() + "*";
}

std::string getBufPtrType() {
   return getBufEltType() + "*";
}

// --- [end] different sized hash tables helpers ---

namespace cudacodegen {

IdGenerator<const void*> idGen;

std::string HT(const void* op) {
    return "HT_" + GetId(op);
}
std::string KEY(const void* op) {
    return "KEY_" + GetId(op);
}
std::string SLOT(const void* op) {
    return "SLOT_" + GetId(op);
}
std::string BUF(const void* op) {
    return "BUF_" + GetId(op);
}
std::string BUF_IDX(const void* op) {
    return "BUF_IDX_" + GetId(op);
}
std::string buf_idx(const void* op) {
    return "buf_idx_" + GetId(op);
}
std::string COUNT(const void* op) {
    return "COUNT" + GetId(op);
}
std::string MAT(const void* op) {
    return "MAT" + GetId(op);
}
std::string MAT_IDX(const void* op) {
    return "MAT_IDX" + GetId(op);
}
std::string mat_idx(const void* op) {
    return "mat_idx" + GetId(op);
}
std::string slot_first(const void* op) {
    return "slot_first" + GetId(op);
}
std::string slot_second(const void* op) {
    return "slot_second" + GetId(op);
}
std::string BF(const void* op) {
    return "BF_" + GetId(op);
}
std::string SHUF_BUF_NAME(const void* op) {
   return "shuffle_buf_" + (op == nullptr ? "tid" : GetId(op));
}
std::string SHUF_BUF_EXPR(const void* op) {
   return op == nullptr ? "tid" : (SLOT(op) + "->second");
}
std::string SHUF_BUF_VAL(const void* op) {
   return (op == nullptr ? "tid" : "slot_val_" + GetId(op));
}
std::string TILE_ID(const void *op) {
   return HT(op) + "_tile_idx";
}
std::string GetId(const void* op){   
   std::string result = idGen.getId(op);
   return result;
}

//TODO(avinash): this function is incorrect (given by chatgpt)
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

   return static_cast<int>(std::ceil(duration.count() / 24.0));
}

std::string mlirTypeToCudaType(const mlir::Type& ty) {
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

std::string translateConstantOp(db::ConstantOp& constantOp) {
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

size_t TupleStreamCode::getL2CacheSize() {
   const char* envVar = std::getenv("CUR_GPU");
   if (envVar != nullptr) {
      std::string gpuName(envVar);
      if (gpuName == "A6000") {
         return 6 * 1024 * 1024; // 6MB for A6000
      } else if (gpuName == "4090") {
         return 72 * 1024 * 1024; // 72MB for RTX 4090
      } else if (gpuName == "4060") {
         return 24 * 1024 * 1024; // 24MB for RTX 4060
      } else if (gpuName == "A100") {
         return 40 * 1024 * 1024; // 40MB for A100
      }
   } 
   std::cerr << "Warning: Unknown GPU name in CUR_GPU environment variable.\n";
   exit(1);
}

void TupleStreamCode::genCreateHashTable(mlir::Operation* op, const mlir::ArrayAttr& keys, std::string joinType = "") {
   appendControl("// Create hash table control;");
   printHashTableSize(COUNT(op), getHTKeyType(keys), getHTValueType(), "2", op);
   mainArgs[HT(op)] = "HASHTABLE_INSERT_" + joinType;
   mlirToGlobalSymbol[HT(op)] = fmt::format("d_{}.ref(cuco::insert)", HT(op));
   if (gTileHashTables) {
      appendControlDecl(fmt::format("auto d_{0} = cuco::tiled_static_map{{ (int) 1, cuco::empty_key{{({1})-1}},cuco::empty_value{{({2})-1}},thrust::equal_to<{1}>{{}},cuco::linear_probing<1, cuco::default_hash_function<{1}>>() }};",
                              HT(op), getHTKeyType(keys), getHTValueType()));
      appendControl(fmt::format("d_{0}.clear();", HT(op)));
   } else {
      appendControlDecl(fmt::format("auto d_{0} = cuco::static_map{{ (int) 1, cuco::empty_key{{({1})-1}},cuco::empty_value{{({2})-1}},thrust::equal_to<{1}>{{}},cuco::linear_probing<1, cuco::default_hash_function<{1}>>() }};",
                           HT(op), getHTKeyType(keys), getHTValueType()));
      appendControl(fmt::format("d_{0}.clear();", HT(op)));
   }
}

void TupleStreamCode::AddPreHTProbeCounter(mlir::Operation* op) {
   if (!gPrintHashTableSizes) 
      return;

   recordHashTableProbe(op);

   appendControlDecl(fmt::format("uint64_t* d_{0}_PreCounter = nullptr;", HT(op)));      
   appendKernel(fmt::format("atomicAdd((int*){0}_PreCounter, 1);", HT(op)), KernelType::Main);
   mainArgs[fmt::format("{0}_PreCounter", HT(op))] = "uint64_t*";
   mlirToGlobalSymbol[fmt::format("{0}_PreCounter", HT(op))] = fmt::format("d_{0}_PreCounter", HT(op));      
   appendControl(fmt::format("cudaMallocExt(&d_{0}_PreCounter, sizeof(uint64_t));", HT(op)));
   deviceFrees.insert(fmt::format("d_{0}_PreCounter", HT(op)));
   appendControl(fmt::format("cudaMemset(d_{0}_PreCounter, 0, sizeof(uint64_t));", HT(op)));

   // host
   appendControlDecl(fmt::format("uint64_t {0}_PreCounter;", HT(op)));
}

void TupleStreamCode::AddPostHTProbeCounter(mlir::Operation* op) {
   if (!gPrintHashTableSizes) 
      return;

   // device
   appendControlDecl(fmt::format("uint64_t* d_{0}_PostCounter = nullptr;", HT(op)));      
   appendKernel(fmt::format("atomicAdd((int*){0}_PostCounter, 1);", HT(op)), KernelType::Main);
   mainArgs[fmt::format("{0}_PostCounter", HT(op))] = "uint64_t*";
   mlirToGlobalSymbol[fmt::format("{0}_PostCounter", HT(op))] = fmt::format("d_{0}_PostCounter", HT(op));      
   appendControl(fmt::format("cudaMallocExt(&d_{0}_PostCounter, sizeof(uint64_t));", HT(op)));
   deviceFrees.insert(fmt::format("d_{0}_PostCounter", HT(op)));
   appendControl(fmt::format("cudaMemset(d_{0}_PostCounter, 0, sizeof(uint64_t));", HT(op)));

   // host
   appendControlDecl(fmt::format("uint64_t {0}_PostCounter;", HT(op)));
}

} // namespace cudacodegen

// --- [start] code generation switches helpers ---

bool gStaticMapOnly = true;
bool gUseBloomFiltersForJoin = false;
bool gThreadsAlwaysAlive = true;
bool gPyperShuffle = false;
bool gCompilingSSB = false;
bool gShuffleAllOps = false;
bool gPrintHashTableSizes = false;
bool gEnableLogging = false;
BloomFilterPolicy gBloomFilterPolicy = AddBloomFiltersToAllJoins;
bool gTwoItemsPerThread = false;
bool gOneItemPerThread = false; // TODO: Make this an enum or an int config
bool gTileHashTables = false;
bool gUseBallotShuffle = false;

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

static void checkForCodegenSwitch(int &argc, char** argv, bool* config, const std::string& switchName, const std::string& descr) {
   for (int i = 0; i < argc; i++) {
      if (std::string(argv[i]) == switchName) {
         std::clog << "Enabled " << descr << "\n";
         *config = true;
         removeCodeGenSwitch(argc, argv, i);
         break;
      }
   }
}

void checkForBloomFilterOptions(int& args, char **argv) {
   bool hasBloomFilterFlags = false;
   auto bloom_filter_configs = {
      std::make_tuple("--bloom-filter-policy-large-ht", BloomFilterLargeHT, "Add bloom filter only when HT is larger than L2 cache"),
      std::make_tuple("--bloom-filter-policy-large-ht-small-bf", BloomFilterLargeHTSmallBF, "Add bloom filter only when HT is larger than L2 cache and the bloom filter can fit in L2 cache"),
      std::make_tuple("--bloom-filter-policy-large-ht-fit-bf", BloomFilterLargeHTFitBF, "Add bloom filter only when HT is larger than L2 cache, but fit the bloom filter to L2 cache")
   };
   for (const auto& [switchName, policy, descr] : bloom_filter_configs) {
      for (int i = 0; i < args; i++) {
         if (std::string(argv[i]) == switchName) {
            if (gBloomFilterPolicy != AddBloomFiltersToAllJoins) {
               std::cerr << "A bloom filter policy has already been set. Only one bloom filter policy can be set at a time." << std::endl;
               exit(1);
            }
            std::clog << "Enabled " << descr << "\n";
            gBloomFilterPolicy = policy;
            hasBloomFilterFlags = true;
            removeCodeGenSwitch(args, argv, i);
            break;
         }
      }
   }
   if (!gUseBloomFiltersForJoin && hasBloomFilterFlags) {
      std::cerr << "Warning: Bloom filters for join are disabled. To enable, use --use-bloom-filters option.\n";
      exit(1);
   }
}

void checkForCodeGenSwitches(int& argc, char** argv) {
   std::tuple<bool*, std::string, std::string> switches[]  = {
      std::make_tuple(&gStaticMapOnly, "--static-map-only", "static map only code generation"),
      std::make_tuple(&gUseBloomFiltersForJoin, "--use-bloom-filters", "bloom filters for join"),
      std::make_tuple(&gThreadsAlwaysAlive, "--threads-always-alive", "threads always alive"),
      std::make_tuple(&gPyperShuffle, "--pyper-shuffle", "pyper shuffle code generation"),
      std::make_tuple(&gGenKernelTimingCode, "--gen-kernel-timing", "kernel timing code generation"),
      std::make_tuple(&gGenPerOperationProfile, "--gen-per-operation-profile", "per operation profile code generation"),
      std::make_tuple(&gSmallerHashTables, "--smaller-hash-tables", "smaller hash tables"),
      std::make_tuple(&gGenIsProfiling, "--profiling", "profiling code generation"),
      std::make_tuple(&gShuffleAllOps, "--shuffle-all-ops", "shuffling all ops"),
      std::make_tuple(&gPrintHashTableSizes, "--print-hash-table-sizes", "print hash table sizes"),
      std::make_tuple(&gPartitionHashJoinCodeGenEnabled, "--use-partition-hash-join", "partitioned hash join code generation"),
      std::make_tuple(&gEnableLogging, "--enable-logging", "enable logging"),
      std::make_tuple(&gTwoItemsPerThread, "--two-items-per-thread", "use two items per thread for crystal codegen"),
      std::make_tuple(&gOneItemPerThread, "--one-item-per-thread", "use one item per thread for crystal codegen"),
      std::make_tuple(&gTileHashTables, "--tile-hashtables", "tiled hash tables"),
      std::make_tuple(&gUseBallotShuffle, "--use-ballot-shuffle", "use ballot based shuffle idx generation")
   };
   for (const auto& [switchPtr, switchName, descr] : switches) {
      checkForCodegenSwitch(argc, argv, switchPtr, switchName, descr);
   }

   // assert that gShuffleAllOps and gPyperShuffle can only be enabled if gThreadsAlwaysAlive is true
   assert((gThreadsAlwaysAlive || !gShuffleAllOps || !gPyperShuffle) && "gThreadsAlwaysAlive must be true if gShuffleAllOps or gPyperShuffle is enabled");
   // assert that only one of gShuffleAllOps or gGeneratingShuffles must be enabled. Or both must be disabled.

   checkForBloomFilterOptions(argc, argv);
}

// --- [end] code generation switches helpers ---


bool isPrimaryKey(const std::set<std::string>& keysSet) {
   bool pk = false;
   std::vector<std::string> tpch_pks = {
      "o_orderkey", "r_regionkey", "c_custkey", "n_nationkey", "s_suppkey", "p_partkey"};
   std::vector<std::string> ssb_pks = {
      "p_partkey", "s_suppkey", "d_datekey", "c_custkey"};
   if (!gCompilingSSB) {
      for (auto k : tpch_pks) {
         if (keysSet.find(k) != keysSet.end()) pk = true;
      }
      pk |= (keysSet.find("ps_partkey") != keysSet.end() && keysSet.find("ps_suppkey") != keysSet.end());
   } else {
      for (auto k : ssb_pks) {
         if (keysSet.find(k) != keysSet.end()) pk = true;
      }
   }
   return pk;
}

bool invertJoinIfPossible(std::set<std::string>& rightkeysSet, bool left_pk) {
   if (left_pk == false && gStaticMapOnly) {
      // check if right side is a pk
      bool right_pk = isPrimaryKey(rightkeysSet);
      if (right_pk == false && gStaticMapOnly) {
         assert(false && "This join is not possible without multimap, since both sides are not pk");
      }
      if (right_pk == true && gStaticMapOnly) {
         return true;
      }
   }
   return false;
}

void emitTimingEventCreation(std::ostream& outputFile) {
   if (generateKernelTimingCode()) {
      outputFile << "cudaEvent_t start, stop;" << std::endl;
      outputFile << "cudaEventCreate(&start);" << std::endl;
      outputFile << "cudaEventCreate(&stop);" << std::endl;
   }
}

void emitControlFunctionSignature(std::ostream& outputFile) {
   if (!gCompilingSSB)
      outputFile << "extern \"C\" void control (DBI32Type * d_nation__n_nationkey, DBStringType * d_nation__n_name, DBI32Type * d_nation__n_regionkey, DBStringType * d_nation__n_comment, size_t nation_size, DBI32Type * d_supplier__s_suppkey, DBI32Type * d_supplier__s_nationkey, DBStringType * d_supplier__s_name, DBStringType * d_supplier__s_address, DBStringType * d_supplier__s_phone, DBDecimalType * d_supplier__s_acctbal, DBStringType * d_supplier__s_comment, size_t supplier_size, DBI32Type * d_partsupp__ps_suppkey, DBI32Type * d_partsupp__ps_partkey, DBI32Type * d_partsupp__ps_availqty, DBDecimalType * d_partsupp__ps_supplycost, DBStringType * d_partsupp__ps_comment, size_t partsupp_size, DBI32Type * d_part__p_partkey, DBStringType * d_part__p_name, DBStringType * d_part__p_mfgr, DBStringType * d_part__p_brand, DBStringType * d_part__p_type, DBI32Type * d_part__p_size, DBStringType * d_part__p_container, DBDecimalType * d_part__p_retailprice, DBStringType * d_part__p_comment, size_t part_size, DBI32Type * d_lineitem__l_orderkey, DBI32Type * d_lineitem__l_partkey, DBI32Type * d_lineitem__l_suppkey, DBI64Type * d_lineitem__l_linenumber, DBDecimalType * d_lineitem__l_quantity, DBDecimalType * d_lineitem__l_extendedprice, DBDecimalType * d_lineitem__l_discount, DBDecimalType * d_lineitem__l_tax, DBCharType * d_lineitem__l_returnflag, DBCharType * d_lineitem__l_linestatus, DBI32Type * d_lineitem__l_shipdate, DBI32Type * d_lineitem__l_commitdate, DBI32Type * d_lineitem__l_receiptdate, DBStringType * d_lineitem__l_shipinstruct, DBStringType * d_lineitem__l_shipmode, DBStringType * d_lineitem__comments, size_t lineitem_size, DBI32Type * d_orders__o_orderkey, DBCharType * d_orders__o_orderstatus, DBI32Type * d_orders__o_custkey, DBDecimalType * d_orders__o_totalprice, DBI32Type * d_orders__o_orderdate, DBStringType * d_orders__o_orderpriority, DBStringType * d_orders__o_clerk, DBI32Type * d_orders__o_shippriority, DBStringType * d_orders__o_comment, size_t orders_size, DBI32Type * d_customer__c_custkey, DBStringType * d_customer__c_name, DBStringType * d_customer__c_address, DBI32Type * d_customer__c_nationkey, DBStringType * d_customer__c_phone, DBDecimalType * d_customer__c_acctbal, DBStringType * d_customer__c_mktsegment, DBStringType * d_customer__c_comment, size_t customer_size, DBI32Type * d_region__r_regionkey, DBStringType * d_region__r_name, DBStringType * d_region__r_comment, size_t region_size, DBI16Type* d_nation__n_name_encoded, std::unordered_map<DBI16Type, DBStringType> &nation__n_name_map, std::unordered_map<DBI16Type, DBStringType> &n1___n_name_map, std::unordered_map<DBI16Type, DBStringType> &n2___n_name_map, DBI16Type* d_orders__o_orderpriority_encoded, std::unordered_map<DBI16Type, std::string>& orders__o_orderpriority_map, DBI16Type* d_customer__c_name_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_name_map, DBI16Type* d_customer__c_comment_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_comment_map, DBI16Type* d_customer__c_phone_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_phone_map, DBI16Type* d_customer__c_address_encoded, std::unordered_map<DBI16Type, std::string>& customer__c_address_map, DBI16Type* d_supplier__s_name_encoded, std::unordered_map<DBI16Type, std::string>& supplier__s_name_map, DBI16Type* d_part__p_brand_encoded, std::unordered_map<DBI16Type, std::string>& part__p_brand_map, DBI16Type* d_part__p_type_encoded, std::unordered_map<DBI16Type, std::string>& part__p_type_map, DBI16Type* d_lineitem__l_shipmode_encoded, std::unordered_map<DBI16Type, std::string>& lineitem__l_shipmode_map, DBI16Type* d_supplier__s_address_encoded, std::unordered_map<DBI16Type, std::string>& supplier__s_address_map) {\n";
   else
      outputFile << "extern \"C\" void control (DBI32Type* d_supplier__s_suppkey, DBStringType* d_supplier__s_name, DBStringType* d_supplier__s_address, DBStringType* d_supplier__s_city, DBStringType* d_supplier__s_nation, DBStringType* d_supplier__s_region, DBStringType* d_supplier__s_phone, size_t supplier_size, DBI32Type* d_part__p_partkey, DBStringType* d_part__p_name, DBStringType* d_part__p_mfgr, DBI16Type* d_part__p_mfgr_encoded, DBStringType* d_part__p_category, DBStringType* d_part__p_brand1, DBStringType* d_part__p_color, DBStringType* d_part__p_type, DBI32Type* d_part__p_size, DBStringType* d_part__p_container, size_t part_size, DBI32Type* d_lineorder__lo_orderkey, DBI32Type* d_lineorder__lo_linenumber, DBI32Type* d_lineorder__lo_custkey, DBI32Type* d_lineorder__lo_partkey, DBI32Type* d_lineorder__lo_suppkey, DBDateType* d_lineorder__lo_orderdate, DBDateType* d_lineorder__lo_commitdate, DBStringType* d_lineorder__lo_orderpriority, DBCharType* d_lineorder__lo_shippriority, DBI32Type* d_lineorder__lo_quantity, DBDecimalType* d_lineorder__lo_extendedprice, DBDecimalType* d_lineorder__lo_ordtotalprice, DBDecimalType* d_lineorder__lo_revenue, DBDecimalType* d_lineorder__lo_supplycost, DBI32Type* d_lineorder__lo_discount, DBI32Type* d_lineorder__lo_tax, DBStringType* d_lineorder__lo_shipmode, size_t lineorder_size, DBI32Type* d_date__d_datekey, DBStringType* d_date__d_date, DBStringType* d_date__d_dayofweek, DBStringType* d_date__d_month, DBI32Type* d_date__d_year, DBI32Type* d_date__d_yearmonthnum, DBStringType* d_date__d_yearmonth, DBI32Type* d_date__d_daynuminweek, DBI32Type* d_date__d_daynuminmonth, DBI32Type* d_date__d_daynuminyear, DBI32Type* d_date__d_monthnuminyear, DBI32Type* d_date__d_weeknuminyear, DBStringType* d_date__d_sellingseason, DBI32Type* d_date__d_lastdayinweekfl, DBI32Type* d_date__d_lastdayinmonthfl, DBI32Type* d_date__d_holidayfl, DBI32Type* d_date__d_weekdayfl, size_t date_size, DBI32Type* d_customer__c_custkey, DBStringType* d_customer__c_name, DBStringType* d_customer__c_address, DBStringType* d_customer__c_city, DBStringType* d_customer__c_nation, DBStringType* d_customer__c_region, DBI16Type* d_customer__c_region_encoded, DBStringType* d_customer__c_phone, DBStringType* d_customer__c_mktsegment, size_t customer_size, DBI32Type* d_region__r_regionkey, DBStringType* d_region__r_name, DBStringType* d_region__r_comment, size_t region_size, DBI16Type* d_part__p_brand1_encoded, DBI16Type* d_supplier__s_nation_encoded,DBI16Type* d_supplier__s_region_encoded, DBI16Type* d_customer__c_city_encoded, DBI16Type* d_supplier__s_city_encoded, DBI16Type* d_customer__c_nation_encoded, DBI16Type* d_part__p_category_encoded, std::unordered_map<DBI16Type, std::string>& part__p_brand1_map, std::unordered_map<DBI16Type, std::string>& supplier__s_nation_map, std::unordered_map<DBI16Type, std::string>& customer__c_city_map, std::unordered_map<DBI16Type, std::string>& supplier__s_city_map, std::unordered_map<DBI16Type, std::string>& customer__c_nation_map, std::unordered_map<DBI16Type, std::string>& part__p_category_map) {\n";
}