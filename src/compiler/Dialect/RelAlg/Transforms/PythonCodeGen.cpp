
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
enum class KernelType {
	Main, 
	Count
};

// for all the cudaidentifier that create a state for example join, aggregation, use the operation address
// instead of the stream address, which ensures that uniqueness for the data structure used by the operation
// is maintained
std::string convertToHex(void* op)
{
	std::stringstream sstream;
	sstream << std::hex << (unsigned long long)(void*)op;
	std::string result = sstream.str();
	return result;
}
typedef std::map<std::string, std::string> RIDMAP;
typedef std::set<std::string> LOADEDCOLUMNS;
static std::string mlirTypeToCudaType(mlir::Type ty) {
			if (mlir::isa<db::StringType>(ty))
				return "char* ";
			else if (ty.isInteger(32))
				return  "int32_t* ";
			else if (mlir::isa<db::DecimalType>(ty))
				return  "float* ";
			else if (mlir::isa<db::StringType>(ty))
				return  "StringColumn* ";
			else if (mlir::isa<db::DateType>(ty))
				return  "int32_t* ";
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
	LOADEDCOLUMNS loadedColumns;
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
	std::string launchKernel(KernelType ktype) {
		std::string type;
		if (ktype == KernelType::Main)
			type = "main";
		else 
			type = "count";
		std::string res = type + "_pipeline_" + convertToHex((void*)this);	
		res += "<<<std::ceil((float)" + baseRelation[baseRelation.size()-1] + "_size/(float)" + std::to_string(threadBlockSize) + "), " + std::to_string(threadBlockSize) + ">>>(";
		auto i=0ull;
		if (ktype == KernelType::Main)
		for (auto p: kernelArgs) {
			res += "d_" + p.first + ", ";
		}
		else
		for (auto p: kernelCountArgs) {
			res += "d_" + p.first + ", ";
		}
		if (ktype == KernelType::Main)
		for (auto p: stateArgs) {
			if (i < stateArgs.size()-1)
				res += "d_" + p.first + ", ";
			else 
				res += "d_" + p.first + ");";
			i++;
		}
		else 
		for (auto p: stateArgs) {
			if (i < stateArgs.size()-1)
				res += "d_" + p.first + ", ";
			else 
				res += "d_" + p.first + ");";
			i++;
		}
		return res;
	}
	void printCountKernel() {

		std::cout << "kernelCountArgs -> /\n";
		for (auto p: kernelCountArgs) {
			std::cout << p.first << ": " ;
			std::cout << mlirTypeToCudaType(p.second);
			std::cout <<  ", ";
		}
		for (auto p: stateCountArgs) {
			std::cout << p.second << " " << p.first << ", ";
		}
		std::cout << std::endl;
		std::cout << kernelCountCode << "\n";
	}
	void printKernel() {
		std::cout << "kernelArgs -> \n";
		for (auto p: kernelArgs) {
			std::cout << p.first << ": " ;
			std::cout << mlirTypeToCudaType(p.second);
			std::cout <<  ", ";
		}
		for (auto p: stateArgs) {
			std::cout << p.second << " " << p.first << ", ";
		}
		std::cout << std::endl;
		std::cout << kernelCode << "\n";
	}
	void printControl() {
		std::cout << controlCode + "\n";
	}
	void printCountControl() {
		std::cout << controlCountCode + "\n";
	}
};

template<typename ColumnAttrTy>
std::string getColumnName(const ColumnAttrTy &colAttr) {
	
	for (auto n: colAttr.getName().getNestedReferences()) {
		return n.getAttr().str();
	}
	assert(false && "No column for columnattr found");
	return "";
}

template<typename ColumnAttrTy>
static std::string getTableName(const ColumnAttrTy &colAttr)
{
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
		std::cout << std::endl;
	}

	ColumnDetail(const tuples::ColumnRefAttr &colAttr) {
		relation = getTableName<tuples::ColumnRefAttr>(colAttr);
		name = getColumnName<tuples::ColumnRefAttr>(colAttr);
		type = colAttr.getColumn().type;
	}
};


std::string LoadColumnIntoStream(TupleStreamCode *streamCode, const tuples::ColumnRefAttr &colAttr, KernelType type ) {
	// add to the kernel argument, get the name and type from colAttr
	ColumnDetail detail(colAttr);
	if (type == KernelType::Main)
		streamCode->kernelArgs[detail.relation + "__" + detail.name] = detail.type; // add information to the arguments
	else {
		streamCode->kernelCountArgs[detail.relation + "__" + detail.name] = detail.type; // add information to the arguments
	}
	std::string cudaIdentifier = "reg__" + detail.relation + "__" + detail.name; 
	if (type == KernelType::Main) {

		if (streamCode->loadedColumns.find(cudaIdentifier) == streamCode->loadedColumns.end()) {
			// load the column into register 
			streamCode->loadedColumns.insert(cudaIdentifier);
			streamCode->appendKernel("auto reg__" + detail.relation + "__" + detail.name + " = " + detail.relation + "__" + detail.name + "[" + streamCode->ridMap[detail.relation] + "];");
		}
		assert(streamCode->loadedColumns.find(cudaIdentifier) != streamCode->loadedColumns.end());
	}
	else {
		if (streamCode->loadedCountColumns.find(cudaIdentifier) == streamCode->loadedCountColumns.end()) {
			streamCode->loadedCountColumns.insert(cudaIdentifier);
			streamCode->appendCountKernel("auto reg__" + detail.relation + "__" + detail.name + " = " + detail.relation + "__" + detail.name + "[" + streamCode->ridMap[detail.relation] + "];");
		}
		assert(streamCode->loadedCountColumns.find(cudaIdentifier) != streamCode->loadedCountColumns.end());
	}
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

static std::string translateExpression(mlir::Region& expression, TupleStreamCode *streamCode) {
	// walk over the region block
	std::vector<mlir::Operation*> toMove;
	for (auto& op: expression.front()) {
		toMove.push_back(&op);
	}
	return "";
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
					LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Count);// selection always needs to be done in count code as well
					leftOperand = LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Main);
				} else {
					leftOperand = operandToString(left.getDefiningOp());
				}

				auto right = compareOp.getRight();
				std::string rightOperand;
				if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(right.getDefiningOp())) {
					LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Count); // selection always needs to be done in count code as well
					rightOperand = LoadColumnIntoStream(streamCode, getColOp.getAttr(), KernelType::Main);
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

static std::string d_HT(void* op)
{
	return "d_HT_" + convertToHex(op);
}
static std::string HT(void* op)
{
	return "HT_" + convertToHex(op);
}
static std::string KEY(void*  op)
{
	return "KEY_" + convertToHex(op);
}
static std::string SLOT( void* op)
{
	return "SLOT_" + convertToHex( op);
}
static std::string d_BUF( void* op)
{
	return "d_BUF_" + convertToHex( op);
}
static std::string d_BUF_IDX( void* op)
{
	return "d_BUF_IDX_" + convertToHex( op);
}
static std::string BUF( void* op)
{
	return "BUF_" + convertToHex( op);
}
static std::string BUF_IDX( void* op)
{
	return "BUF_IDX_" + convertToHex( op);
}
static std::string buf_idx( void* op)
{
	return "buf_idx_" + convertToHex( op);
}

static std::string MakeKeysInStream(mlir::Operation* op, TupleStreamCode* stream, const mlir::ArrayAttr &keys, KernelType kernelType) {
	std::string keyMakerString = ("int64_t " + KEY(op) + " = make_keys(");
	for (auto i = 0ull; i<keys.size(); i++) {
		tuples::ColumnRefAttr key = mlir::cast<tuples::ColumnRefAttr>(keys[i]);
		std::string cudaIdentifierKey = LoadColumnIntoStream(stream, key, kernelType);
		keyMakerString += cudaIdentifierKey;
		if (i != keys.size() - 1)
		{
			keyMakerString += ", ";
		}
		else {
			keyMakerString += ")";
		}
	}
	if (kernelType == KernelType::Main)
		stream->appendKernel(keyMakerString);
	else 
		stream->appendCountKernel(keyMakerString);
	return KEY(op);
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
				streamCode->appendCountKernel("if (!(" + condition + ")) return;");
				streamCodeMap[op] = streamCode;
				/**
					  * TODO(avinash): check if the implemented predicate in python is good
					  *     after re-evaluating its design, handle the predicate by the region: getPredicate
					  */

				//this is basically produce code for the scan
			}
			else if (auto aggregation = llvm::dyn_cast<relalg::AggregationOp>(op)) {
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

				mlir::ArrayAttr groupByKeys = aggregation.getGroupByCols(); // these are columnrefs 
				mlir::ArrayAttr computedCols = aggregation.getComputedCols(); // these are columndefs

				// create hash table for the stream for this aggregation
				streamCode->stateArgs[HT(op)] = "HASHTABLE_INSERT";

				// compute the keys
				auto cudaIdentifierKey = MakeKeysInStream(op, streamCode, groupByKeys, KernelType::Main);
				// just insert a dummy value with this key, which will later be handled in user space
				// TODO(avinash): this is actually supposed to go into the count kernel
				streamCode->appendKernel(HT(op) + ".insert(cuco::pair{" + cudaIdentifierKey + ", 1});");

				streamCode->appendKernel("//After inserting appropriate indices into the hash table, we next take this index and do atomic aggregation on it");
				
				for (auto &col: computedCols) {
					std::string colName = getColumnName<tuples::ColumnDefAttr>(mlir::cast<tuples::ColumnDefAttr>(col));
					std::string tableName = getTableName<tuples::ColumnDefAttr>(mlir::cast<tuples::ColumnDefAttr>(col));
					// create buffers of aggregation length obtained in the count kernel, and create new buffers in the control code
					streamCode->kernelArgs[tableName + "__" + colName] = (mlir::cast<tuples::ColumnDefAttr>(col)).getColumn().type;
				}
				streamCode->appendKernel(buf_idx(op) + " = " + HT(op) + ".find(" + cudaIdentifierKey + ");");
				// walk through the region
				auto& aggRgn = aggregation.getAggrFunc();
				//TODO(avinash, p1): aggregation add to kernel, computed aggregate_fn(&buffer, computed_col[idx])

				std::map<mlir::Operation*, std::string> newColumnMap;
				if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(aggRgn.front().getTerminator())) {

					int i = 0;
					for (mlir::Value col: returnOp.getResults()) {
						// map each aggrfunc which is col.getDefiningOp to computedColName
						auto newcol = mlir::cast<tuples::ColumnDefAttr>(computedCols[i]);
						auto newbuffername = getTableName<tuples::ColumnDefAttr>(newcol) + "__" + getColumnName<tuples::ColumnDefAttr>(newcol);
						newColumnMap[col.getDefiningOp()] = newbuffername;
						i++;
					}
				} else {
					assert(false && "nothing to aggregate!!");
				}
				for (auto &regionOp: aggRgn.front()) {
					// now materialize all the computed columns here
					if (auto aggrFunc = llvm::dyn_cast<relalg::AggrFuncOp>(regionOp)) {
						auto fn = aggrFunc.getFn();
						tuples::ColumnRefAttr col = aggrFunc.getAttr(); // we dont need the tuplestream that is getRel here for now.
						auto colName = getColumnName<tuples::ColumnRefAttr>(col);
						auto tableName = getTableName<tuples::ColumnRefAttr>(col);

						auto cudaRegIdentifier = LoadColumnIntoStream(streamCode, col, KernelType::Main);
						assert(streamCode->kernelArgs.find(tableName + "__" + colName) != streamCode->kernelArgs.end() && "existing column (input to aggregation) not found in kernel args.");
						
						auto slot = newColumnMap[&regionOp];
						assert(streamCode->kernelArgs.find(slot) != streamCode->kernelArgs.end() && "the new column is not in the kernel args.");

						slot += "[" + buf_idx(op) + "]";

						// how do you get the slot, that is which column corresponds to the computed column??
						// the return values have one to one corr to computed cold
						switch (fn) {
							case relalg::AggrFunc::sum : {
								streamCode->appendKernel("aggregate_sum(&" + slot + ", " + cudaRegIdentifier + ");"); 
							}
							break;
							case relalg::AggrFunc::count :{
								streamCode->appendKernel("aggregate_count(&" + slot + ", " + cudaRegIdentifier + ");"); 
							}
							break;
							case relalg::AggrFunc::any : {
								std::clog << "found any\n";
								streamCode->appendKernel("aggregate_any(&" + slot + ", " + cudaRegIdentifier + ");"); 
							}
							break;
							case relalg::AggrFunc::avg : {
								assert(false && "average should be split into sum and divide");
							}
							break;
							case relalg::AggrFunc::min : {
								streamCode->appendKernel("aggregate_min(&" + slot + ", " + cudaRegIdentifier + ");"); 
							}
							break;
							case relalg::AggrFunc::max : {
								streamCode->appendKernel("aggregate_max(&" + slot + ", " + cudaRegIdentifier + ");"); 
							}
							break;
							default:
								assert(false && "this aggregation is not handled");
							break;
						}
					}
				}

				streamCode->appendKernel("return;");
				kernelSchedule.push_back(streamCode);
				// any upstream op should start a new kernel.
				// for example the topk. 
				streamCodeMap[op] = streamCode;
		}
		else if (auto table_scan = llvm::dyn_cast<relalg::BaseTableOp>(op)) {
				std::string tableIdentifier = table_scan.getTableIdentifier().data();
				TupleStreamCode* streamCode = new TupleStreamCode();
				streamCode->stateCountArgs[tableIdentifier + "_size"] = "uint64_t";
				streamCode->stateArgs[tableIdentifier + "_size"] = "uint64_t";
				streamCode->baseRelation.push_back(tableIdentifier);
				streamCode->ridMap[tableIdentifier] = "tid";
				streamCode->appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
				streamCode->appendKernel("if (tid >= " + tableIdentifier + "_size) return;");
				streamCode->appendCountKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
				streamCode->appendCountKernel("if (tid >= " + tableIdentifier + "_size) return;");
				streamCodeMap[op] = streamCode;
		}
		else if (auto mapOp = llvm::dyn_cast<relalg::MapOp>(op)) {
			// TODO(avinash): Scheduled on 9th march
				mlir::Operation* stream = mapOp.getRelMutable().get().getDefiningOp();
				TupleStreamCode* streamCode = streamCodeMap[stream];
				// get the array attribute, that is computed cols
				// these will be the new definitions of the columns, and we need this information
				// in the upstream tuple operation
				auto computedCols = mapOp.getComputedCols(); // returns mlir::ArrayAttr
				// each element here is a columndefattr, it is fairly easy to retrieve the table/column names
				// the predicate region has the computation graph for the expression that we need to translate

				translateExpression(mapOp.getPredicate(), streamCode);

				streamCode->appendKernel("auto mapped_register = expression(attributes, constants, operations);");
				streamCodeMap[op] = streamCode;
			}
		else if (auto joinOp = llvm::dyn_cast<relalg::InnerJoinOp>(op)) {
				// left side is a materialization point, so end the kernel and push it to the pipelineschedules
				// Generate 2 kernels one to get the count, and another to fill in the buffers
				mlir::Operation* leftStream = joinOp.getLeftMutable().get().getDefiningOp();
				TupleStreamCode* leftStreamCode = streamCodeMap[leftStream];

				leftStreamCode->appendCountKernel("atomicAdd(" + BUF_IDX(op) + ", 1);");
				leftStreamCode->appendCountKernel("return;");
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
				for (auto br: leftStreamCode->baseRelation) {
					leftStreamCode->appendKernel(BUF(op) + "[" + buf_idx(op) + " * " + std::to_string(leftStreamCode->baseRelation.size())
							+ " + " + std::to_string(i) + "] = " + leftStreamCode->ridMap[br] + ";"); 
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
				leftStreamCode->appendControl("cudaMemcpy(" + BUF_IDX(op) + ", " + d_BUF_IDX(op) + ", sizeof(int), cudaMemcpyDeviceToHost);"); 
				leftStreamCode->appendControl("cudaMemset(" + d_BUF_IDX(op) + ",0 , sizeof(int));");
				leftStreamCode->appendControl("auto " + d_HT(op) + " = cuco::static_map{ " + BUF_IDX(op) + "* 2,cuco::empty_key{(int64_t)-1},cuco::empty_value{(int64_t)-1},thrust::equal_to<int64_t>{},cuco::linear_probing<1, cuco::default_hash_function<int64_t>>()};");
				leftStreamCode->appendControl("uint64_t *" + d_BUF(op) + ";"); 
				leftStreamCode->appendControl("cudaMalloc(&" + d_BUF(op) + ", sizeof(uint64_t) * " + std::to_string(leftStreamCode->baseRelation.size()) + " * " + BUF_IDX(op) + ");");
				leftStreamCode->appendControl("cudaMemset(" + d_BUF(op) + ",0 , sizeof(int));");
				leftStreamCode->appendControl(leftStreamCode->launchKernel(KernelType::Main));

				kernelSchedule.push_back(leftStreamCode);
				
				// continue the right stream code gen
				// TODO(avinash): Add predicate region handling for the probe side
				
				mlir::Operation* rightStream = joinOp.getRightMutable().get().getDefiningOp();
				TupleStreamCode* rightStreamCode = streamCodeMap[rightStream];
				auto rightHash = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
				MakeKeysInStream(op, rightStreamCode, rightHash, KernelType::Count);
				auto cudaIdentifierRightKey = MakeKeysInStream(op, rightStreamCode, rightHash, KernelType::Main);
				rightStreamCode->stateCountArgs[BUF(op)] = "uint64_t*";
				rightStreamCode->stateCountArgs[HT(op)] = "HASHTABLE_FIND";
				rightStreamCode->stateArgs[BUF(op)] = "uint64_t*";
				rightStreamCode->stateArgs[HT(op)] = "HASHTABLE_FIND";
				rightStreamCode->appendCountKernel("auto " + SLOT(op) + " = " + HT(op) + ".find(" + cudaIdentifierRightKey +");");
				rightStreamCode->appendCountKernel("auto " + buf_idx(op) + " = " + SLOT(op) + "->second;");
				rightStreamCode->appendKernel("auto " + SLOT(op) + " = " + HT(op) + ".find(" + cudaIdentifierRightKey +");");
				rightStreamCode->appendKernel("auto " + buf_idx(op) + " = " + SLOT(op) + "->second;");
				i = 0;
				// emplace
				for (auto br: leftStreamCode->baseRelation) {
					auto rbr_beg = rightStreamCode->baseRelation.begin();
					rightStreamCode->baseRelation.emplace(rbr_beg + i, br);
					rightStreamCode->ridMap[br] = BUF(op) + "[" + buf_idx(op) + "*" + std::to_string(leftStreamCode->baseRelation.size()) + " + " + std::to_string(i) + "]";
					i++;
				}

				// upstream operator would use the probe side of the hashjoin
				streamCodeMap[op] = rightStreamCode;
			}
		});
		for (auto code : kernelSchedule) {
			code->printCountKernel();
			code->printKernel();
		}

		for (auto code: kernelSchedule) {
			code->printCountControl();
			code->printControl();
		}
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
