
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
	std::vector<std::string> baseRelation; // in data centric code gen, each stream will have exactly one base relation where it scans from
	std::string kernelCode;
	std::string kernelCountCode;
	// argument represents type and name of kernel arguments
	// using map to preserve the order of arguments, kernelArgs[argument] = type
	std::map<std::string, mlir::Type> kernelArgs; // for storing database columns
	std::map<std::string, std::string> stateArgs; // for storing out custom data structures
	RIDMAP ridMap; // row identifier map. maps table to cuda identifier containing the RID of the table
	LOADEDCOLUMNS loadedColumns;
	TupleStreamCode() : kernelCode("") {}

	void appendKernel(std::string code) {
		kernelCode += code + "\n";
	}
	void print() {
		std::cout << "kernelArgs -> \n";
		for (auto p: kernelArgs) {
			if (mlir::isa<db::StringType>(p.second))
				std::cout << "char* ";
			else if (p.second.isInteger(32))
				std::cout << "int32_t* ";
			else if (mlir::isa<db::DecimalType>(p.second))
				std::cout << "float* ";
			else if (mlir::isa<db::StringType>(p.second))
				std::cout << "StringColumn* ";
			else if (mlir::isa<db::DateType>(p.second))
				std::cout << "int32_t* ";
			std::cout << p.first << ", " ;
			std::cout <<  " ";
		}
		for (auto p: stateArgs) {
			std::cout << p.second << " " << p.first << ", ";
		}
		std::cout << std::endl;
		std::cout << kernelCode << "\n";
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


std::string LoadColumnIntoStream(TupleStreamCode *streamCode, const tuples::ColumnRefAttr &colAttr) {
	// add to the kernel argument, get the name and type from colAttr
	ColumnDetail detail(colAttr);
	streamCode->kernelArgs[detail.relation + "__" + detail.name] = detail.type; // add information to the arguments
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

// for all the cudaidentifier that create a state for example join, aggregation, use the operation address
// instead of the stream address, which ensures that uniqueness for the data structure used by the operation
// is maintained
std::string convertToHex(mlir::Operation* op)
{
	std::stringstream sstream;
	sstream << std::hex << (unsigned long long)(void*)op;
	std::string result = sstream.str();
	return result;
}
static std::string HT(mlir::Operation* op)
{
	return "HT_" + convertToHex(op);
}
static std::string KEY(mlir::Operation* op)
{
	return "KEY_" + convertToHex(op);
}
static std::string SLOT(mlir::Operation* op)
{
	return "SLOT_" + convertToHex(op);
}
static std::string BUF(mlir::Operation* op)
{
	return "BUF_" + convertToHex(op);
}
static std::string BUF_IDX(mlir::Operation* op)
{
	return "BUF_IDX_" + convertToHex(op);
}
static std::string buf_idx(mlir::Operation* op)
{
	return "buf_idx_" + convertToHex(op);
}

static std::string MakeKeysInStream(mlir::Operation* op, TupleStreamCode* stream, const mlir::ArrayAttr &keys) {
	std::string keyMakerString = ("int64_t " + KEY(op) + " = make_keys(");
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
				auto cudaIdentifierKey = MakeKeysInStream(op, streamCode, groupByKeys);
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

						auto cudaRegIdentifier = LoadColumnIntoStream(streamCode, col);
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
				streamCode->stateArgs[tableIdentifier + "_size"] = "uint64_t";
				streamCode->baseRelation.push_back(tableIdentifier);
				streamCode->ridMap[tableIdentifier] = "tid";
				streamCode->appendKernel("size_t tid = blockIdx.x * blockDim.x + threadIdx.x;");
				streamCode->appendKernel("if (tid >= " + tableIdentifier + "_size) return;");
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
				auto leftHash = joinOp->getAttrOfType<mlir::ArrayAttr>("leftHash");
				auto cudaIdentifierLeftKey = MakeKeysInStream(op, leftStreamCode, leftHash);
				// increment the buffer idx, and insert tid into the buffer
				leftStreamCode->stateArgs[BUF_IDX(op)] = "uint64_t*";
				leftStreamCode->stateArgs[BUF(op)] = "uint64_t*";
				leftStreamCode->stateArgs[HT(op)] = "HASHTABLE_INSERT";
				leftStreamCode->appendKernel("auto " + buf_idx(op) + " = atomicAdd(" + BUF_IDX(op) + ", 1);");
				leftStreamCode->appendKernel(HT(op) + ".insert(cuco::pair{" + cudaIdentifierLeftKey + ", " + buf_idx(op) + "});");
				int i = 0;
				for (auto br: leftStreamCode->baseRelation) {
					leftStreamCode->appendKernel(BUF(op) + "[" + buf_idx(op) + " * " + std::to_string(leftStreamCode->baseRelation.size())
							+ " + " + std::to_string(i) + "] = " + leftStreamCode->ridMap[br] + ";"); 
					i++;
				}
				// load keys into the register
				leftStreamCode->appendKernel("return;"); // end the kernel
				kernelSchedule.push_back(leftStreamCode);
				
				// continue the right stream code gen
				// TODO(avinash): Add predicate region handling for the probe side
				
				mlir::Operation* rightStream = joinOp.getRightMutable().get().getDefiningOp();
				TupleStreamCode* rightStreamCode = streamCodeMap[rightStream];
				auto rightHash = joinOp->getAttrOfType<mlir::ArrayAttr>("rightHash");
				auto cudaIdentifierRightKey = MakeKeysInStream(op, rightStreamCode, rightHash);
				rightStreamCode->stateArgs[BUF(op)] = "uint64_t*";
				rightStreamCode->stateArgs[HT(op)] = "HASHTABLE_FIND";
				rightStreamCode->appendKernel("auto " + SLOT(op) + " = " + HT(op) + ".find(" + cudaIdentifierRightKey +");");
				rightStreamCode->appendKernel("auto " + buf_idx(op) + " = " + SLOT(op) + "->second;");
				i = 0;
				// emplace
				for (auto br: leftStreamCode->baseRelation) {
					auto rbr_beg = rightStreamCode->baseRelation.begin();
					rightStreamCode->baseRelation.emplace(rbr_beg + i, br);
					rightStreamCode->ridMap[br] = BUF(op) + "[" + buf_idx(op) + "]";
					i++;
				}

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
 * 6 = selection(lineitem, shipd/ate > 1995-03-15)
 * 7 = join(orders, lineitem)
 * 8 = join
 */
}

std::unique_ptr<mlir::Pass> relalg::createPythonCodeGenPass() { return std::make_unique<PythonCodeGen>(); }
