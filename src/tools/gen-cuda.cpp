#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/Timing.h"
#include "lingodb/scheduler/Scheduler.h"
#include "lingodb/utility/Setting.h"

#include <fstream>
#include <iostream>
#include <string>

extern std::string gOpFilePath; // TODO: This is bad coupling between codegen and main. Pass it as a param

namespace {
utility::GlobalSetting<bool> eagerLoading("system.eager_loading", false);
} // namespace
int main(int argc, char** argv) {
   using namespace lingodb;
   if (argc <= 2) {
      std::cerr << "USAGE: gen-sql database [<sql-file> <op-file>]" << std::endl;
      return 1;
   }   

   std::string directory = std::string(argv[1]);

   // Enable cuda code generation if required
   lingodb::compiler::dialect::relalg::conditionallyEnableCudaCodeGen(argc, argv);

   auto sql_files = std::vector<std::string>();
   auto op_files = std::vector<std::string>();
   for (int i = 2; i < argc; i+=2) {
      sql_files.push_back(std::string(argv[i]));
      op_files.push_back(std::string(argv[i+1]));
   }

   if ((argc - 2) <= 0 || (argc % 2 != 0)) {
      std::cerr << "USAGE: gen-sql database [<sql-file> <op-file>]" << std::endl;
      return 1;
   }

   std::cerr << "Loading Database from: " << directory << '\n';
   auto session = runtime::Session::createSession(directory, eagerLoading.getValue());


   for (size_t i = 0; i < sql_files.size(); i++) {
      std::string inputFileName = sql_files[i];
      gOpFilePath = op_files[i];
      std::cerr << "Processing SQL file: " << inputFileName << " with output file: " << gOpFilePath << '\n';
         
      lingodb::compiler::support::eval::init();
      execution::ExecutionMode runMode = execution::getExecutionMode();
      auto queryExecutionConfig = execution::createQueryExecutionConfig(runMode, true);
      if (const char* numRuns = std::getenv("QUERY_RUNS")) {
         queryExecutionConfig->executionBackend->setNumRepetitions(std::atoi(numRuns));
         std::cerr << "using " << queryExecutionConfig->executionBackend->getNumRepetitions() << " runs" << std::endl;
      }
      unsetenv("PERF_BUILDID_DIR");
      // Set timing processor to print timings to a file named after the input SQL file
      queryExecutionConfig->timingProcessor = std::make_unique<execution::TimingPrinter>(inputFileName);

      auto scheduler = scheduler::startScheduler();
      auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(queryExecutionConfig), *session);
      executer->fromFile(inputFileName);
      scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(std::move(executer)));
   }      
   return 0;
}
