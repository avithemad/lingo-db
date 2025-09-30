#ifndef LINGODB_EXECUTION_TIMING_H
#define LINGODB_EXECUTION_TIMING_H
#include "Error.h"

#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
namespace lingodb::execution {
class TimingProcessor {
   public:
   virtual void addTiming(const std::unordered_map<std::string, double>& timing) = 0;
   virtual void process() = 0;
   virtual ~TimingProcessor() {}
};
class TimingPrinter : public TimingProcessor {
   std::unordered_map<std::string, double> timing;
   std::string queryName;

   public:
   TimingPrinter(std::string queryFile) {
      if (queryFile.find('/') != std::string::npos) {
         queryName = queryFile.substr(queryFile.find_last_of("/\\") + 1);
      } else {
         queryName = queryFile;
      }
   }
   void addTiming(const std::unordered_map<std::string, double>& timing) override {
      this->timing.insert(timing.begin(), timing.end());
   }
   void process() override {
      double total = 0.0;
      for (auto [name, t] : timing) {
         total += t;
      }
      timing["total"] = total;
      std::vector<std::string> printOrder = {"QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA", "lowerToLLVM", "toLLVMIR", "llvmOptimize", "llvmCodeGen", "executionTime", "total"};
      std::cerr << std::endl
                << std::endl;
      std::cerr << std::setw(10) << "name";
      for (auto n : printOrder) {
         std::cerr << std::setw(15) << n;
      }
      std::cerr << std::endl;
      std::cerr << std::setw(10) << queryName;
      for (auto n : printOrder) {
         if (timing.contains(n)) {
            std::cerr << std::setw(15) << timing[n];
         } else {
            std::cerr << std::setw(15) << "";
         }
      }
   }
};
class CPUTimingPrinter : public TimingProcessor {
   std::unordered_map<std::string, double> timing;
   std::string queryName;
   std::string perf_file;

   public:
   CPUTimingPrinter(std::string perf_file, std::string queryFile) : perf_file(perf_file) {
      if (queryFile.find('/') != std::string::npos) {
         queryName = queryFile.substr(queryFile.find_last_of("/\\") + 1);
      } else {
         queryName = queryFile;
      }
   }
   void addTiming(const std::unordered_map<std::string, double>& timing) override {
      this->timing.insert(timing.begin(), timing.end());
   }
   void process() override {
      double total = 0.0;
      for (auto [name, t] : timing) {
         total += t;
      }
      timing["total"] = total;
      std::ofstream outfile(perf_file, std::ios::app);
      if (outfile.good()) {
         outfile << "---" << std::endl;
         outfile << "tpch-q" << queryName << std::endl;
         outfile << "total_query, " << timing["executionTime"] << std::endl;
      }
   }
};
} // namespace lingodb::execution
#endif //LINGODB_EXECUTION_TIMING_H
