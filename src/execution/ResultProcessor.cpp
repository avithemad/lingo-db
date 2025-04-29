#include <iomanip>
#include <iostream>

#include <arrow/pretty_print.h>
#include <arrow/table.h>

#include <arrow/api.h> // Core Arrow
#include <arrow/csv/api.h> // For CSV API
#include <arrow/csv/writer.h> // For CSV Writer
#include <arrow/io/stdio.h> // For StdoutStream
#include <arrow/result.h> // For arrow::Result

#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/runtime/ArrowTable.h"
#include <functional>

namespace {
unsigned char hexval(unsigned char c) {
   if ('0' <= c && c <= '9')
      return c - '0';
   else if ('a' <= c && c <= 'f')
      return c - 'a' + 10;
   else if ('A' <= c && c <= 'F')
      return c - 'A' + 10;
   else
      abort();
}

class TableRetriever : public lingodb::execution::ResultProcessor {
   std::shared_ptr<arrow::Table>& result;

   public:
   TableRetriever(std::shared_ptr<arrow::Table>& result) : result(result) {}
   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(0);
      if (!resultTable) return;
      result = resultTable.value()->get();
   }
};

void printTable(const std::shared_ptr<arrow::Table>& table) {
   // Do not output anything for insert or copy statements
   if (table->columns().empty()) {
      std::cout << "Statement executed successfully." << std::endl;
      return;
   }

   std::vector<std::string> columnReps;
   std::vector<size_t> positions;
   arrow::PrettyPrintOptions options;
   options.indent_size = 0;
   options.window = 100;
   std::cout << "|";
   std::string rowSep = "-";
   std::vector<bool> convertHex;
   for (auto c : table->columns()) {
      std::cout << std::setw(30) << table->schema()->field(positions.size())->name() << "  |";
      convertHex.push_back(table->schema()->field(positions.size())->type()->id() == arrow::Type::FIXED_SIZE_BINARY);
      rowSep += std::string(33, '-');
      std::stringstream sstr;
      arrow::PrettyPrint(*c.get(), options, &sstr); //NOLINT (clang-diagnostic-unused-result)
      columnReps.push_back(sstr.str());
      positions.push_back(0);
   }
   std::cout << std::endl
             << rowSep << std::endl;
   bool cont = true;
   while (cont) {
      cont = false;
      bool skipNL = false;
      for (size_t column = 0; column < columnReps.size(); column++) {
         char lastHex = 0;
         bool first = true;
         std::stringstream out;
         while (positions[column] < columnReps[column].size()) {
            cont = true;
            char curr = columnReps[column][positions[column]];
            char next = columnReps[column][positions[column] + 1];
            positions[column]++;
            if (first && (curr == '[' || curr == ']' || curr == ',')) {
               continue;
            }
            if (curr == ',' && next == '\n') {
               continue;
            }
            if (curr == '\n') {
               break;
            } else {
               if (convertHex[column]) {
                  if (isxdigit(curr)) {
                     if (lastHex == 0) {
                        first = false;
                        lastHex = curr;
                     } else {
                        char converted = (hexval(lastHex) << 4 | hexval(curr));
                        out << converted;
                        lastHex = 0;
                     }
                  } else {
                     first = false;
                     out << curr;
                  }
               } else {
                  first = false;
                  out << curr;
               }
            }
         }
         if (first) {
            skipNL = true;
         } else {
            if (column == 0) {
               std::cout << "|";
            }
            std::cout << std::setw(30) << out.str() << "  |";
         }
      }
      if (!skipNL) {
         std::cout << "\n";
      }
   }
}

void printTableToCSV(const std::shared_ptr<arrow::Table>& table) {
   // Do not output anything for insert or copy statements
   if (table->columns().empty()) {
      std::cout << "Statement executed successfully." << std::endl;
      return;
   }
   // 2. Get an output stream for stdout
   arrow::io::StdoutStream output_stream;

   // 3. Configure CSV writing options (using defaults here)
   //    Default includes header, uses comma delimiter, empty string for nulls
   auto write_options = arrow::csv::WriteOptions::Defaults();
   // Example customization: write_options.include_header = false;
   // Example customization: write_options.delimiter = ';';
   // Example customization: write_options.null_string = "NA";

   if (!(WriteCSV(*table, write_options, &output_stream).ok())) {
      std::cerr << "Failed to write table to CSV";
   }
}

class TablePrinter : public lingodb::execution::ResultProcessor {
   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(0);
      if (!resultTable) return;
      auto table = resultTable.value()->get();
      // printTable(table);
      printTableToCSV(table);
   }
};
class BatchedTablePrinter : public lingodb::execution::ResultProcessor {
   void process(lingodb::runtime::ExecutionContext* executionContext) override {
      for (size_t i = 0;; i++) {
         auto resultTable = executionContext->getResultOfType<lingodb::runtime::ArrowTable>(i);
         if (!resultTable) return;
         auto table = resultTable.value()->get();
         printTable(table);
      }
   }
};
} // namespace

std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createTableRetriever(std::shared_ptr<arrow::Table>& result) {
   return std::make_unique<TableRetriever>(result);
}

std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createTablePrinter() {
   return std::make_unique<TablePrinter>();
}

std::unique_ptr<lingodb::execution::ResultProcessor> lingodb::execution::createBatchedTablePrinter() {
   return std::make_unique<BatchedTablePrinter>();
}