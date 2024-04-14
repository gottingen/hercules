// Copyright 2023 The Turbo Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include "turbo/format/table/table_internal.h"
#include "turbo/format/table/printer.h"
#include <string_view>
#include <variant>
#include <utility>

namespace turbo {

    /**
     * @ingroup turbo_fmt_format_table
     * @brief The Table class is the main class of the table module.
     *        It provides a simple way to format data into a table.
     *
     *        Example:
     *        @code{.cpp}
     *        Table process_table;
     *        process_table.add_row({"turbo", "1234", "0.1"});
     *        process_table.add_row({"turbo", "1235", "0.2"});
     *        std::cout << process_table << std::endl;
     *        @endcode
     *        Output:
     *        @code{.unparsed}
     *        +--------+------+-----+
     *        |turbo   |1234  |0.1  |
     *        +--------+------+-----+
     *        |turbo   |1235  |0.2  |
     *        +--------+------+-----+
     *        @endcode
     *        The table module is part of the turbo::format module.
     */

    class Table {
    public:
        Table() : table_(TableInternal::create()) {}

        using Row_t = std::vector<std::variant<std::string, const char *, std::string_view, Table>>;

        Table &add_row(const Row_t &cells) {

            if (rows_ == 0) {
                // This is the first row added
                // cells.size() is the number of columns
                cols_ = cells.size();
            }

            std::vector<std::string> cell_strings;
            if (cells.size() < cols_) {
                cell_strings.resize(cols_);
                std::fill(cell_strings.begin(), cell_strings.end(), "");
            } else {
                cell_strings.resize(cells.size());
                std::fill(cell_strings.begin(), cell_strings.end(), "");
            }

            for (size_t i = 0; i < cells.size(); ++i) {
                auto cell = cells[i];
                if (std::holds_alternative<std::string>(cell)) {
                    cell_strings[i] = *std::get_if<std::string>(&cell);
                } else if (std::holds_alternative<const char *>(cell)) {
                    cell_strings[i] = *std::get_if<const char *>(&cell);
                } else if (std::holds_alternative<std::string_view>(cell)) {
                    cell_strings[i] = std::string{*std::get_if<std::string_view>(&cell)};
                } else {
                    auto table = *std::get_if<Table>(&cell);
                    std::stringstream stream;
                    table.print(stream);
                    cell_strings[i] = stream.str();
                }
            }

            table_->add_row(cell_strings);
            rows_ += 1;
            return *this;
        }

        Row &operator[](size_t index) { return row(index); }

        Row &row(size_t index) { return (*table_)[index]; }

        Column column(size_t index) { return table_->column(index); }

        EntityFormat &format() { return table_->format(); }

        void print(std::ostream &stream) { Printer::print_table(stream, *table_);}

        std::string str() {
            std::stringstream stream;
            print(stream);
            return stream.str();
        }

        size_t size() const { return table_->size(); }

        std::pair<size_t, size_t> shape() {
            std::pair<size_t, size_t> result{0, 0};
            std::stringstream stream;
            print(stream);
            auto buffer = stream.str();
            auto lines = EntityFormat::split_lines(buffer, "\n", "", true);
            if (lines.size()) {
                result = {get_sequence_length(lines[0], "", true), lines.size()};
            }
            return result;
        }

        class RowIterator {
        public:
            explicit RowIterator(std::vector<std::shared_ptr<Row>>::iterator ptr) : ptr(ptr) {}

            RowIterator operator++() {
                ++ptr;
                return *this;
            }

            bool operator!=(const RowIterator &other) const { return ptr != other.ptr; }

            Row &operator*() { return **ptr; }

        private:
            std::vector<std::shared_ptr<Row>>::iterator ptr;
        };

        auto begin() -> RowIterator { return RowIterator(table_->rows_.begin()); }

        auto end() -> RowIterator { return RowIterator(table_->rows_.end()); }

    private:
        friend class MarkdownExporter;

        friend class LatexExporter;

        friend class AsciiDocExporter;

        friend std::ostream &operator<<(std::ostream &stream, const Table &table);

        size_t rows_{0};
        size_t cols_{0};
        std::shared_ptr<TableInternal> table_;
    };

    inline std::ostream &operator<<(std::ostream &stream, const Table &table) {
        const_cast<Table &>(table).print(stream);
        return stream;
    }

    inline size_t table_print_line(const Table &table) {
        return 2 * table.size() + 1;
    }

} // namespace turbo
