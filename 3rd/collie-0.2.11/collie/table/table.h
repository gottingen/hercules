// Copyright 2024 The Elastic AI Search Authors.
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

#include <collie/table/table_internal.h>
#include <string_view>
#include <variant>
#include <utility>

namespace collie::table {

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

        Format &format() { return table_->format(); }

        void print(std::ostream &stream) { table_->print(stream); }

        std::string str() {
            std::stringstream stream;
            print(stream);
            return stream.str();
        }

        size_t size() const { return table_->size(); }

        std::pair<size_t, size_t> shape() { return table_->shape(); }

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

    class RowStream {
    public:
        operator const Table::Row_t &() const { return row_; }

        template<typename T, typename = typename std::enable_if<
                !std::is_convertible<T, Table::Row_t::value_type>::value>::type>
        RowStream &operator<<(const T &obj) {
            oss_ << obj;
            std::string cell{oss_.str()};
            oss_.str("");
            if (!cell.empty()) {
                row_.push_back(cell);
            }
            return *this;
        }

        RowStream &operator<<(const Table::Row_t::value_type &cell) {
            row_.push_back(cell);
            return *this;
        }

        RowStream &copyfmt(const RowStream &other) {
            oss_.copyfmt(other.oss_);
            return *this;
        }

        RowStream &copyfmt(const std::ios &other) {
            oss_.copyfmt(other);
            return *this;
        }

        std::ostringstream::char_type fill() const { return oss_.fill(); }

        std::ostringstream::char_type fill(std::ostringstream::char_type ch) { return oss_.fill(ch); }

        std::ios_base::iostate exceptions() const { return oss_.exceptions(); }

        void exceptions(std::ios_base::iostate except) { oss_.exceptions(except); }

        std::locale imbue(const std::locale &loc) { return oss_.imbue(loc); }

        std::locale getloc() const { return oss_.getloc(); }

        char narrow(std::ostringstream::char_type c, char dfault) const { return oss_.narrow(c, dfault); }

        std::ostringstream::char_type widen(char c) const { return oss_.widen(c); }

        std::ios::fmtflags flags() const { return oss_.flags(); }

        std::ios::fmtflags flags(std::ios::fmtflags flags) { return oss_.flags(flags); }

        std::ios::fmtflags setf(std::ios::fmtflags flags) { return oss_.setf(flags); }

        std::ios::fmtflags setf(std::ios::fmtflags flags, std::ios::fmtflags mask) {
            return oss_.setf(flags, mask);
        }

        void unsetf(std::ios::fmtflags flags) { oss_.unsetf(flags); }

        std::streamsize precision() const { return oss_.precision(); }

        std::streamsize precision(std::streamsize new_precision) { return oss_.precision(new_precision); }

        std::streamsize width() const { return oss_.width(); }

        std::streamsize width(std::streamsize new_width) { return oss_.width(new_width); }

    private:
        Table::Row_t row_;
        std::ostringstream oss_;
    };

} // namespace collie::table
