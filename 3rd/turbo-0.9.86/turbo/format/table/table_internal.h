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

#include <algorithm>
#include <iostream>
#include <string>
#include "turbo/format/table/column.h"
#include "turbo/format/table/row.h"
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace turbo {

    class TableInternal : public std::enable_shared_from_this<TableInternal> {
    public:
        static std::shared_ptr<TableInternal> create() {
            auto result = std::shared_ptr<TableInternal>(new TableInternal());
            result->format_.set_defaults();
            return result;
        }

        void add_row(const std::vector<std::string> &cells) {
            auto row = std::make_shared<Row>(shared_from_this());
            for (auto &c: cells) {
                auto cell = std::make_shared<Cell>(row);
                cell->set_text(c);
                row->add_cell(cell);
            }
            rows_.push_back(row);
        }

        Row &operator[](size_t index) { return *(rows_[index]); }

        const Row &operator[](size_t index) const { return *(rows_[index]); }

        Column column(size_t index) {
            Column column(shared_from_this());
            for (size_t i = 0; i < rows_.size(); ++i) {
                auto row = rows_[i];
                auto &cell = row->cell(index);
                column.add_cell(cell);
            }
            return column;
        }

        size_t size() const { return rows_.size(); }

        EntityFormat &format() { return format_; }

        //void print(std::ostream &stream) { Printer::print_table(stream, *this); }

        size_t estimate_num_columns() const {
            size_t result{0};
            if (size()) {
                auto first_row = operator[](size_t(0));
                result = first_row.size();
            }
            return result;
        }

    private:
        friend class Table;

        friend class MarkdownExporter;

        TableInternal() {}

        TableInternal &operator=(const TableInternal &);

        TableInternal(const TableInternal &);

        std::vector<std::shared_ptr<Row>> rows_;
        EntityFormat format_;
    };

    inline EntityFormat &Cell::format() {
        std::shared_ptr<Row> parent = parent_.lock();
        if (!format_.has_value()) {   // no cell format
            format_ = parent->format(); // Use parent row format
        } else {
            // Cell has formatting
            // Merge cell formatting with parent row formatting
            format_ = EntityFormat::merge(*format_, parent->format());
        }
        return *format_;
    }

    inline bool Cell::is_multi_byte_character_support_enabled() {
        return (*format().multi_byte_characters_);
    }

    inline EntityFormat &Row::format() {
        std::shared_ptr<TableInternal> parent = parent_.lock();
        if (!format_.has_value()) {   // no row format
            format_ = parent->format(); // Use parent table format
        } else {
            // Row has formatting rules
            // Merge with parent table format
            format_ = EntityFormat::merge(*format_, parent->format());
        }
        return *format_;
    }


} // namespace turbo
