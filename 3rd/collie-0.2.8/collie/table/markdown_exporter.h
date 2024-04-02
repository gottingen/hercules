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

#include <collie/table/exporter.h>

namespace collie::table {

    class MarkdownExporter : public Exporter {
    public:
        std::string dump(Table &table) override {
            std::string result{""};
            apply_markdown_format(table);
            result = table.str();
            restore_table_format(table);
            return result;
        }

        virtual ~MarkdownExporter() {}

    private:
        void add_alignment_header_row(Table &table) {
            auto &rows = table.table_->rows_;

            if (rows.size() >= 1) {
                auto alignment_row = std::make_shared<Row>(table.table_->shared_from_this());

                // Create alignment header cells
                std::vector<std::string> alignment_cells{};
                for (auto &cell: table[0]) {
                    auto format = cell.format();
                    if (format.font_align_.value() == FontAlign::left) {
                        alignment_cells.push_back(":----");
                    } else if (format.font_align_.value() == FontAlign::center) {
                        alignment_cells.push_back(":---:");
                    } else if (format.font_align_.value() == FontAlign::right) {
                        alignment_cells.push_back("----:");
                    }
                }

                // Add alignment header cells to alignment row
                for (auto &c: alignment_cells) {
                    auto cell = std::make_shared<Cell>(alignment_row);
                    cell->format()
                            .hide_border_top()
                            .hide_border_bottom()
                            .border_left("|")
                            .border_right("|")
                            .column_separator("|")
                            .corner("|");
                    cell->set_text(c);
                    if (c == ":---:")
                        cell->format().font_align(FontAlign::center);
                    else if (c == "----:")
                        cell->format().font_align(FontAlign::right);
                    alignment_row->add_cell(cell);
                }

                // Insert alignment header row
                if (rows.size() > 1)
                    rows.insert(rows.begin() + 1, alignment_row);
                else
                    rows.push_back(alignment_row);
            }
        }

        void remove_alignment_header_row(Table &table) {
            auto &rows = table.table_->rows_;
            table.table_->rows_.erase(rows.begin() + 1);
        }

        void apply_markdown_format(Table &table) {
            // Apply markdown format to cells in each row
            for (auto row: table) {
                for (auto &cell: row) {
                    auto format = cell.format();
                    formats_.push_back(format);
                    cell.format()
                            .hide_border_top()
                            .hide_border_bottom()
                            .border_left("|")
                            .border_right("|")
                            .column_separator("|")
                            .corner("|");
                }
            }
            // Add alignment header row at position 1
            add_alignment_header_row(table);
        }

        void restore_table_format(Table &table) {
            // Remove alignment header row at position 1
            remove_alignment_header_row(table);

            // Restore original formatting for each cell
            size_t format_index{0};
            for (auto row: table) {
                for (auto &cell: row) {
                    cell.format() = formats_[format_index];
                    format_index += 1;
                }
            }
        }

        std::vector<Format> formats_;
    };

} // namespace collie::table
