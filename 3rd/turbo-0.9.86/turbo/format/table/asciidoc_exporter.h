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
#include <optional>
#include <sstream>
#include <string>
#include "turbo/format/table/exporter.h"

namespace turbo {

    class AsciiDocExporter : public Exporter {

        static const char new_line = '\n';

    public:
        std::string dump(Table &table) override {
            std::stringstream ss;
            ss << add_alignment_header(table);
            ss << new_line;

            const auto rows = table.rows_;
            // iterate content and put text into the table.
            for (size_t row_index = 0; row_index < rows; row_index++) {
                auto &row = table[row_index];

                for (size_t cell_index = 0; cell_index < row.size(); cell_index++) {
                    ss << "|";
                    ss << add_formatted_cell(row[cell_index]);
                }
                ss << new_line;
                if (row_index == 0) {
                    ss << new_line;
                }
            }

            ss << "|===";
            return ss.str();
        }

        virtual ~AsciiDocExporter() {}

    private:
        std::string add_formatted_cell(Cell &cell) const {
            std::stringstream ss;
            auto format = cell.format();
            std::string cell_string = cell.get_text();

            auto font_style = format.font_style_;

            bool format_bold = false;
            bool format_italic = false;
            if(font_style.get().has_emphasis()) {
                auto emphasis = font_style.get().get_emphasis();
                    if(emphasis & turbo::emphasis::bold) {
                        format_bold = true;
                    } else if(emphasis & turbo::emphasis::italic) {
                        format_italic = true;
                    }
            }

            if (format_bold) {
                ss << '*';
            }
            if (format_italic) {
                ss << '_';
            }

            ss << cell_string;
            if (format_italic) {
                ss << '_';
            }
            if (format_bold) {
                ss << '*';
            }
            return ss.str();
        }

        std::string add_alignment_header(Table &table) {
            std::stringstream ss;
            ss << (R"([cols=")");

            size_t column_count = table[0].size();
            size_t column_index = 0;
            for (auto &cell: table[0]) {
                auto format = cell.format();

                if (format.font_align_.value() == FontAlign::left) {
                    ss << '<';
                } else if (format.font_align_.value() == FontAlign::center) {
                    ss << '^';
                } else if (format.font_align_.value() == FontAlign::right) {
                    ss << '>';
                }

                ++column_index;
                if (column_index != column_count) {
                    ss << ",";
                }
            }

            ss << R"("])";
            ss << new_line;
            ss << "|===";

            return ss.str();
        }
    };

} // namespace turbo
