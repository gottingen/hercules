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

#include "turbo/format/table/exporter.h"
#include <optional>

namespace turbo {

    class LatexExporter : public Exporter {

        static const char new_line = '\n';

    public:
        class ExportOptions {
        public:
            ExportOptions &indentation(std::size_t value) {
                indentation_ = value;
                return *this;
            }

        private:
            friend class LatexExporter;

            std::optional<size_t> indentation_;
        };

        ExportOptions &configure() { return options_; }

        std::string dump(Table &table) override {
            std::string result{"\\begin{tabular}"};
            result += new_line;

            result += add_alignment_header(table);
            result += new_line;
            const auto rows = table.rows_;
            // iterate content and put text into the table.
            for (size_t i = 0; i < rows; i++) {
                auto &row = table[i];
                // apply row content indentation
                if (options_.indentation_.has_value()) {
                    result += std::string(options_.indentation_.value(), ' ');
                }

                for (size_t j = 0; j < row.size(); j++) {

                    result += row[j].get_text();

                    // check column position, need "\\" at the end of each row
                    if (j < row.size() - 1) {
                        result += " & ";
                    } else {
                        result += " \\\\";
                    }
                }
                result += new_line;
            }

            result += "\\end{tabular}";
            return result;
        }

        virtual ~LatexExporter() {}

    private:
        std::string add_alignment_header(Table &table) {
            std::string result{"{"};

            for (auto &cell: table[0]) {
                auto format = cell.format();
                if (format.font_align_.value() == FontAlign::left) {
                    result += 'l';
                } else if (format.font_align_.value() == FontAlign::center) {
                    result += 'c';
                } else if (format.font_align_.value() == FontAlign::right) {
                    result += 'r';
                }
            }

            result += "}";
            return result;
        }

        ExportOptions options_;
    };

} // namespace turbo
