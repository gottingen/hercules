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

#include <iostream>
#include <memory>
#include <string>
#include <collie/table/cell.h>
#include <optional>
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace collie::table {

    class Row {
    public:
        explicit Row(std::shared_ptr<class TableInternal> parent) : parent_(parent) {}

        void add_cell(std::shared_ptr<Cell> cell) { cells_.push_back(cell); }

        Cell &operator[](size_t index) { return cell(index); }

        Cell &cell(size_t index) { return *(cells_[index]); }

        std::vector<std::shared_ptr<Cell>> cells() const { return cells_; }

        size_t size() const { return cells_.size(); }

        Format &format();

        class CellIterator {
        public:
            explicit CellIterator(std::vector<std::shared_ptr<Cell>>::iterator ptr) : ptr(ptr) {}

            CellIterator operator++() {
                ++ptr;
                return *this;
            }

            bool operator!=(const CellIterator &other) const { return ptr != other.ptr; }

            Cell &operator*() { return **ptr; }

        private:
            std::vector<std::shared_ptr<Cell>>::iterator ptr;
        };

        auto begin() -> CellIterator { return CellIterator(cells_.begin()); }

        auto end() -> CellIterator { return CellIterator(cells_.end()); }

    private:
        friend class Printer;

        // Returns the row height as configured
        // For each cell in the row, check the cell.format.height
        // property and return the largest configured row height
        // This is used to ensure that all cells in a row are
        // aligned when printing the column
        size_t get_configured_height() {
            size_t result{0};
            for (size_t i = 0; i < size(); ++i) {
                auto cell = cells_[i];
                auto format = cell->format();
                if (format.height_.has_value())
                    result = std::max(result, *format.height_);
            }
            return result;
        }

        // Computes the height of the row based on cell contents
        // and configured cell padding
        // For each cell, compute:
        //   padding_top + (cell_contents / column height) + padding_bottom
        // and return the largest value
        //
        // This is useful when no cell.format.height is configured
        // Call get_configured_height()
        // - If this returns 0, then use get_computed_height()
        size_t get_computed_height(const std::vector<size_t> &column_widths) {
            size_t result{0};
            for (size_t i = 0; i < size(); ++i) {
                result = std::max(result, get_cell_height(i, column_widths[i]));
            }
            return result;
        }

        // Returns padding_top + cell_contents / column_height + padding_bottom
        // for a given cell in the column
        // e.g.,
        // column width = 5
        // cell_contents = "I love tabulate" (size/length = 15)
        // padding top and padding bottom are 1
        // then, cell height = 1 + (15 / 5) + 1 = 1 + 3 + 1 = 5
        // The cell will look like this:
        //
        // .....
        // I lov
        // e tab
        // ulate
        // .....
        size_t get_cell_height(size_t cell_index, size_t column_width) {
            size_t result{0};
            Cell &cell = *(cells_[cell_index]);
            auto format = cell.format();
            auto text = cell.get_text();

            auto padding_left = *format.padding_left_;
            auto padding_right = *format.padding_right_;

            result += *format.padding_top_;

            if (column_width > (padding_left + padding_right)) {
                column_width -= (padding_left + padding_right);
            }

            // Check if input text has embedded newline characters
            auto newlines_in_text = std::count(text.begin(), text.end(), '\n');
            std::string word_wrapped_text;
            if (newlines_in_text == 0) {
                // No new lines in input
                // Apply automatic word wrapping and compute row height
                word_wrapped_text = Format::word_wrap(text, column_width, cell.locale(),
                                                      cell.is_multi_byte_character_support_enabled());
            } else {
                // There are embedded '\n' characters
                // Respect these characters
                word_wrapped_text = text;
            }

            auto newlines_in_wrapped_text =
                    std::count(word_wrapped_text.begin(), word_wrapped_text.end(), '\n');
            auto estimated_row_height = newlines_in_wrapped_text;

            if (!word_wrapped_text.empty() &&
                word_wrapped_text[word_wrapped_text.size() - 1] != '\n') // text doesn't end with a newline
                estimated_row_height += 1;

            result += estimated_row_height;

            result += *format.padding_bottom_;

            return result;
        }

        std::vector<std::shared_ptr<Cell>> cells_;
        std::weak_ptr<class TableInternal> parent_;
        std::optional<Format> format_;
    };

} // namespace collie::table
