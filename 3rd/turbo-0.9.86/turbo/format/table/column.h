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
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include "turbo/format/table/cell.h"
#include "turbo/format/table/column_format.h"
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace turbo {

    class Column {
    public:
        explicit Column(std::shared_ptr<class TableInternal> parent) : parent_(parent) {}

        void add_cell(Cell &cell) { cells_.push_back(cell); }

        Cell &operator[](size_t index) { return cells_[index]; }

        std::vector<std::reference_wrapper<Cell>> cells() const { return cells_; }

        size_t size() const { return cells_.size(); }

        ColumnFormat format() { return ColumnFormat(*this); }

        class CellIterator {
        public:
            explicit CellIterator(std::vector<std::reference_wrapper<Cell>>::iterator ptr) : ptr(ptr) {}

            CellIterator operator++() {
                ++ptr;
                return *this;
            }

            bool operator!=(const CellIterator &other) const { return ptr != other.ptr; }

            Cell &operator*() { return *ptr; }

        private:
            std::vector<std::reference_wrapper<Cell>>::iterator ptr;
        };

        auto begin() -> CellIterator { return CellIterator(cells_.begin()); }

        auto end() -> CellIterator { return CellIterator(cells_.end()); }

    private:
        friend class ColumnFormat;

        friend class Printer;

        // Returns the column width as configured
        // For each cell in the column, check the cell.format.width
        // property and return the largest configured column width
        // This is used to ensure that all cells in a column are
        // aligned when printing the column
        size_t get_configured_width() {
            size_t result{0};
            for (size_t i = 0; i < size(); ++i) {
                auto cell = cells_[i];
                auto format = cell.get().format();
                if (format.width_.has_value())
                    result = std::max(result, *format.width_);
            }
            return result;
        }

        // Computes the width of the column based on cell contents
        // and configured cell padding
        // For each cell, compute padding_left + cell_contents + padding_right
        // and return the largest value
        //
        // This is useful when no cell.format.width is configured
        // Call get_configured_width()
        // - If this returns 0, then use get_computed_width()
        size_t get_computed_width() {
            size_t result{0};
            for (size_t i = 0; i < size(); ++i) {
                result = std::max(result, get_cell_width(i));
            }
            return result;
        }

        // Returns padding_left + cell_contents.size() + padding_right
        // for a given cell in the column
        size_t get_cell_width(size_t cell_index) {
            size_t result{0};
            Cell &cell = cells_[cell_index].get();
            auto format = cell.format();
            if (format.padding_left_.has_value())
                result += *format.padding_left_;

            // Check if input text has newlines
            auto text = cell.get_text();
            auto split_lines = EntityFormat::split_lines(text, "\n", cell.locale(),
                                                   cell.is_multi_byte_character_support_enabled());

            // If there are no newlines in input, set column_width = text.size()
            if (split_lines.size() == 1) {
                result += cell.size();
            } else {
                // There are newlines in input
                // Find widest substring in input and use this as column_width
                size_t widest_sub_string_size{0};
                for (auto &line: split_lines)
                    if (get_sequence_length(line, cell.locale(),
                                            cell.is_multi_byte_character_support_enabled()) >
                        widest_sub_string_size)
                        widest_sub_string_size = get_sequence_length(
                                line, cell.locale(), cell.is_multi_byte_character_support_enabled());
                result += widest_sub_string_size;
            }

            if (format.padding_right_.has_value())
                result += *format.padding_right_;

            return result;
        }

        std::vector<std::reference_wrapper<Cell>> cells_;
        std::weak_ptr<class TableInternal> parent_;
    };

    inline ColumnFormat &ColumnFormat::width(size_t value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().width(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::height(size_t value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().height(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::padding(size_t value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().padding(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::padding_left(size_t value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().padding_left(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::padding_right(size_t value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().padding_right(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::padding_top(size_t value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().padding_top(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::padding_bottom(size_t value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().padding_bottom(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_left(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_left(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_left_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_left_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_left_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_left_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_right(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_right(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_right_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_right_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_right_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_right_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_top(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_top(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_top_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_top_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_top_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_top_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_bottom(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_bottom(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_bottom_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_bottom_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::border_bottom_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().border_bottom_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::corner(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().corner(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::corner_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().corner_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::corner_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().corner_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::column_separator(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().column_separator(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::column_separator_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().column_separator_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::column_separator_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().column_separator_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::font_align(FontAlign value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().font_align(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::font_style(const turbo::emphasis &style) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().font_style(style);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::font_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().font_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::font_color(turbo::terminal_color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().font_color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::font_background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().font_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::font_background_color(turbo::terminal_color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().font_color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().color(fg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::background_color(turbo::color value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().color(bg(value));
        return *this;
    }

    inline ColumnFormat &ColumnFormat::multi_byte_characters(bool value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().multi_byte_characters(value);
        return *this;
    }

    inline ColumnFormat &ColumnFormat::locale(const std::string &value) {
        for (auto &cell: column_.get().cells_)
            cell.get().format().locale(value);
        return *this;
    }

} // namespace turbo
