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

#include <utility>
#include <vector>
#include "turbo/format/table/table_internal.h"
#include "turbo/format/print.h"
#include "turbo/format/terminal.h"

namespace turbo {

    class Printer {
    public:
        static std::pair<std::vector<size_t>, std::vector<size_t>>
        compute_cell_dimensions(TableInternal &table);

        static void print_table(std::ostream &stream, TableInternal &table);

        static void print_row_in_cell(std::ostream &stream, TableInternal &table,
                                      const std::pair<size_t, size_t> &index,
                                      const std::pair<size_t, size_t> &dimension, size_t num_columns,
                                      size_t row_index,
                                      const std::vector<std::string> &splitted_cell_text);

        static bool print_cell_border_top(std::ostream &stream, TableInternal &table,
                                          const std::pair<size_t, size_t> &index,
                                          const std::pair<size_t, size_t> &dimension, size_t num_columns);

        static bool print_cell_border_bottom(std::ostream &stream, TableInternal &table,
                                             const std::pair<size_t, size_t> &index,
                                             const std::pair<size_t, size_t> &dimension,
                                             size_t num_columns);


    private:
        static void print_content_left_aligned(std::ostream &stream, const std::string &cell_content,
                                               const EntityFormat &format, size_t text_with_padding_size,
                                               size_t column_width) {
            // Apply font style
            apply_style(stream, format.font_style_ | format.font_color_);
            stream << cell_content;
            // Only apply font_style to the font
            // Not the padding. So calling apply_element_style with font_style = {}
            reset_style(stream);
            apply_style(stream, format.font_color_);

            if (text_with_padding_size < column_width) {
                for (size_t j = 0; j < (column_width - text_with_padding_size); ++j) {
                    stream << " ";
                }
            }
        }

        static void print_content_center_aligned(std::ostream &stream, const std::string &cell_content,
                                                 const EntityFormat &format, size_t text_with_padding_size,
                                                 size_t column_width) {
            auto num_spaces = column_width - text_with_padding_size;
            if (num_spaces % 2 == 0) {
                // Even spacing on either side
                for (size_t j = 0; j < num_spaces / 2; ++j)
                    stream << " ";

                // Apply font style
                apply_style(stream, format.font_style_ | format.font_color_);
                stream << cell_content;
                // Only apply font_style to the font
                // Not the padding. So calling apply_element_style with font_style = {}
                reset_style(stream);
                apply_style(stream, format.font_color_);

                for (size_t j = 0; j < num_spaces / 2; ++j)
                    stream << " ";
            } else {
                auto num_spaces_before = num_spaces / 2 + 1;
                for (size_t j = 0; j < num_spaces_before; ++j)
                    stream << " ";

                // Apply font style
                apply_style(stream, format.font_style_ | format.font_color_);
                stream << cell_content;
                // Only apply font_style to the font
                // Not the padding. So calling apply_element_style with font_style = {}
                reset_style(stream);
                apply_style(stream, format.font_color_);

                for (size_t j = 0; j < num_spaces - num_spaces_before; ++j)
                    stream << " ";
            }
        }

        static void print_content_right_aligned(std::ostream &stream, const std::string &cell_content,
                                                const EntityFormat &format, size_t text_with_padding_size,
                                                size_t column_width) {
            if (text_with_padding_size < column_width) {
                for (size_t j = 0; j < (column_width - text_with_padding_size); ++j) {
                    stream << " ";
                }
            }

            // Apply font style
            apply_style(stream, format.font_style_ | format.font_color_);
            stream << cell_content;
            // Only apply font_style to the font
            // Not the padding. So calling apply_element_style with font_style = {}
            reset_style(stream);
            apply_style(stream, format.font_color_);
        }

    };

    inline std::pair<std::vector<size_t>, std::vector<size_t>>
    Printer::compute_cell_dimensions(TableInternal &table) {
        std::pair<std::vector<size_t>, std::vector<size_t>> result;
        size_t num_rows = table.size();
        size_t num_columns = table.estimate_num_columns();

        std::vector<size_t> row_heights, column_widths{};

        for (size_t i = 0; i < num_columns; ++i) {
            Column column = table.column(i);
            size_t configured_width = column.get_configured_width();
            size_t computed_width = column.get_computed_width();
            if (configured_width != 0)
                column_widths.push_back(configured_width);
            else
                column_widths.push_back(computed_width);
        }

        for (size_t i = 0; i < num_rows; ++i) {
            Row row = table[i];
            size_t configured_height = row.get_configured_height();
            size_t computed_height = row.get_computed_height(column_widths);

            // NOTE: Unlike column width, row height is calculated as the max
            // b/w configured height and computed height
            // which means that .width() has higher precedence than .height()
            // when both are configured by the user
            //
            // TODO: Maybe this can be configured?
            // If such a configuration is exposed, i.e., prefer height over width
            // then the logic will be reversed, i.e.,
            // column_widths.push_back(std::max(configured_width, computed_width))
            // and
            // row_height = configured_height if != 0 else computed_height

            row_heights.push_back(std::max(configured_height, computed_height));
        }

        result.first = row_heights;
        result.second = column_widths;

        return result;
    }

    inline void Printer::print_table(std::ostream &stream, TableInternal &table) {
        size_t num_rows = table.size();
        size_t num_columns = table.estimate_num_columns();
        auto dimensions = compute_cell_dimensions(table);
        auto row_heights = dimensions.first;
        auto column_widths = dimensions.second;
        auto splitted_cells_text = std::vector<std::vector<std::vector<std::string>>>(
                num_rows, std::vector<std::vector<std::string>>(num_columns, std::vector<std::string>{}));

        // Pre-compute the cells' content and split them into lines before actually
        // iterating the cells.
        for (size_t i = 0; i < num_rows; ++i) {
            Row row = table[i];
            for (size_t j = 0; j < num_columns; ++j) {
                Cell cell = row.cell(j);
                const std::string &text = cell.get_text();
                auto padding_left = *cell.format().padding_left_;
                auto padding_right = *cell.format().padding_right_;

                // Check if input text has embedded \n that are to be respected
                bool has_new_line = text.find_first_of('\n') != std::string::npos;

                if (has_new_line) {
                    // Respect to the embedded '\n' characters
                    splitted_cells_text[i][j] = EntityFormat::split_lines(
                            text, "\n", cell.locale(), cell.is_multi_byte_character_support_enabled());
                } else {
                    // If there are no embedded \n characters, then apply word wrap.
                    //
                    // Configured column width cannot be lower than (padding_left +
                    // padding_right) This is a bad configuration E.g., the user is trying
                    // to force the column width to be 5 when padding_left and padding_right
                    // are each configured to 3 (padding_left + padding_right) = 6 >
                    // column_width
                    auto content_width = column_widths[j] > padding_left + padding_right
                                         ? column_widths[j] - padding_left - padding_right
                                         : column_widths[j];
                    auto word_wrapped_text = EntityFormat::word_wrap(text, content_width, cell.locale(),
                                                                     cell.is_multi_byte_character_support_enabled());
                    splitted_cells_text[i][j] = EntityFormat::split_lines(
                            word_wrapped_text, "\n", cell.locale(), cell.is_multi_byte_character_support_enabled());
                }
            }
        }

        // For each row,
        for (size_t i = 0; i < num_rows; ++i) {

            // Print top border
            bool border_top_printed{true};
            for (size_t j = 0; j < num_columns; ++j) {
                border_top_printed &= print_cell_border_top(stream, table, {i, j},
                                                            {row_heights[i], column_widths[j]}, num_columns);
            }
            if (border_top_printed)
                stream << "\n";

            // Print row contents with word wrapping
            for (size_t k = 0; k < row_heights[i]; ++k) {
                for (size_t j = 0; j < num_columns; ++j) {
                    print_row_in_cell(stream, table, {i, j}, {row_heights[i], column_widths[j]}, num_columns, k,
                                      splitted_cells_text[i][j]);
                }
                if (k + 1 < row_heights[i])
                    stream <<"\n";
            }

            if (i + 1 == num_rows) {

                // Check if there is bottom border to print:
                auto bottom_border_needed{true};
                for (size_t j = 0; j < num_columns; ++j) {
                    auto cell = table[i][j];
                    auto format = cell.format();
                    auto corner = *format.corner_bottom_left_;
                    auto border_bottom = *format.border_bottom_;
                    if (corner == "" && border_bottom == "") {
                        bottom_border_needed = false;
                        break;
                    }
                }

                if (bottom_border_needed)
                    stream <<"\n";
                // Print bottom border for table
                for (size_t j = 0; j < num_columns; ++j) {
                    print_cell_border_bottom(stream, table, {i, j}, {row_heights[i], column_widths[j]},
                                             num_columns);
                }
            }
            if (i + 1 < num_rows)
                stream <<"\n"; // Don't add newline after last row
        }
    }

    inline void Printer::print_row_in_cell(std::ostream &stream, TableInternal &table,
                                           const std::pair<size_t, size_t> &index,
                                           const std::pair<size_t, size_t> &dimension,
                                           size_t num_columns, size_t row_index,
                                           const std::vector<std::string> &splitted_cell_text) {
        auto column_width = dimension.second;
        auto cell = table[index.first][index.second];
        auto locale = cell.locale();
        auto is_multi_byte_character_support_enabled = cell.is_multi_byte_character_support_enabled();
        auto old_locale = std::locale::global(std::locale(locale));
        auto format = cell.format();
        auto text_height = splitted_cell_text.size();
        auto padding_top = *format.padding_top_;

        if (*format.show_border_left_) {
            apply_style(stream, format.border_left_color_);
            stream << *format.border_left_;
            reset_style(stream);
        }

        apply_style(stream, format.font_color_);
        if (row_index < padding_top) {
            // Padding top
            stream << std::string(column_width, ' ');
        } else if (row_index >= padding_top && (row_index <= (padding_top + text_height))) {
            // Retrieve padding left and right
            // (column_width - padding_left - padding_right) is the amount of space
            // available for cell text - Use this to word wrap cell contents
            auto padding_left = *format.padding_left_;
            auto padding_right = *format.padding_right_;

            if (row_index - padding_top < text_height) {
                auto line = splitted_cell_text[row_index - padding_top];

                // Print left padding characters
                stream << std::string(padding_left, ' ');

                // Print word-wrapped line
                line = EntityFormat::trim(line);
                auto line_with_padding_size =
                        get_sequence_length(line, cell.locale(), is_multi_byte_character_support_enabled) +
                        padding_left + padding_right;
                switch (*format.font_align_) {
                    case FontAlign::left:
                        print_content_left_aligned(stream, line, format, line_with_padding_size, column_width);
                        break;
                    case FontAlign::center:
                        print_content_center_aligned(stream, line, format, line_with_padding_size, column_width);
                        break;
                    case FontAlign::right:
                        print_content_right_aligned(stream, line, format, line_with_padding_size, column_width);
                        break;
                }

                // Print right padding characters
                stream << std::string(padding_right, ' ');
            } else
                stream << std::string(column_width, ' ');

        } else {
            // Padding bottom
            stream << std::string(column_width, ' ');
        }

        reset_style(stream);

        if (index.second + 1 == num_columns) {
            // Print right border after last column
            if (*format.show_border_right_) {
                apply_style(stream, format.border_right_color_);
                stream << *format.border_right_;
                reset_style(stream);
            }
        }
        std::locale::global(old_locale);
    }

    inline bool Printer::print_cell_border_top(std::ostream &stream, TableInternal &table,
                                               const std::pair<size_t, size_t> &index,
                                               const std::pair<size_t, size_t> &dimension,
                                               size_t num_columns) {
        auto cell = table[index.first][index.second];
        auto locale = cell.locale();
        auto old_locale = std::locale::global(std::locale(locale));
        auto format = cell.format();
        auto column_width = dimension.second;

        auto corner = *format.corner_top_left_;
        auto corner_color = format.corner_top_left_color_;
        auto border_top = *format.border_top_;

        if ((corner == "" && border_top == "") || !*format.show_border_top_)
            return false;

        apply_style(stream, corner_color);
        stream << corner;
        reset_style(stream);

        for (size_t i = 0; i < column_width; ++i) {
            apply_style(stream, format.border_top_color_);
            stream << border_top;
            reset_style(stream);
        }

        if (index.second + 1 == num_columns) {
            // Print corner after last column
            corner = *format.corner_top_right_;
            corner_color = format.corner_top_right_color_;
            apply_style(stream, corner_color);
            stream << corner;
            reset_style(stream);
        }
        std::locale::global(old_locale);
        return true;
    }

    inline bool Printer::print_cell_border_bottom(std::ostream &stream, TableInternal &table,
                                                  const std::pair<size_t, size_t> &index,
                                                  const std::pair<size_t, size_t> &dimension,
                                                  size_t num_columns) {
        auto cell = table[index.first][index.second];
        auto locale = cell.locale();
        auto old_locale = std::locale::global(std::locale(locale));
        auto format = cell.format();
        auto column_width = dimension.second;

        auto corner = *format.corner_bottom_left_;
        auto corner_color = format.corner_bottom_left_color_;
        auto border_bottom = *format.border_bottom_;

        if ((corner == "" && border_bottom == "") || !*format.show_border_bottom_)
            return false;

        apply_style(stream, corner_color);
        stream << corner;
        reset_style(stream);

        for (size_t i = 0; i < column_width; ++i) {
            apply_style(stream, format.border_bottom_color_);
            stream << border_bottom;
            reset_style(stream);
        }

        if (index.second + 1 == num_columns) {
            // Print corner after last column
            corner = *format.corner_bottom_right_;
            apply_style(stream, format.corner_bottom_right_color_);
            stream << corner;
            reset_style(stream);
        }
        std::locale::global(old_locale);
        return true;
    }
} // namespace turbo
