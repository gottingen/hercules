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

namespace collie::table {

    class ColumnFormat : public Format {
    public:
        explicit ColumnFormat(class Column &column) : column_(column) {}

        ColumnFormat &width(size_t value);

        ColumnFormat &height(size_t value);

        // Padding
        ColumnFormat &padding(size_t value);

        ColumnFormat &padding_left(size_t value);

        ColumnFormat &padding_right(size_t value);

        ColumnFormat &padding_top(size_t value);

        ColumnFormat &padding_bottom(size_t value);

        // Border
        ColumnFormat &border(const std::string &value);

        ColumnFormat &border_color(Color value);

        ColumnFormat &border_background_color(Color value);

        ColumnFormat &border_left(const std::string &value);

        ColumnFormat &border_left_color(Color value);

        ColumnFormat &border_left_background_color(Color value);

        ColumnFormat &border_right(const std::string &value);

        ColumnFormat &border_right_color(Color value);

        ColumnFormat &border_right_background_color(Color value);

        ColumnFormat &border_top(const std::string &value);

        ColumnFormat &border_top_color(Color value);

        ColumnFormat &border_top_background_color(Color value);

        ColumnFormat &border_bottom(const std::string &value);

        ColumnFormat &border_bottom_color(Color value);

        ColumnFormat &border_bottom_background_color(Color value);

        // Corner
        ColumnFormat &corner(const std::string &value);

        ColumnFormat &corner_color(Color value);

        ColumnFormat &corner_background_color(Color value);

        // Column separator
        ColumnFormat &column_separator(const std::string &value);

        ColumnFormat &column_separator_color(Color value);

        ColumnFormat &column_separator_background_color(Color value);

        // Font styling
        ColumnFormat &font_align(FontAlign value);

        ColumnFormat &font_style(const std::vector<FontStyle> &style);

        ColumnFormat &font_color(Color value);

        ColumnFormat &font_background_color(Color value);

        ColumnFormat &color(Color value);

        ColumnFormat &background_color(Color value);

        // Locale
        ColumnFormat &multi_byte_characters(bool value);

        ColumnFormat &locale(const std::string &value);

    private:
        std::reference_wrapper<class Column> column_;
    };

} // namespace collie::table
