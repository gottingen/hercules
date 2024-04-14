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

namespace turbo {

    class ColumnFormat : public EntityFormat {
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

        ColumnFormat &border_color(turbo::color value);

        ColumnFormat &border_background_color(turbo::color value);

        ColumnFormat &border_left(const std::string &value);

        ColumnFormat &border_left_color(turbo::color value);

        ColumnFormat &border_left_background_color(turbo::color value);

        ColumnFormat &border_right(const std::string &value);

        ColumnFormat &border_right_color(turbo::color value);

        ColumnFormat &border_right_background_color(turbo::color value);

        ColumnFormat &border_top(const std::string &value);

        ColumnFormat &border_top_color(turbo::color value);

        ColumnFormat &border_top_background_color(turbo::color value);

        ColumnFormat &border_bottom(const std::string &value);

        ColumnFormat &border_bottom_color(turbo::color value);

        ColumnFormat &border_bottom_background_color(turbo::color value);

        // Corner
        ColumnFormat &corner(const std::string &value);

        ColumnFormat &corner_color(turbo::color value);

        ColumnFormat &corner_background_color(turbo::color value);

        // Column separator
        ColumnFormat &column_separator(const std::string &value);

        ColumnFormat &column_separator_color(turbo::color value);

        ColumnFormat &column_separator_background_color(turbo::color value);

        // Font styling
        ColumnFormat &font_align(FontAlign value);

        ColumnFormat &font_style(const turbo::emphasis &style);

        ColumnFormat &font_color(turbo::color value);

        ColumnFormat &font_color(turbo::terminal_color value);

        ColumnFormat &font_background_color(turbo::color value);

        ColumnFormat &font_background_color(turbo::terminal_color value);

        ColumnFormat &color(turbo::color value);

        ColumnFormat &background_color(turbo::color value);

        // Locale
        ColumnFormat &multi_byte_characters(bool value);

        ColumnFormat &locale(const std::string &value);

    private:
        std::reference_wrapper<class Column> column_;
    };

} // namespace turbo
