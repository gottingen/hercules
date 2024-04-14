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

#include "turbo/format/table.h"

using namespace turbo;
using Row_t = Table::Row_t;

int main() {
    Table colors;

    colors.add_row(Row_t{"Font Color is Red", "Font Color is Blue", "Font Color is Green"});
    colors.add_row(Row_t{"Everything is Red", "Everything is Blue", "Everything is Green"});
    colors.add_row(
            Row_t{"Font Background is Red", "Font Background is Blue", "Font Background is Green"});

    colors[0][0].format().font_color(fg(color::red)).font_color({emphasis::bold});
    colors[0][1].format().font_color(fg(color::blue)).font_color({emphasis::bold});
    colors[0][2].format().font_color(fg(color::green)).font_color({emphasis::bold});

    colors[1][0]
            .format()
            .border_left_color(fg(color::red))
            .border_left_color(bg(color::red))
            .font_color(bg(color::red))
            .font_color(fg(color::red));

    colors[1][1]
            .format()
            .border_left_color(fg(color::blue))
            .border_left_color(bg(color::blue))
            .font_color(bg(color::blue))
            .font_color(fg(color::blue));

    colors[1][2]
            .format()
            .border_left_color(fg(color::green))
            .border_left_color(bg(color::green))
            .font_color(bg(color::green))
            .font_color(fg(color::green))
            .border_right_color(fg(color::green))
            .border_right_color(bg(color::green));

    colors[2][0].format().font_color(bg(color::red)).font_color({emphasis::bold});
    colors[2][1].format().font_color(bg(color::blue)).font_color({emphasis::bold});
    colors[2][2].format().font_color(bg(color::green)).font_color({emphasis::bold});

    std::cout << colors << std::endl;
}