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
  Table styled_table;
  styled_table.add_row(Row_t{"Bold", "Italic", "Bold & Italic", "Blinking"});
  styled_table.add_row(Row_t{"Underline", "Crossed", "Dark", "Bold, Italic & Underlined"});

  styled_table[0][0].format().font_style({emphasis::bold});

  styled_table[0][1].format().font_style({emphasis::italic});

  styled_table[0][2].format().font_style(emphasis::bold| emphasis::italic);

  styled_table[0][3].format().font_style({emphasis::blink});

  styled_table[1][0].format().font_style({emphasis::underline});

  styled_table[1][1].format().font_style({emphasis::strikethrough});

  styled_table[1][2].format().font_style({emphasis::faint});

  styled_table[1][3].format().font_style(
      emphasis::bold| emphasis::italic| emphasis::underline);

  std::cout << styled_table << std::endl;
}