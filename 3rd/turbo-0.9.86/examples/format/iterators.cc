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
  Table table;

  table.add_row(Row_t{"Company", "Contact", "Country"});
  table.add_row(Row_t{"Alfreds Futterkiste", "Maria Anders", "Germany"});
  table.add_row(Row_t{"Centro comercial Moctezuma", "Francisco Chang", "Mexico"});
  table.add_row(Row_t{"Ernst Handel", "Roland Mendel", "Austria"});
  table.add_row(Row_t{"Island Trading", "Helen Bennett", "UK"});
  table.add_row(Row_t{"Laughing Bacchus Winecellars", "Yoshi Tannamuri", "Canada"});
  table.add_row(Row_t{"Magazzini Alimentari Riuniti", "Giovanni Rovelli", "Italy"});

  // Set width of cells in each column
  table.column(0).format().width(40);
  table.column(1).format().width(30);
  table.column(2).format().width(30);

  // Iterate over cells in the first row
  for (auto &cell : table[0]) {
    cell.format().font_style({emphasis::underline}).font_align(FontAlign::center);
  }

  // Iterator over cells in the second column
  for (auto &cell : table.column(0)) {
    if (cell.get_text() != "Company") {
      cell.format().font_align(FontAlign::right);
    }
  }

  // Iterate over rows in the table
  size_t index = 0;
  for (auto &row : table) {
    row.format().font_style({emphasis::bold});

    // Set blue background color for alternate rows
    if (index > 0 && index % 2 == 0) {
      for (auto &cell : row) {
        cell.format().font_color(bg(color::blue));
      }
    }
    index += 1;
  }

  std::cout << table << std::endl;
}