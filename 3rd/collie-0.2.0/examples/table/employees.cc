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

#include <iomanip>
#include <collie/table/table.h>
using namespace collie::table;
using Row_t = Table::Row_t;

int main() {
  Table employees;

  Table department;
  department.add_row(Row_t{"Research", "Engineering"});

  RowStream rs;
  rs << std::setprecision(10);

  // Add rows
  // clang-format off
  employees.add_row(Row_t{"Emp. ID", "First Name", "Last Name", "Department / Business Unit", "Pay Rate"});
  employees.add_row(RowStream{}.copyfmt(rs) << 101 << "Donald" << "Patrick" << "Finance" << 59.61538461);
  employees.add_row(RowStream{}.copyfmt(rs) << 102 << "Rachel" << "Williams" << "Marketing and Operational\nLogistics Planning" << 34.97067307);
  employees.add_row(RowStream{}.copyfmt(rs) << 103 << "Ian" << "Jacob" << department << 57.00480769);
  // clang-format on

  employees.column(0)
      .format()
      .font_style({FontStyle::bold})
      .font_color(Color::white)
      .font_align(FontAlign::right);

  employees.column(3).format().font_align(FontAlign::center);

  // Print the table
  std::cout << employees << std::endl;
}