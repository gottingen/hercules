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
    Table employees;

    Table department;
    department.add_row(Row_t{"Research", "Engineering"});

    // Add rows
    employees.add_row(Row_t{"Emp. ID", "First Name", "Last Name", "Department / Business Unit"});
    employees.add_row(Row_t{"101", "Donald", "Patrick", "Finance"});
    employees.add_row(
            Row_t{"102", "Rachel", "Williams", "Marketing and Operational\nLogistics Planning"});
    employees.add_row(Row_t{"103", "Ian", "Jacob", department});

    employees.column(0)
            .format()
            .font_style({emphasis::bold})
            .font_color(color::white)
            .font_align(FontAlign::right);

    employees.column(3).format().font_align(FontAlign::center);

    // Print the table
    std::cout << employees << std::endl;
}