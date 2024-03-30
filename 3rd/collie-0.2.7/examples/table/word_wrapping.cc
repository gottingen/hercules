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

#include <collie/table/table.h>

using namespace collie::table;
using Row_t = Table::Row_t;

int main() {
    Table table;

    table.add_row(Row_t{"This paragraph contains a veryveryveryveryveryverylong "
                        "word. The long word will "
                        "break and word wrap to the next line.",
                        "This paragraph \nhas embedded '\\n' \ncharacters and\n "
                        "will break\n exactly "
                        "where\n you want it\n to\n break."});

    table[0][0].format().width(20);
    table[0][1].format().width(50);

    std::cout << table << std::endl;
}