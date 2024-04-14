// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
//
// Created by jeff on 23-11-28.
//

#include "turbo/format/table.h"

using namespace turbo;
using Row_t = Table::Row_t;

int main() {
    Table no_padding;
    no_padding.format().font_style({emphasis::bold}).padding(0);
    no_padding.add_row(Row_t{"This cell has no padding"});
    std::cout << "Table with no padding:\n" << no_padding << std::endl;

    Table padding_top_bottom;
    padding_top_bottom.format().font_style({emphasis::bold}).padding(0);
    padding_top_bottom.add_row(
            Row_t{"This cell has padding top = 1", "This cell has padding bottom = 1"});
    padding_top_bottom[0][0].format().padding_top(1);
    padding_top_bottom[0][1].format().padding_bottom(1);

    std::cout << "\nTable with top/bottom padding:\n" << padding_top_bottom << std::endl;
}