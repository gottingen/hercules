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
    Table movies;
    movies.add_row(Row_t{"S/N", "Movie Name", "Director", "Estimated Budget", "Release Date"});
    movies.add_row(Row_t{"tt1979376", "Toy Story 4", "Josh Cooley", "$200,000,000", "21 June 2019"});
    movies.add_row(Row_t{"tt3263904", "Sully", "Clint Eastwood", "$60,000,000", "9 September 2016"});
    movies.add_row(
            Row_t{"tt1535109", "Captain Phillips", "Paul Greengrass", "$55,000,000", " 11 October 2013"});

    // center align 'Director' column
    movies.column(2).format().font_align(FontAlign::center);

    // right align 'Estimated Budget' column
    movies.column(3).format().font_align(FontAlign::right);

    // right align 'Release Date' column
    movies.column(4).format().font_align(FontAlign::right);

    // Color header cells
    for (size_t i = 0; i < 5; ++i) {
        movies[0][i]
                .format()
                .font_color(fg(color::white))
                .font_style({emphasis::bold})
                .font_color(bg(color::blue));
    }

    LatexExporter exporter;
    exporter.configure().indentation(8);
    auto latex = exporter.dump(movies);

    // tabulate::table
    std::cout << movies << "\n\n";

    // Exported Markdown
    std::cout << latex << std::endl;
}
