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
    Table class_diagram;

    // Global styling
    class_diagram.format().font_style({emphasis::bold}).font_align(FontAlign::center).width(60);

    // Animal class
    Table animal;
    animal.add_row(Row_t{"Animal"});
    animal[0].format().font_align(FontAlign::center);

    // Animal properties nested table
    Table animal_properties;
    animal_properties.format().width(20);
    animal_properties.add_row(Row_t{"+age: Int"});
    animal_properties.add_row(Row_t{"+gender: String"});
    animal_properties[1].format().hide_border_top();

    // Animal methods nested table
    Table animal_methods;
    animal_methods.format().width(20);
    animal_methods.add_row(Row_t{"+isMammal()"});
    animal_methods.add_row(Row_t{"+mate()"});
    animal_methods[1].format().hide_border_top();

    animal.add_row(Row_t{animal_properties});
    animal.add_row(Row_t{animal_methods});
    animal[2].format().hide_border_top();

    class_diagram.add_row(Row_t{animal});

    // Add rows in the class diagram for the up-facing arrow
    // THanks to center alignment, these will align just fine
    class_diagram.add_row(Row_t{"▲"});
    class_diagram[1][0].format().hide_border_top().multi_byte_characters(true);
    class_diagram.add_row(Row_t{"|"});
    class_diagram[2].format().hide_border_top();
    class_diagram.add_row(Row_t{"|"});
    class_diagram[3].format().hide_border_top();

    // Duck class
    Table duck;
    duck.add_row(Row_t{"Duck"});
    duck[0].format().font_align(FontAlign::center);

    // Duck proeperties nested table
    Table duck_properties;
    duck_properties.format().width(40);
    duck_properties.add_row(Row_t{"+beakColor: String = \"yellow\""});

    // Duck methods nested table
    Table duck_methods;
    duck_methods.format().width(40);
    duck_methods.add_row(Row_t{"+swim()"});
    duck_methods.add_row(Row_t{"+quack()"});
    duck_methods[1].format().hide_border_top();

    duck.add_row(Row_t{duck_properties});
    duck.add_row(Row_t{duck_methods});
    duck[2].format().hide_border_top();

    class_diagram.add_row(Row_t{duck});
    class_diagram[4].format().hide_border_top();

    std::cout << class_diagram << std::endl;
}