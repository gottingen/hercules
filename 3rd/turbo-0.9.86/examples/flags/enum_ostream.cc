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
#include "turbo/flags/flags.h"
#include <iostream>
#include <map>
#include <string>

enum class Level : int { High, Medium, Low };

inline std::ostream &operator<<(std::ostream &os, const Level &level) {
    switch(level) {
    case Level::High:
        os << "High";
        break;
    case Level::Medium:
        os << "Medium";
        break;
    case Level::Low:
        os << "Low";
        break;
    }
    os << " (ft rom custom ostream)";
    return os;
}

int main(int argc, char **argv) {
    turbo::App app;

    Level level{Level::Low};
    // specify string->value mappings
    std::map<std::string, Level> map{{"high", Level::High}, {"medium", Level::Medium}, {"low", Level::Low}};
    // CheckedTransformer translates and checks whether the results are either in one of the strings or in one of the
    // translations already
    app.add_option("-l,--level", level, "Level settings")
        ->required()
        ->transform(turbo::CheckedTransformer(map, turbo::ignore_case));

    TURBO_FLAGS_PARSE(app, argc, argv);

    using turbo::enums::operator<<;
    std::cout << "Enum received: " << level << std::endl;

    return 0;
}
