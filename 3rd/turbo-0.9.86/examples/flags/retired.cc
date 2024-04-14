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
#include <utility>
#include <vector>

// This example shows the usage of the retired and deprecated option helper methods
int main(int argc, char **argv) {

    turbo::App app("example for retired/deprecated options");
    std::vector<int> x;
    auto *opt1 = app.add_option("--retired_option2", x);

    std::pair<int, int> y;
    auto *opt2 = app.add_option("--deprecate", y);

    app.add_option("--not_deprecated", x);

    // specify that a non-existing option is retired
    turbo::retire_option(app, "--retired_option");

    // specify that an existing option is retired and non-functional: this will replace the option with another that
    // behaves the same but does nothing
    turbo::retire_option(app, opt1);

    // deprecate an existing option and specify the recommended replacement
    turbo::deprecate_option(opt2, "--not_deprecated");

    TURBO_FLAGS_PARSE(app, argc, argv);

    if(!x.empty()) {
        std::cout << "Retired option example: got --not_deprecated values:";
        for(auto &xval : x) {
            std::cout << xval << " ";
        }
        std::cout << '\n';
    } else if(app.count_all() == 1) {
        std::cout << "Retired option example: no arguments received\n";
    }
    return 0;
}
