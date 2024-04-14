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
#include <vector>

int main(int argc, char **argv) {

    turbo::App app{"App to demonstrate exclusionary option groups."};

    std::vector<int> range;
    app.add_option("--range,-R", range, "A range")->expected(-2);

    auto *ogroup = app.add_option_group("min_max_step", "set the min max and step");
    int min{0}, max{0}, step{1};
    ogroup->add_option("--min,-m", min, "The minimum")->required();
    ogroup->add_option("--max,-M", max, "The maximum")->required();
    ogroup->add_option("--step,-s", step, "The step")->capture_default_str();

    app.require_option(1);

    TURBO_FLAGS_PARSE(app, argc, argv);

    if(!range.empty()) {
        if(range.size() == 2) {
            min = range[0];
            max = range[1];
        }
        if(range.size() >= 3) {
            step = range[0];
            min = range[1];
            max = range[2];
        }
    }
    std::cout << "range is [" << min << ':' << step << ':' << max << "]\n";
    return 0;
}
