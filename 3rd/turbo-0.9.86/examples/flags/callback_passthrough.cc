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
#include <string>

int main(int argc, char **argv) {

    turbo::App app("callback_passthrough");
    app.allow_extras();
    std::string argName;
    std::string val;
    app.add_option("--argname", argName, "the name of the custom command line argument");
    app.callback([&app, &val, &argName]() {
        if(!argName.empty()) {
            turbo::App subApp;
            subApp.add_option("--" + argName, val, "custom argument option");
            subApp.parse(app.remaining_for_passthrough());
        }
    });

    TURBO_FLAGS_PARSE(app, argc, argv);
    std::cout << "the value is now " << val << '\n';
}
