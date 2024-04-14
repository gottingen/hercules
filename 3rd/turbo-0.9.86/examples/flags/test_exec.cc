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

int main(int argc, const char *argv[]) {

    int logLevel{0};
    turbo::App app{"Test App"};

    app.add_option("-v", logLevel, "level");

    auto *subcom = app.add_subcommand("sub", "")->fallthrough();
    subcom->preparse_callback([&app](size_t) { app.get_subcommand("sub")->add_option_group("group"); });

    TURBO_FLAGS_PARSE(app, argc, argv);

    std::cout << "level: " << logLevel << std::endl;
}
