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

    turbo::App app("K3Pi goofit fitter");
    app.set_help_all_flag("--help-all", "Expand all help");
    app.add_flag("--random", "Some random flag");
    turbo::App *start = app.add_subcommand("start", "A great subcommand");
    turbo::App *stop = app.add_subcommand("stop", "Do you really want to stop?");
    app.require_subcommand();  // 1 or more

    std::string file;
    start->add_option("-f,--file", file, "File name");

    turbo::Option *s = stop->add_flag("-c,--count", "Counter");

    TURBO_FLAGS_PARSE(app, argc, argv);

    std::cout << "Working on --file from start: " << file << std::endl;
    std::cout << "Working on --count from stop: " << s->count() << ", direct count: " << stop->count("--count")
              << std::endl;
    std::cout << "Count of --random flag: " << app.count("--random") << std::endl;
    for(auto *subcom : app.get_subcommands())
        std::cout << "Subcommand: " << subcom->get_name() << std::endl;

    return 0;
}
