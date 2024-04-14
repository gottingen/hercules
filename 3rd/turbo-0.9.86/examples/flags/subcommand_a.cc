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
#include "subcommand_a.h"
#include <iostream>
#include <memory>

/// Set up a subcommand and capture a shared_ptr to a struct that holds all its options.
/// The variables of the struct are bound to the CLI options.
/// We use a shared ptr so that the addresses of the variables remain for binding,
/// You could return the shared pointer if you wanted to access the values in main.
void setup_subcommand_a(turbo::App &app) {
    // Create the option and subcommand objects.
    auto opt = std::make_shared<SubcommandAOptions>();
    auto *sub = app.add_subcommand("subcommand_a", "performs subcommand a");

    // Add options to sub, binding them to opt.
    sub->add_option("-f,--file", opt->file, "File name")->required();
    sub->add_flag("--with-foo", opt->with_foo, "Counter");

    // Set the run function as callback to be called when this subcommand is issued.
    sub->callback([opt]() { run_subcommand_a(*opt); });
}

/// The function that runs our code.
/// This could also simply be in the callback lambda itself,
/// but having a separate function is cleaner.
void run_subcommand_a(SubcommandAOptions const &opt) {
    // Do stuff...
    std::cout << "Working on file: " << opt.file << std::endl;
    if (opt.with_foo) {
        std::cout << "Using foo!" << std::endl;
    }
}
