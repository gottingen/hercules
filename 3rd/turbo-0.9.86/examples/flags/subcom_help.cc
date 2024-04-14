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

int main(int argc, char *argv[]) {
    turbo::App cli_global{"Demo app"};
    auto &cli_sub = *cli_global.add_subcommand("sub", "Some subcommand");
    std::string sub_arg;
    cli_sub.add_option("sub_arg", sub_arg, "Argument for subcommand")->required();
    TURBO_FLAGS_PARSE(cli_global, argc, argv);
    if(cli_sub) {
        std::cout << "Got: " << sub_arg << std::endl;
    }
    return 0;
}
