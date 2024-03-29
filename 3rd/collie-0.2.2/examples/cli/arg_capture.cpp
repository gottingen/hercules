// Copyright 2024 The Elastic AI Search Authors.
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


#include <collie/cli/cli.h>
#include <iostream>
#include <string>

/** This example demonstrates the use of `prefix_command` on a subcommand
to capture all subsequent arguments along with an alias to make it appear as a regular options.

All the values after the "sub" or "--sub" are available in the remaining() method.
*/
int main(int argc, const char *argv[]) {

    int value{0};
    collie::App app{"Test App"};
    app.add_option("-v", value, "value");

    auto *subcom = app.add_subcommand("sub", "")->prefix_command();
    subcom->alias("--sub");
    COLLIE_CLI_PARSE(app, argc, argv);

    std::cout << "value=" << value << '\n';
    std::cout << "after Args:";
    for(const auto &aarg : subcom->remaining()) {
        std::cout << aarg << " ";
    }
    std::cout << '\n';
}
