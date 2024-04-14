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

    turbo::App app("test for positional arity");

    auto *numbers = app.add_option_group("numbers", "specify key numbers");
    auto *files = app.add_option_group("files", "specify files");
    int num1{-1}, num2{-1};
    numbers->add_option("num1", num1, "first number");
    numbers->add_option("num2", num2, "second number");
    std::string file1, file2;
    files->add_option("file1", file1, "first file")->required();
    files->add_option("file2", file2, "second file");
    // set a pre parse callback that turns the numbers group on or off depending on the number of arguments
    app.preparse_callback([numbers](std::size_t arity) {
        if(arity <= 2) {
            numbers->disabled();
        } else {
            numbers->disabled(false);
        }
    });

    TURBO_FLAGS_PARSE(app, argc, argv);

    if(num1 != -1)
        std::cout << "Num1 = " << num1 << '\n';

    if(num2 != -1)
        std::cout << "Num2 = " << num2 << '\n';

    std::cout << "File 1 = " << file1 << '\n';
    if(!file2.empty()) {
        std::cout << "File 2 = " << file2 << '\n';
    }

    return 0;
}
