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
#include <sstream>

// example file to demonstrate a custom lexical cast function

template<class T = int>
struct Values {
    T a;
    T b;
    T c;
};

// in C++20 this is constructible from a double due to the new aggregate initialization in C++20.
using DoubleValues = Values<double>;

// the lexical cast operator should be in the same namespace as the type for ADL to work properly
bool lexical_cast(const std::string &input, Values<double> & /*v*/) {
    std::cout << "called correct lexical_cast function ! val: " << input << std::endl;
    return true;
}

DoubleValues doubles;

void argparse(turbo::OptionGroup *group) { group->add_option("--dv", doubles)->default_str("0"); }

int main(int argc, char **argv) {
    turbo::App app;

    argparse(app.add_option_group("param"));
    TURBO_FLAGS_PARSE(app, argc, argv);
    return 0;
}
