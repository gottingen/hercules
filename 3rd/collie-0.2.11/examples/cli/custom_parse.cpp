// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <iostream>
#include <sstream>

// example file to demonstrate a custom lexical cast function

template <class T = int> struct Values {
    T a;
    T b;
    T c;
};

// in C++20 this is constructible from a double due to the new aggregate initialization in C++20.
using DoubleValues = Values<double>;

// the lexical cast operator should be in the same namespace as the type for ADL to work properly
bool lexical_cast(const std::string &input, Values<double> & /*v*/) {
    std::cout << "called correct lexical_cast function ! val: " << input << '\n';
    return true;
}

DoubleValues doubles;
void argparse(collie::Option_group *group) { group->add_option("--dv", doubles)->default_str("0"); }

int main(int argc, char **argv) {
    collie::App app;

    argparse(app.add_option_group("param"));
    COLLIE_CLI_PARSE(app, argc, argv);
    return 0;
}
