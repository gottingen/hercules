// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <iostream>

int main(int argc, char **argv) {
    collie::App app;

    int val{0};
    // add a set of flags with default values associate with them
    app.add_flag("-1{1},-2{2},-3{3},-4{4},-5{5},-6{6}, -7{7}, -8{8}, -9{9}", val, "compression level");

    COLLIE_CLI_PARSE(app, argc, argv);

    std::cout << "value = " << val << '\n';
    return 0;
}