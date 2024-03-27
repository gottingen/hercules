// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause


#include <collie/cli/cli.h>
#include <iostream>
#include <string>

int main(int argc, const char *argv[]) {

    int value{0};
    collie::App app{"Test App"};
    app.add_option("-v", value, "value");

    auto *subcom = app.add_subcommand("sub", "")->prefix_command();
    COLLIE_CLI_PARSE(app, argc, argv);

    std::cout << "value =" << value << '\n';
    std::cout << "after Args:";
    for(const auto &aarg : subcom->remaining()) {
        std::cout << aarg << " ";
    }
    std::cout << '\n';
}
