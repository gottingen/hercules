// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    collie::App cli_global{"Demo app"};
    auto &cli_sub = *cli_global.add_subcommand("sub", "Some subcommand");
    std::string sub_arg;
    cli_sub.add_option("sub_arg", sub_arg, "Argument for subcommand")->required();
    COLLIE_CLI_PARSE(cli_global, argc, argv);
    if(cli_sub) {
        std::cout << "Got: " << sub_arg << '\n';
    }
    return 0;
}
