// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {

    collie::App app("test for positional arity");
    std::string lib;
    app.add_option("-l, --lib", "list all the files");
    std::vector<std::string> files;
    app.add_option("file1", files, "first file")->required();

    COLLIE_CLI_PARSE(app, argc, argv);
    for(auto &file : files) {
        std::cout << "File = " << file << '\n';
    }
    return 0;
}
