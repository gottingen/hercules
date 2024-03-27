// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {

    collie::App app("Validator checker");

    std::string file;
    app.add_option("-f,--file,file", file, "File name")->check(collie::ExistingFile);

    int count{0};
    app.add_option("-v,--value", count, "Value in range")->check(collie::Range(3, 6));
    COLLIE_CLI_PARSE(app, argc, argv);

    std::cout << "Try printing help or failing the validator" << '\n';

    return 0;
}
