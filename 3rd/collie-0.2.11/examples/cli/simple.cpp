// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {

    collie::App app("K3Pi goofit fitter");
    // add version output
    app.set_version_flag("--version", std::string(COLLIE_CLI_VERSION));
    std::string file;
    collie::Option *opt = app.add_option("-f,--file,file", file, "File name");

    int count{0};
    collie::Option *copt = app.add_option("-c,--count", count, "Counter");

    int v{0};
    collie::Option *flag = app.add_flag("--flag", v, "Some flag that can be passed multiple times");

    double value{0.0};  // = 3.14;
    app.add_option("-d,--double", value, "Some Value");

    COLLIE_CLI_PARSE(app, argc, argv);

    std::cout << "Working on file: " << file << ", direct count: " << app.count("--file")
              << ", opt count: " << opt->count() << '\n';
    std::cout << "Working on count: " << count << ", direct count: " << app.count("--count")
              << ", opt count: " << copt->count() << '\n';
    std::cout << "Received flag: " << v << " (" << flag->count() << ") times\n";
    std::cout << "Some value: " << value << '\n';

    return 0;
}
