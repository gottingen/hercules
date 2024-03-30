// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <collie/cli/Timer.hpp>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    collie::AutoTimer give_me_a_name("This is a timer");

    collie::App app("K3Pi goofit fitter");

    collie::App_p impOpt = std::make_shared<collie::App>("Important");
    std::string file;
    collie::Option *opt = impOpt->add_option("-f,--file,file", file, "File name")->required();

    int count{0};
    collie::Option *copt = impOpt->add_flag("-c,--count", count, "Counter")->required();

    collie::App_p otherOpt = std::make_shared<collie::App>("Other");
    double value{0.0};  // = 3.14;
    otherOpt->add_option("-d,--double", value, "Some Value");

    // add the subapps to the main one
    app.add_subcommand(impOpt);
    app.add_subcommand(otherOpt);

    try {
        app.parse(argc, argv);
    } catch(const collie::ParseError &e) {
        return app.exit(e);
    }

    std::cout << "Working on file: " << file << ", direct count: " << impOpt->count("--file")
              << ", opt count: " << opt->count() << '\n';
    std::cout << "Working on count: " << count << ", direct count: " << impOpt->count("--count")
              << ", opt count: " << copt->count() << '\n';
    std::cout << "Some value: " << value << '\n';

    return 0;
}
