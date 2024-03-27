// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <iostream>
#include <map>
#include <string>

enum class Level : int { High, Medium, Low };

int main(int argc, char **argv) {
    collie::App app;

    Level level{Level::Low};
    // specify string->value mappings
    std::map<std::string, Level> map{{"high", Level::High}, {"medium", Level::Medium}, {"low", Level::Low}};
    // CheckedTransformer translates and checks whether the results are either in one of the strings or in one of the
    // translations already
    app.add_option("-l,--level", level, "Level settings")
        ->required()
        ->transform(collie::CheckedTransformer(map, collie::ignore_case));

    COLLIE_CLI_PARSE(app, argc, argv);

    using collie::enums::operator<<;
    std::cout << "Enum received: " << level << '\n';

    return 0;
}
