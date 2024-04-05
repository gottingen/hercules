// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <collie/cli/cli.h>
#include <string>

/// Collection of all options of Subcommand A.
struct SubcommandAOptions {
    std::string file;
    bool with_foo;
};

// We could manually make a few variables and use shared pointers for each; this
// is just done this way to be nicely organized

// Function declarations.
void setup_subcommand_a(collie::App &app);
void run_subcommand_a(SubcommandAOptions const &opt);
