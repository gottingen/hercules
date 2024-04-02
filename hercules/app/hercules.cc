// Copyright 2024 The EA Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <hercules/compiler/compiler.h>
#include <hercules/compiler/error.h>
#include <hercules/util/common.h>
#include <hercules/engine/vm.h>
#include <hercules/app/jupyter/jupyter.h>
#include <collie/cli/cli.h>


int main(int argc, const char **argv) {

    auto& ins = hercules::VmContext::instance();
    for (int i = 0; i < argc; i++)
        ins.args.push_back(argv[i]);
    ins.orig_argv0 = argv[0];
    collie::App app("hercules", "Hercules Programming Language");
    app.add_subcommand("version", "Print the version of Hercules")->callback([]() {
        hercules::version_dump(std::cout);
    });
    /// set up run command
    auto *run = app.add_subcommand("run", "Run a program interactively");
    hercules::set_up_run_command(run);
    /// set up build command
    auto *build = app.add_subcommand("build", "Build a program");
    hercules::set_up_build_command(build);
    /// set up doc command
    auto *doc = app.add_subcommand("doc", "Generate program documentation");
    hercules::set_up_doc_command(doc);
    /// set up jit command
    auto *jit = app.add_subcommand("jit", "Run a program using JIT");
    hercules::set_up_jit_command(jit);
    /// set up jupyter command
    auto *jupyter = app.add_subcommand("jupyter", "Start a Jupyter kernel");
    hercules::set_up_jupyter_command(jupyter);
    app.parse_complete_callback([]() {
        hercules::tidy_program_args();
    });
    app.require_subcommand();

    COLLIE_CLI_PARSE(app, argc, argv);
    return ins.ret_code;
}
