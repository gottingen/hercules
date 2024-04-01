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
#include <hercules/util/jupyter.h>
#include <hercules/engine/vm.h>
#include <collie/cli/cli.h>


int jupyterMode(const std::vector<const char *> &args) {
    llvm::cl::list<std::string> plugins("plugin",
                                        llvm::cl::desc("Load specified plugin"));
    llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                     llvm::cl::desc("<connection file>"),
                                     llvm::cl::init("connection.json"));
    llvm::cl::ParseCommandLineOptions(args.size(), args.data());
    int code = hercules::startJupyterKernel(args[0], plugins, input);
    return code;
}

void showCommandsAndExit() {
    hercules::compilationError("Available commands: hercules <run|build|doc>");
}

int otherMode(const std::vector<const char *> &args) {
    llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::desc("<mode>"));
    llvm::cl::extrahelp("\nMODES:\n\n"
                        "  run   - run a program interactively\n"
                        "  build - build a program\n"
                        "  doc   - generate program documentation\n");
    llvm::cl::ParseCommandLineOptions(args.size(), args.data());

    if (!input.empty())
        showCommandsAndExit();
    return EXIT_SUCCESS;
}

int main(int argc, const char **argv) {
    if (argc < 2)
        showCommandsAndExit();

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

    COLLIE_CLI_PARSE(app, argc, argv);
    hercules::tidy_program_args();

    hercules::EngineVM vm;
    if (ins.mode == "run") {
        auto r = vm.prepare_run();
        if (r != EXIT_SUCCESS)
            return r;
        return vm.run();
    }
    if (ins.mode == "build") {
        return vm.build(ins.orig_argv0);
    }
    if (ins.mode == "doc") {
        return vm.document();
    }
    if (ins.mode == "jit") {
        return vm.jit();
    }
    if (ins.mode == "jupyter") {
        return jupyterMode(ins.args);
    }
    return otherMode({argv, argv + argc});
}
