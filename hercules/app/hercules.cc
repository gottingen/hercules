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
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <hercules/engine/vm.h>


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

    llvm::cl::SetVersionPrinter(hercules::version_dump);
    std::vector<const char *> args{argv[0]};
    for (int i = 2; i < argc; i++)
        args.push_back(argv[i]);

    std::string mode(argv[1]);
    std::string argv0 = std::string(args[0]) + " " + mode;
    hercules::EngineVM vm;
    if (mode == "run") {
        args[0] = argv0.data();
        auto r = vm.prepare_run(args);
        if (r != EXIT_SUCCESS)
            return r;
        return vm.run();
    }
    if (mode == "build") {
        const char *oldArgv0 = args[0];
        args[0] = argv0.data();
        return vm.build(args, oldArgv0);
    }
    if (mode == "doc") {
        const char *oldArgv0 = args[0];
        args[0] = argv0.data();
        return vm.document(args, oldArgv0);
    }
    if (mode == "jit") {
        args[0] = argv0.data();
        return vm.jit(args);
    }
    if (mode == "jupyter") {
        args[0] = argv0.data();
        return jupyterMode(args);
    }
    return otherMode({argv, argv + argc});
}
