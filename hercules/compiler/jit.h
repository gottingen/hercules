// Copyright 2023 The titan-search Authors.
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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hercules/cir/llvm/llvisitor.h"
#include "hercules/cir/transform/manager.h"
#include "hercules/cir/var.h"
#include "hercules/compiler/compiler.h"
#include "hercules/compiler/engine.h"
#include "hercules/compiler/error.h"
#include "hercules/parser/cache.h"
#include "hercules/runtime/lib.h"

#include "hercules/compiler/jit_extern.h"

namespace hercules::jit {

    class JIT {
    public:
        struct PythonData {
            ir::types::Type *cobj;
            std::unordered_map<std::string, ir::Func *> cache;

            PythonData();

            ir::types::Type *getCObjType(ir::Module *M);
        };

    private:
        std::unique_ptr<Compiler> compiler;
        std::unique_ptr<Engine> engine;
        std::unique_ptr<PythonData> pydata;
        std::string mode;

    public:
        explicit JIT(const std::string &argv0, const std::string &mode = "");

        Compiler *getCompiler() const { return compiler.get(); }

        Engine *getEngine() const { return engine.get(); }

        // General
        llvm::Error init();

        llvm::Error compile(const ir::Func *input);

        llvm::Expected<ir::Func *> compile(const std::string &code,
                                           const std::string &file = "", int line = 0);

        llvm::Expected<void *> address(const ir::Func *input);

        llvm::Expected<std::string> run(const ir::Func *input);

        llvm::Expected<std::string> execute(const std::string &code,
                                            const std::string &file = "", int line = 0,
                                            bool debug = false);

        // Python
        llvm::Expected<void *> runPythonWrapper(const ir::Func *wrapper, void *arg);

        llvm::Expected<ir::Func *> getWrapperFunc(const std::string &name,
                                                  const std::vector<std::string> &types);

        JITResult executePython(const std::string &name,
                                const std::vector<std::string> &types,
                                const std::string &pyModule,
                                const std::vector<std::string> &pyVars, void *arg,
                                bool debug);

        JITResult executeSafe(const std::string &code, const std::string &file, int line,
                              bool debug);

        // Errors
        llvm::Error handleJITError(const runtime::JITError &e);
    };

} // namespace hercules::jit
