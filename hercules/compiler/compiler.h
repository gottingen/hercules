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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hercules/hir/llvm/llvisitor.h"
#include "hercules/hir/module.h"
#include "hercules/hir/transform/manager.h"
#include "hercules/compiler/error.h"
#include "hercules/dsl/plugins.h"
#include "hercules/parser/cache.h"

namespace hercules {

    class Compiler {
    public:
        enum Mode {
            DEBUG,
            RELEASE,
            JIT,
        };

    private:
        std::string argv0;
        bool debug;
        bool pyNumerics;
        bool pyExtension;
        std::string input;
        std::unique_ptr<PluginManager> plm;
        std::unique_ptr<ast::Cache> cache;
        std::unique_ptr<ir::Module> module;
        std::unique_ptr<ir::transform::PassManager> pm;
        std::unique_ptr<ir::LLVMVisitor> llvisitor;

        llvm::Error parse(bool isCode, const std::string &file, const std::string &code,
                          int startLine, int testFlags,
                          const std::unordered_map<std::string, std::string> &defines);

    public:
        Compiler(const std::string &argv0, Mode mode,
                 const std::vector<std::string> &disabledPasses = {}, bool isTest = false,
                 bool pyNumerics = false, bool pyExtension = false);

        explicit Compiler(const std::string &argv0, bool debug = false,
                          const std::vector<std::string> &disabledPasses = {},
                          bool isTest = false, bool pyNumerics = false,
                          bool pyExtension = false)
                : Compiler(argv0, debug ? Mode::DEBUG : Mode::RELEASE, disabledPasses, isTest,
                           pyNumerics, pyExtension) {}

        std::string getInput() const { return input; }

        PluginManager *getPluginManager() const { return plm.get(); }

        ast::Cache *getCache() const { return cache.get(); }

        ir::Module *getModule() const { return module.get(); }

        ir::transform::PassManager *getPassManager() const { return pm.get(); }

        ir::LLVMVisitor *getLLVMVisitor() const { return llvisitor.get(); }

        llvm::Error load(const std::string &plugin);

        llvm::Error
        parseFile(const std::string &file, int testFlags = 0,
                  const std::unordered_map<std::string, std::string> &defines = {});

        llvm::Error
        parseCode(const std::string &file, const std::string &code, int startLine = 0,
                  int testFlags = 0,
                  const std::unordered_map<std::string, std::string> &defines = {});

        llvm::Error compile();

        llvm::Expected<std::string> docgen(const std::vector<std::string> &files);

        std::unordered_map<std::string, std::string> getEarlyDefines();
    };

} // namespace hercules
