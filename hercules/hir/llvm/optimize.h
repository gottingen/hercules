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

#include "hercules/hir/llvm/llvm.h"
#include "hercules/dsl/plugins.h"

namespace hercules::ir {
    std::unique_ptr<llvm::TargetMachine>
    getTargetMachine(llvm::Triple triple, llvm::StringRef cpuStr,
                     llvm::StringRef featuresStr, const llvm::TargetOptions &options,
                     bool pic = false);

    std::unique_ptr<llvm::TargetMachine>
    getTargetMachine(llvm::Module *module, bool setFunctionAttributes = false,
                     bool pic = false);

    void optimize(llvm::Module *module, bool debug, bool jit = false,
                  PluginManager *plugins = nullptr);
} // namespace hercules::ir
