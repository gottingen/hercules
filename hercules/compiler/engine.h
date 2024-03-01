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
#include <vector>

#include "hercules/hir/llvm/llvm.h"
#include "hercules/compiler/debug_listener.h"

namespace hercules::jit {

    class Engine {
    private:
        std::unique_ptr<llvm::orc::ExecutionSession> sess;
        std::unique_ptr<llvm::orc::EPCIndirectionUtils> epciu;

        llvm::DataLayout layout;
        llvm::orc::MangleAndInterner mangle;

        llvm::orc::RTDyldObjectLinkingLayer objectLayer;
        llvm::orc::IRCompileLayer compileLayer;
        llvm::orc::IRTransformLayer optimizeLayer;
        llvm::orc::CompileOnDemandLayer codLayer;

        llvm::orc::JITDylib &mainJD;

        std::unique_ptr<DebugListener> dbListener;

        static void handleLazyCallThroughError();

        static llvm::Expected<llvm::orc::ThreadSafeModule>
        optimizeModule(llvm::orc::ThreadSafeModule module,
                       const llvm::orc::MaterializationResponsibility &R);

    public:
        Engine(std::unique_ptr<llvm::orc::ExecutionSession> sess,
               std::unique_ptr<llvm::orc::EPCIndirectionUtils> epciu,
               llvm::orc::JITTargetMachineBuilder jtmb, llvm::DataLayout layout);

        ~Engine();

        static llvm::Expected<std::unique_ptr<Engine>> create();

        const llvm::DataLayout &getDataLayout() const { return layout; }

        llvm::orc::JITDylib &getMainJITDylib() { return mainJD; }

        DebugListener *getDebugListener() const { return dbListener.get(); }

        llvm::Error addModule(llvm::orc::ThreadSafeModule module,
                              llvm::orc::ResourceTrackerSP rt = nullptr);

        llvm::Expected<llvm::orc::ExecutorSymbolDef> lookup(llvm::StringRef name);
    };

} // namespace hercules::jit
