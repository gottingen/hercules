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

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include <hercules/hir/llvm/llvm.h>

namespace hercules {

    /// Debug info tracker for MCJIT.
    class DebugListener : public llvm::JITEventListener {
    public:
        class ObjectInfo {
        private:
            ObjectKey key;
            std::unique_ptr<llvm::object::ObjectFile> object;
            std::unique_ptr<llvm::MemoryBuffer> buffer;
            uintptr_t start;
            uintptr_t stop;

        public:
            ObjectInfo(ObjectKey key, std::unique_ptr<llvm::object::ObjectFile> object,
                       std::unique_ptr<llvm::MemoryBuffer> buffer, uintptr_t start,
                       uintptr_t stop)
                    : key(key), object(std::move(object)), buffer(std::move(buffer)), start(start),
                      stop(stop) {}

            ObjectKey getKey() const { return key; }

            const llvm::object::ObjectFile &getObject() const { return *object; }

            uintptr_t getStart() const { return start; }

            uintptr_t getStop() const { return stop; }

            bool contains(uintptr_t pc) const { return start <= pc && pc < stop; }
        };

    private:
        std::vector<ObjectInfo> objects;

        void notifyObjectLoaded(ObjectKey key, const llvm::object::ObjectFile &obj,
                                const llvm::RuntimeDyld::LoadedObjectInfo &L) override;

        void notifyFreeingObject(ObjectKey key) override;

    public:
        DebugListener() : llvm::JITEventListener(), objects() {}

        llvm::Expected<llvm::DILineInfo> symbolize(uintptr_t pc);

        llvm::Expected<std::string> getPrettyBacktrace(uintptr_t pc);

        std::string getPrettyBacktrace(const std::vector<uintptr_t> &backtrace);
    };

    /// Debug info tracker for JITLink. Adapted from Julia's implementation:
    /// https://github.com/JuliaLang/julia/blob/master/src/jitlayers.cpp
    class DebugPlugin : public llvm::orc::ObjectLinkingLayer::Plugin {
        struct JITObjectInfo {
            std::unique_ptr<llvm::MemoryBuffer> backingBuffer;
            std::unique_ptr<llvm::object::ObjectFile> object;
            llvm::StringMap<uint64_t> sectionLoadAddresses;
        };

        std::mutex pluginMutex;
        std::map<llvm::orc::MaterializationResponsibility *, std::unique_ptr<JITObjectInfo>>
                pendingObjs;
        std::map<llvm::orc::ResourceKey, std::vector<std::unique_ptr<JITObjectInfo>>>
                registeredObjs;

    public:
        void notifyMaterializing(llvm::orc::MaterializationResponsibility &mr,
                                 llvm::jitlink::LinkGraph &graph,
                                 llvm::jitlink::JITLinkContext &ctx,
                                 llvm::MemoryBufferRef inputObject) override;

        llvm::Error notifyEmitted(llvm::orc::MaterializationResponsibility &mr) override;

        llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility &mr) override;

        llvm::Error notifyRemovingResources(llvm::orc::JITDylib &jd,
                                            llvm::orc::ResourceKey key) override;

        void notifyTransferringResources(llvm::orc::JITDylib &jd,
                                         llvm::orc::ResourceKey dstKey,
                                         llvm::orc::ResourceKey srcKey) override;

        void modifyPassConfig(llvm::orc::MaterializationResponsibility &mr,
                              llvm::jitlink::LinkGraph &,
                              llvm::jitlink::PassConfiguration &config) override;

        llvm::Expected<llvm::DILineInfo> symbolize(uintptr_t pc);

        llvm::Expected<std::string> getPrettyBacktrace(uintptr_t pc);

        std::string getPrettyBacktrace(const std::vector<uintptr_t> &backtrace);
    };

} // namespace hercules
