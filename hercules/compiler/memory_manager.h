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

#include <mutex>
#include <utility>
#include <vector>

#include "hercules/cir/llvm/llvm.h"

namespace hercules {

    /// Simple extension of LLVM's SectionMemoryManager which catches data section
    /// allocations and registers them with the GC. This allows the GC to know not
    /// to collect globals even in JIT mode.
    class BoehmGCMemoryManager : public llvm::SectionMemoryManager {
    private:
        /// Vector of (start, end) address pairs registered with GC.
        std::vector<std::pair<void *, void *>> roots;

        uint8_t *allocateDataSection(uintptr_t size, unsigned alignment, unsigned sectionID,
                                     llvm::StringRef sectionName, bool isReadOnly) override;

    public:
        BoehmGCMemoryManager();

        ~BoehmGCMemoryManager() override;
    };

    /// Basically a copy of LLVM's jitlink::InProcessMemoryManager that registers
    /// relevant allocated sections with the GC. TODO: Avoid copying this entire
    /// class if/when there's an API to perform the registration externally.
    class BoehmGCJITLinkMemoryManager : public llvm::jitlink::JITLinkMemoryManager {
    public:
        class IPInFlightAlloc;

        /// Attempts to auto-detect the host page size.
        static llvm::Expected<std::unique_ptr<BoehmGCJITLinkMemoryManager>> Create();

        /// Create an instance using the given page size.
        BoehmGCJITLinkMemoryManager(uint64_t PageSize) : PageSize(PageSize) {}

        void allocate(const llvm::jitlink::JITLinkDylib *JD, llvm::jitlink::LinkGraph &G,
                      OnAllocatedFunction OnAllocated) override;

        // Use overloads from base class.
        using llvm::jitlink::JITLinkMemoryManager::allocate;

        void deallocate(std::vector<FinalizedAlloc> Alloc,
                        OnDeallocatedFunction OnDeallocated) override;

        // Use overloads from base class.
        using llvm::jitlink::JITLinkMemoryManager::deallocate;

    private:
        // FIXME: Use an in-place array instead of a vector for DeallocActions.
        //        There shouldn't need to be a heap alloc for this.
        struct FinalizedAllocInfo {
            llvm::sys::MemoryBlock StandardSegments;
            std::vector<llvm::orc::shared::WrapperFunctionCall> DeallocActions;
        };

        FinalizedAlloc createFinalizedAlloc(
                llvm::sys::MemoryBlock StandardSegments,
                std::vector<llvm::orc::shared::WrapperFunctionCall> DeallocActions);

        uint64_t PageSize;
        std::mutex FinalizedAllocsMutex;
        llvm::RecyclingAllocator<llvm::BumpPtrAllocator, FinalizedAllocInfo> FinalizedAllocInfos;
    };

} // namespace hercules
