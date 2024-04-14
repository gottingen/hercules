// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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

#ifndef TURBO_BASE_THREADING_THREAD_NAME_REGISTRY_H_
#define TURBO_BASE_THREADING_THREAD_NAME_REGISTRY_H_

#include <map>
#include <string>
#include "turbo/platform/port.h"
#include "turbo/system/threading.h"
#include "turbo/concurrent/spinlock.h"

namespace turbo {


    class TURBO_DLL ThreadNameRegistry {
    public:
        ~ThreadNameRegistry() = default;

        static ThreadNameRegistry *get_instance();

        static const char *get_default_interned_string();

        // Register the mapping between a thread |id| and |handle|.
        void register_thread(PlatformThreadHandle::Handle handle, PlatformThreadId id);

        // Set the name for the given id.
        void set_name(PlatformThreadId id, const char *name);

        // Get the name for the given id.
        const char *get_name(PlatformThreadId id);

        // Remove the name for the given id.
        void remove_name(PlatformThreadHandle::Handle handle, PlatformThreadId id);

    private:

        using ThreadIdToHandleMap = std::map<PlatformThreadId, PlatformThreadHandle::Handle>;
        using ThreadHandleToInternedNameMap=std::map<PlatformThreadHandle::Handle, std::string *>;
        using NameToInternedNameMap = std::map<std::string, std::string *> ;

        ThreadNameRegistry();

        // lock_ protects the name_to_interned_name_, thread_id_to_handle_ and
        // thread_handle_to_interned_name_ maps.
        turbo::SpinLock lock_;

        NameToInternedNameMap name_to_interned_name_;
        ThreadIdToHandleMap thread_id_to_handle_;
        ThreadHandleToInternedNameMap thread_handle_to_interned_name_;

        // Treat the main process specially as there is no PlatformThreadHandle.
        std::string *main_process_name_;
        PlatformThreadId main_process_id_;

        // noinlint to avoid -Wthread-safety-analysis
        TURBO_NON_COPYABLE(ThreadNameRegistry);
    };


}  // namespace turbo

#endif  // TURBO_BASE_THREADING_THREAD_NAME_REGISTRY_H_
