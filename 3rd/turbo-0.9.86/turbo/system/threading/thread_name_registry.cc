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

#include "turbo/system/threading/thread_name_registry.h"
#include "turbo/log/logging.h"

namespace turbo {

    namespace {

        static const char kDefaultName[] = "";
        static std::string *g_default_name;

    }

    ThreadNameRegistry::ThreadNameRegistry()
            : main_process_name_(nullptr),
              main_process_id_(kInvalidThreadId) {
        g_default_name = new std::string(kDefaultName);

        turbo::SpinLockHolder locked(&lock_);
        name_to_interned_name_[kDefaultName] = g_default_name;
    }

    ThreadNameRegistry *ThreadNameRegistry::get_instance() {
        static ThreadNameRegistry instance;
        return &instance;
    }

    const char *ThreadNameRegistry::get_default_interned_string() {
        return g_default_name->c_str();
    }

    void ThreadNameRegistry::register_thread(PlatformThreadHandle::Handle handle,
                                             PlatformThreadId id) {
        turbo::SpinLockHolder locked(&lock_);
        thread_id_to_handle_[id] = handle;
        thread_handle_to_interned_name_[handle] =
                name_to_interned_name_[kDefaultName];
    }

    void ThreadNameRegistry::set_name(PlatformThreadId id, const char *name) {
        std::string str_name(name);

        turbo::SpinLockHolder locked(&lock_);
        auto iter = name_to_interned_name_.find(str_name);
        std::string *leaked_str = nullptr;
        if (iter != name_to_interned_name_.end()) {
            leaked_str = iter->second;
        } else {
            leaked_str = new std::string(str_name);
            name_to_interned_name_[str_name] = leaked_str;
        }

        auto id_to_handle_iter =
                thread_id_to_handle_.find(id);

        // The main thread of a process will not be created as a Thread object which
        // means there is no PlatformThreadHandler registered.
        if (id_to_handle_iter == thread_id_to_handle_.end()) {
            main_process_name_ = leaked_str;
            main_process_id_ = id;
            return;
        }
        thread_handle_to_interned_name_[id_to_handle_iter->second] = leaked_str;
    }

    const char *ThreadNameRegistry::get_name(PlatformThreadId id) {
        turbo::SpinLockHolder locked(&lock_);

        if (id == main_process_id_) {
            return main_process_name_->c_str();
        }

        auto id_to_handle_iter =
                thread_id_to_handle_.find(id);
        if (id_to_handle_iter == thread_id_to_handle_.end()) {
            return name_to_interned_name_[kDefaultName]->c_str();
        }

        auto handle_to_name_iter =
                thread_handle_to_interned_name_.find(id_to_handle_iter->second);
        return handle_to_name_iter->second->c_str();
    }

    void ThreadNameRegistry::remove_name(PlatformThreadHandle::Handle handle,
                                         PlatformThreadId id) {
        turbo::SpinLockHolder locked(&lock_);
        auto handle_to_name_iter = thread_handle_to_interned_name_.find(handle);

        TDLOG_CHECK(handle_to_name_iter != thread_handle_to_interned_name_.end());
        thread_handle_to_interned_name_.erase(handle_to_name_iter);

        auto id_to_handle_iter = thread_id_to_handle_.find(id);
        TDLOG_CHECK((id_to_handle_iter != thread_id_to_handle_.end()));
        // The given |id| may have been re-used by the system. Make sure the
        // mapping points to the provided |handle| before removal.
        if (id_to_handle_iter->second != handle) {
            return;
        }

        thread_id_to_handle_.erase(id_to_handle_iter);
    }

}  // namespace turbo
