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
#include "turbo/system/threading.h"
#include "turbo/system/threading/platform_thread_mac.cc"
#include "turbo/system/threading/platform_thread_posix.cc"
#include "turbo/system/threading/platform_thread_linux.cc"
#include "turbo/system/threading/platform_thread_freebsd.cc"

namespace turbo {
    once_flag PlatformThread::sigaction_flag;

    int PlatformThread::kill_thread(pthread_t handle, int signo, signal_handler handler) {
        turbo::call_once(sigaction_flag,[signo, handler](){
            register_sigurg(signo, handler);
        });
        return pthread_kill(handle, SIGURG);
    }

    void PlatformThread::do_nothing_handler(int signo) {
        TURBO_UNUSED(signo);
    }

    void PlatformThread::register_sigurg(int signo, signal_handler handler) {
        signal(signo, handler);
    }

    int PlatformThread::set_current_affinity(std::vector<int> affinity) {
        return set_affinity(current_handle(), affinity);
    }
}  // namespace turbo
