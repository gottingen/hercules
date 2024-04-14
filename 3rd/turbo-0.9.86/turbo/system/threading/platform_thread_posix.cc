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
#include "turbo/platform/port.h"

#if defined(TURBO_PLATFORM_POSIX)

#include "turbo/times/clock.h"
#include "turbo/system/threading/thread_name_registry.h"

#if defined(TURBO_PLATFORM_OSX)
#include <sys/resource.h>
#include <algorithm>
#endif

#if defined(TURBO_PLATFORM_LINUX)

#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#endif

namespace turbo {

    void InitThreading();

    void InitOnThread();

    void TerminateOnThread();

    PlatformThreadId PlatformThread::current_id() {
        // Pthreads doesn't have the concept of a thread ID, so we have to reach down
        // into the kernel.
#if defined(TURBO_PLATFORM_OSX)
        return pthread_mach_thread_np(pthread_self());
#elif defined(TURBO_PLATFORM_LINUX)
        return syscall(__NR_gettid);
#elif defined(TURBO_PLATFORM_ANDROID)
        return pthread_self();
#elif defined(TURBO_PLATFORM_NACL) && defined(__GLIBC__)
        return pthread_self();
#elif defined(TURBO_PLATFORM_NACL) && !defined(__GLIBC__)
        // Pointers are 32-bits in NaCl.
  return reinterpret_cast<int32_t>(pthread_self());
#elif defined(TURBO_PLATFORM_POSIX)
        return reinterpret_cast<int64_t>(pthread_self());
#endif
    }

    PlatformThreadRef PlatformThread::current_ref() {
        return PlatformThreadRef(pthread_self());
    }


    PlatformThreadHandle PlatformThread::current_handle() {
        return PlatformThreadHandle(pthread_self(), current_id());
    }

    void PlatformThread::yield_current_thread() {
        sched_yield();
    }

    void PlatformThread::sleep_for(turbo::Duration duration) {
        turbo::sleep_for(duration);
    }

    void PlatformThread::sleep_until(turbo::Time deadline) {
        turbo::sleep_until(deadline);
    }


    const char *PlatformThread::get_name() {
        return ThreadNameRegistry::get_instance()->get_name(current_id());
    }

    void PlatformThread::join(PlatformThreadHandle thread_handle) {
        ::pthread_join(thread_handle.platform_handle(), nullptr);
    }

}  // namespace turbo
#endif  // defined(TURBO_PLATFORM_POSIX)