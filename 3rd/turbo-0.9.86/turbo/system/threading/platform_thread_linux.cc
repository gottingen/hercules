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
#if defined(TURBO_PLATFORM_LINUX)
#include "turbo/base/assume.h"
#include "turbo/log/logging.h"
#include "turbo/system/threading/thread_name_registry.h"
#if !defined(TURBO_PLATFORM_NACL)
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#endif  // !defined(TURBO_PLATFORM_NACL)

namespace turbo {

    namespace {

        int ThreadNiceValue(ThreadPriority priority) {
            switch (priority) {
                case kThreadPriority_RealtimeAudio:
                    return -10;
                case kThreadPriority_Background:
                    return 10;
                case kThreadPriority_Normal:
                    return 0;
                case kThreadPriority_Display:
                    return -6;
                default:
                    turbo::assume_unreachable();
                    return 0;
            }
        }

    }  // namespace

// NOTE(gejun): PR_SET_NAME was added in 2.6.9, should be working on most of
// our machines, but missing from our linux headers.
#if !defined(PR_SET_NAME)
#define PR_SET_NAME 15
#endif

    // static
    void PlatformThread::set_name_internal(const char* name) {
        ThreadNameRegistry::get_instance()->set_name(current_id(), name);

#if !defined(TURBO_PLATFORM_NACL)
        // On linux we can get the thread names to show up in the debugger by setting
        // the process name for the LWP.  We don't want to do this for the main
        // thread because that would rename the process, causing tools like killall
        // to stop working.
        if (PlatformThread::current_id() == getpid())
            return;

        // http://0pointer.de/blog/projects/name-your-threads.html
        // Set the name for the LWP (which gets truncated to 15 characters).
        // Note that glibc also has a 'pthread_setname_np' api, but it may not be
        // available everywhere and it's only benefit over using prctl directly is
        // that it can set the name of threads other than the current thread.
        int err = prctl(PR_SET_NAME, name);
        // We expect EPERM failures in sandboxed processes, just ignore those.
        if (err < 0 && errno != EPERM) {
            TLOG_ERROR("prctl(PR_SET_NAME) failed with error {}", errno);
        }
#endif  //  !defined(TURBO_PLATFORM_NACL)
    }

// static
    void PlatformThread::set_thread_priority(PlatformThreadHandle handle,
                                           ThreadPriority priority) {
#if !defined(TURBO_PLATFORM_NACL)
        if (priority == kThreadPriority_RealtimeAudio) {
            const struct sched_param kRealTimePrio = { 8 };
            if (pthread_setschedparam(pthread_self(), SCHED_RR, &kRealTimePrio) == 0) {
                // Got real time priority, no need to set nice level.
                return;
            }
        }

        // setpriority(2) will set a thread's priority if it is passed a tid as
        // the 'process identifier', not affecting the rest of the threads in the
        // process. Setting this priority will only succeed if the user has been
        // granted permission to adjust nice values on the system.
        TDLOG_CHECK_NE(handle.id_, kInvalidThreadId);
        const int kNiceSetting = ThreadNiceValue(priority);
        if (setpriority(PRIO_PROCESS, handle.id_, kNiceSetting)) {
            TLOG_ERROR("Failed to set nice value of thread ({}) to {}",
                       handle.id_, kNiceSetting);
        }
#endif  //  !defined(TURBO_PLATFORM_NACL)
    }

    void InitThreading() {}

    void InitOnThread() {}

    void TerminateOnThread() {}

    int PlatformThread::set_affinity(PlatformThreadHandle thread_handle, std::vector<int> affinity) {
        if (affinity.empty()) {
            return -1;
        }
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (auto&& e : affinity) {
            CPU_SET(e, &cpuset);
        }
        return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
}  // namespace turbo
#endif  // defined(TURBO_PLATFORM_LINUX)