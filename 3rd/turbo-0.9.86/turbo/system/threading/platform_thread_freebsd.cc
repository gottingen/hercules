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


#include <errno.h>
#include <sched.h>
#include "turbo/system/threading.h"
#include "turbo/platform/port.h"
#if defined(TURBO_PLATFORM_FREEBSD)

#include "turbo/base/assume.h"
#include "turbo/log/logging.h"
#include "turbo/system/threading/thread_name_registry.h"

#if !defined(TURBO_PLATFORM_NACL)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
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
    } // namespace

    void PlatformThread::set_name_internal(const char* name) {
        ThreadNameRegistry::get_instance()->set_name(current_id(), name);

#if !defined(TURBO_PLATFORM_NACL)
        // On FreeBSD we can get the thread names to show up in the debugger by
        // setting the process name for the LWP.  We don't want to do this for the
        // main thread because that would rename the process, causing tools like
        // killall to stop working.
        if (PlatformThread::current_id() == getpid())
            return;
        setproctitle("%s", name);
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


}  // namespace turbo
#endif  // defined(TURBO_PLATFORM_FREEBSD)