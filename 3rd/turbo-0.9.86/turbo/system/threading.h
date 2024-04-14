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

#ifndef TURBO_SYSTEM_THREADING_H_
#define TURBO_SYSTEM_THREADING_H_

#include "turbo/platform/port.h"
#include "turbo/times/time.h"
#include "turbo/format/format.h"
#include "turbo/concurrent/call_once.h"
#include <csignal>
#include <vector>

namespace turbo {


#if defined(TURBO_PLATFORM_WINDOWS)
    using PlatformThreadId = DWORD;
#else
    using PlatformThreadId = pid_t;
#endif

    // Used for thread checking and debugging.
    // Meant to be as fast as possible.
    // These are produced by PlatformThread::CurrentRef(), and used to later
    // check if we are on the same thread or not by using ==. These are safe
    // to copy between threads, but can't be copied to another process as they
    // have no meaning there. Also, the internal identifier can be re-used
    // after a thread dies, so a PlatformThreadRef cannot be reliably used
    // to distinguish a new thread from an old, dead thread.
    class PlatformThreadRef {
    public:
#if defined(TURBO_PLATFORM_WINDOWS)
        using RefType = DWORD;
#elif defined(TURBO_PLATFORM_POSIX)
        using RefType = pthread_t;
#endif

        PlatformThreadRef()
                : id_(0) {
        }

        explicit PlatformThreadRef(RefType id)
                : id_(id) {
        }

        bool operator==(PlatformThreadRef other) const {
            return id_ == other.id_;
        }

        bool is_null() const {
            return id_ == 0;
        }

    private:
        RefType id_;
    };


    // Used to operate on threads.
    class PlatformThreadHandle {
    public:
#if defined(TURBO_PLATFORM_WINDOWS)
        typedef void* Handle;
#elif defined(TURBO_PLATFORM_POSIX)
        typedef pthread_t Handle;
#endif

        PlatformThreadHandle()
                : handle_(0),
                  id_(0) {
        }

        explicit PlatformThreadHandle(Handle handle)
                : handle_(handle),
                  id_(0) {
        }

        PlatformThreadHandle(Handle handle,
                             PlatformThreadId id)
                : handle_(handle),
                  id_(id) {
        }

        bool is_equal(const PlatformThreadHandle &other) const {
            return handle_ == other.handle_;
        }

        bool is_null() const {
            return !handle_;
        }

        Handle platform_handle() const {
            return handle_;
        }

    private:
        friend class PlatformThread;

        Handle handle_;
        PlatformThreadId id_;
    };

    static constexpr PlatformThreadId kInvalidThreadId(0);

    // Valid values for SetThreadPriority()
    enum ThreadPriority {
        kThreadPriority_Normal,
        // Suitable for low-latency, glitch-resistant audio.
        kThreadPriority_RealtimeAudio,
        // Suitable for threads which generate data for the display (at ~60Hz).
        kThreadPriority_Display,
        // Suitable for threads that shouldn't disrupt high priority work.
        kThreadPriority_Background
    };

    class TURBO_DLL PlatformThread {
    public:
        using signal_handler = void (*)(int);

        // Implement this interface to run code on a background thread.  Your
        // ThreadMain method will be called on the newly created thread.
        class TURBO_DLL Delegate {
        public:
            virtual void ThreadMain() = 0;

        protected:
            virtual ~Delegate() {}
        };

        // Gets the current thread id, which may be useful for logging purposes.
        static PlatformThreadId current_id();

        // Gets the current thread reference, which can be used to check if
        // we're on the right thread quickly.
        static PlatformThreadRef current_ref();

        // Get the current handle.
        static PlatformThreadHandle current_handle();

        // Yield the current thread so another thread can be scheduled.
        static void yield_current_thread();

        // Sleeps for the specified duration.
        static void sleep_for(turbo::Duration duration);

        // Sleeps for the specified duration.
        static void sleep_until(turbo::Time deadline);

        // Sets the thread name visible to debuggers/tools. This has no effect
        // otherwise. This name pointer is not copied internally. Thus, it must stay
        // valid until the thread ends.
        template<typename ...Args>
        static void set_name(const std::string &fmt, Args... args);

        static void set_name(const std::string &name);

        // Gets the thread name, if previously set by SetName.
        static const char *get_name();

        static void set_thread_priority(PlatformThreadHandle handle, ThreadPriority priority);

        static int
        kill_thread(pthread_t handle, int signo = SIGURG, signal_handler handler = do_nothing_handler);

        static void join(PlatformThreadHandle thread_handle);

        static int set_affinity(PlatformThreadHandle thread_handle, std::vector<int> affinity);

        static int set_current_affinity(std::vector<int> affinity);
    private:

        static once_flag sigaction_flag;

        static void do_nothing_handler(int signo);

        static void register_sigurg(int signo, signal_handler handler);

        static void set_name_internal(const char *name);
        // nolint
        TURBO_NON_COPYABLE(PlatformThread);
    };

    template<typename ...Args>
    inline void PlatformThread::set_name(const std::string &fmt, Args... args) {
        set_name_internal(turbo::format(fmt, args...).c_str());
    }

    inline void PlatformThread::set_name(const std::string &name) {
        set_name_internal(name.c_str());
    }

}  // namespace turbo

#endif  // TURBO_SYSTEM_THREADING_H_
