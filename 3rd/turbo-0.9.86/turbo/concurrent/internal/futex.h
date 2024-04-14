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
//
// Created by jeff on 23-12-16.
//

#ifndef TURBO_CONCURRENT_INTERNAL_FUTEX_H_
#define TURBO_CONCURRENT_INTERNAL_FUTEX_H_

#include "turbo/platform/port.h"
#include <unistd.h>
#include "turbo/times/time.h"

#ifdef TURBO_PLATFORM_LINUX

#include <linux/futex.h>
#include <sys/syscall.h>

#endif  // TURBO_PLATFORM_LINUX

namespace turbo::concurrent_internal {

#if  defined(TURBO_PLATFORM_LINUX)

#ifndef FUTEX_PRIVATE_FLAG
#define FUTEX_PRIVATE_FLAG 128
#endif

    inline int futex_wait_private(
            void *addr1, int expected, const timespec *timeout) {
        return syscall(SYS_futex, addr1, (FUTEX_WAIT | FUTEX_PRIVATE_FLAG),
                       expected, timeout, NULL, 0);
    }

    inline int futex_wake_private(void *addr1, int nwake) {
        return syscall(SYS_futex, addr1, (FUTEX_WAKE | FUTEX_PRIVATE_FLAG),
                       nwake, NULL, NULL, 0);
    }

#else

    int futex_wait_private(void* addr1, int expected, const timespec* timeout);

    int futex_wake_private(void* addr1, int nwake);

#endif  //

}  // namespace turbo::concurrent_internal

#endif  // TURBO_CONCURRENT_INTERNAL_FUTEX_H_
