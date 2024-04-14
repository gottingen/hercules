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
// Created by jeff on 24-1-6.
//

#ifndef TURBO_PLATFORM_POLL_H_
#define TURBO_PLATFORM_POLL_H_

#include "turbo/platform/port.h"

#ifdef TURBO_PLATFORM_LINUX

#include <sys/epoll.h>
#include <pthread.h>

#elif defined(TURBO_PLATFORM_OSX)
#include <sys/cdefs.h>
#include <stdint.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <pthread.h>
#include <sys/types.h>                           // struct kevent
#include <sys/event.h>                           // kevent(), kqueue()
#elif defined(TURBO_PLATFORM_WINDOWS)

#else
#error "The platform does not support epoll-like APIs"
#endif

#endif  // TURBO_PLATFORM_POLL_H_
