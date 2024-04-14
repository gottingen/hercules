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
// Created by jeff on 23-12-6.
//

#ifndef TURBO_SYSTEM_ATEXIT_H_
#define TURBO_SYSTEM_ATEXIT_H_

namespace turbo {

    // |fn| or |fn(arg)| will be called at caller's exit. If caller is not a
    // thread, fn will be called at program termination. Calling sequence is LIFO:
    // last registered function will be called first. Duplication of functions
    // are not checked. This function is often used for releasing thread-local
    // resources declared with __thread which is much faster than
    // pthread_getspecific or boost::thread_specific_ptr.
    // Returns 0 on success, -1 otherwise and errno is set.
    int thread_atexit(void (*fn)());
    int thread_atexit(void (*fn)(void*), void* arg);

    // Remove registered function, matched functions will not be called.
    void thread_atexit_cancel(void (*fn)());
    void thread_atexit_cancel(void (*fn)(void*), void* arg);
}  // namespace turbo

#endif  // TURBO_SYSTEM_ATEXIT_H_
