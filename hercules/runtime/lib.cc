// Copyright 2024 The EA Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#include <cassert>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <collie/strings/format.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <unwind.h>
#include <vector>

#define GC_THREADS

#include "hercules/runtime/lib.h"
#include <gc.h>

/*
 * General
 */

// OpenMP patch with GC callbacks
typedef int (*gc_setup_callback)(GC_stack_base *);

typedef void (*gc_roots_callback)(void *, void *);

extern "C" void __kmpc_set_gc_callbacks(gc_setup_callback get_stack_base,
                                        gc_setup_callback register_thread,
                                        gc_roots_callback add_roots,
                                        gc_roots_callback del_roots);

void hs_exc_init();

#ifdef HERCULES_GPU
void hs_nvptx_init();
#endif

int hs_flags;

HS_FUNC void hs_init(int flags) {
    GC_INIT();
    GC_set_warn_proc(GC_ignore_warn_proc);
    GC_allow_register_threads();
    __kmpc_set_gc_callbacks(GC_get_stack_base, (gc_setup_callback) GC_register_my_thread,
                            GC_add_roots, GC_remove_roots);
    hs_exc_init();
#ifdef HERCULES_GPU
    hs_nvptx_init();
#endif
    hs_flags = flags;
}

HS_FUNC bool hs_is_macos() {
#ifdef __APPLE__
    return true;
#else
    return false;
#endif
}

HS_FUNC hs_int_t hs_pid() { return (hs_int_t) getpid(); }

HS_FUNC hs_int_t hs_time() {
    auto duration = std::chrono::system_clock::now().time_since_epoch();
    hs_int_t nanos =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return nanos;
}

HS_FUNC hs_int_t hs_time_monotonic() {
    auto duration = std::chrono::steady_clock::now().time_since_epoch();
    hs_int_t nanos =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return nanos;
}

HS_FUNC hs_int_t hs_time_highres() {
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch();
    hs_int_t nanos =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    return nanos;
}

static void copy_time_c_to_seq(struct tm *x, hs_time_t *output) {
    output->year = x->tm_year;
    output->yday = x->tm_yday;
    output->sec = x->tm_sec;
    output->min = x->tm_min;
    output->hour = x->tm_hour;
    output->mday = x->tm_mday;
    output->mon = x->tm_mon;
    output->wday = x->tm_wday;
    output->isdst = x->tm_isdst;
}

static void copy_time_hs_to_c(hs_time_t *x, struct tm *output) {
    output->tm_year = x->year;
    output->tm_yday = x->yday;
    output->tm_sec = x->sec;
    output->tm_min = x->min;
    output->tm_hour = x->hour;
    output->tm_mday = x->mday;
    output->tm_mon = x->mon;
    output->tm_wday = x->wday;
    output->tm_isdst = x->isdst;
}

HS_FUNC bool hs_localtime(hs_int_t secs, hs_time_t *output) {
    struct tm result;
    time_t now = (secs >= 0 ? secs : time(nullptr));
    if (now == (time_t) -1 || !localtime_r(&now, &result))
        return false;
    copy_time_c_to_seq(&result, output);
    return true;
}

HS_FUNC bool hs_gmtime(hs_int_t secs, hs_time_t *output) {
    struct tm result;
    time_t now = (secs >= 0 ? secs : time(nullptr));
    if (now == (time_t) -1 || !gmtime_r(&now, &result))
        return false;
    copy_time_c_to_seq(&result, output);
    return true;
}

HS_FUNC hs_int_t hs_mktime(hs_time_t *time) {
    struct tm result;
    copy_time_hs_to_c(time, &result);
    return mktime(&result);
}

HS_FUNC void hs_sleep(double secs) {
    std::this_thread::sleep_for(std::chrono::duration<double, std::ratio<1>>(secs));
}

extern char **environ;
HS_FUNC char **hs_env() { return environ; }

/*
 * GC
 */
#define USE_STANDARD_MALLOC 0

HS_FUNC void *hs_alloc(size_t n) {
#if USE_STANDARD_MALLOC
    return malloc(n);
#else
    return GC_MALLOC(n);
#endif
}

HS_FUNC void *hs_alloc_atomic(size_t n) {
#if USE_STANDARD_MALLOC
    return malloc(n);
#else
    return GC_MALLOC_ATOMIC(n);
#endif
}

HS_FUNC void *hs_alloc_uncollectable(size_t n) {
#if USE_STANDARD_MALLOC
    return malloc(n);
#else
    return GC_MALLOC_UNCOLLECTABLE(n);
#endif
}

HS_FUNC void *hs_alloc_atomic_uncollectable(size_t n) {
#if USE_STANDARD_MALLOC
    return malloc(n);
#else
    return GC_MALLOC_ATOMIC_UNCOLLECTABLE(n);
#endif
}

HS_FUNC void *hs_calloc(size_t m, size_t n) {
#if USE_STANDARD_MALLOC
    return calloc(m, n);
#else
    size_t s = m * n;
    void *p = GC_MALLOC(s);
    memset(p, 0, s);
    return p;
#endif
}

HS_FUNC void *hs_calloc_atomic(size_t m, size_t n) {
#if USE_STANDARD_MALLOC
    return calloc(m, n);
#else
    size_t s = m * n;
    void *p = GC_MALLOC_ATOMIC(s);
    memset(p, 0, s);
    return p;
#endif
}

HS_FUNC void *hs_realloc(void *p, size_t newsize, size_t oldsize) {
#if USE_STANDARD_MALLOC
    return realloc(p, newsize);
#else
    return GC_REALLOC(p, newsize);
#endif
}

HS_FUNC void hs_free(void *p) {
#if USE_STANDARD_MALLOC
    free(p);
#else
    GC_FREE(p);
#endif
}

HS_FUNC void hs_register_finalizer(void *p, void (*f)(void *obj, void *data)) {
#if !USE_STANDARD_MALLOC
    GC_REGISTER_FINALIZER(p, f, nullptr, nullptr, nullptr);
#endif
}

HS_FUNC void hs_gc_add_roots(void *start, void *end) {
#if !USE_STANDARD_MALLOC
    GC_add_roots(start, end);
#endif
}

HS_FUNC void hs_gc_remove_roots(void *start, void *end) {
#if !USE_STANDARD_MALLOC
    GC_remove_roots(start, end);
#endif
}

HS_FUNC void hs_gc_clear_roots() {
#if !USE_STANDARD_MALLOC
    GC_clear_roots();
#endif
}

HS_FUNC void hs_gc_exclude_static_roots(void *start, void *end) {
#if !USE_STANDARD_MALLOC
    GC_exclude_static_roots(start, end);
#endif
}

/*
 * String conversion
 */
static hs_str_t string_conv(const std::string &s) {
    auto n = s.size();
    auto *p = (char *) hs_alloc_atomic(n);
    memcpy(p, s.data(), n);
    return {(hs_int_t) n, p};
}

template<typename T>
std::string default_format(T n) {
    return collie::format(FMT_STRING("{}"), n);
}

template<>
std::string default_format(double n) {
    return collie::format(FMT_STRING("{:g}"), n);
}

template<typename T>
hs_str_t fmt_conv(T n, hs_str_t format, bool *error) {
    *error = false;
    try {
        if (format.len == 0) {
            return string_conv(default_format(n));
        } else {
            std::string fstr(format.str, format.len);
            return string_conv(
                    collie::format(collie::runtime(collie::format(FMT_STRING("{{:{}}}"), fstr)), n));
        }
    } catch (const std::runtime_error &f) {
        *error = true;
        return string_conv(f.what());
    }
}

HS_FUNC hs_str_t hs_str_int(hs_int_t n, hs_str_t format, bool *error) {
    return fmt_conv<hs_int_t>(n, format, error);
}

HS_FUNC hs_str_t hs_str_uint(hs_int_t n, hs_str_t format, bool *error) {
    return fmt_conv<uint64_t>(n, format, error);
}

HS_FUNC hs_str_t hs_str_float(double f, hs_str_t format, bool *error) {
    return fmt_conv<double>(f, format, error);
}

HS_FUNC hs_str_t hs_str_ptr(void *p, hs_str_t format, bool *error) {
    return fmt_conv(collie::ptr(p), format, error);
}

HS_FUNC hs_str_t hs_str_str(hs_str_t s, hs_str_t format, bool *error) {
    std::string t(s.str, s.len);
    return fmt_conv(t, format, error);
}

/*
 * General I/O
 */

HS_FUNC hs_str_t hs_check_errno() {
    if (errno) {
        std::string msg = strerror(errno);
        auto *buf = (char *) hs_alloc_atomic(msg.size());
        memcpy(buf, msg.data(), msg.size());
        return {(hs_int_t) msg.size(), buf};
    }
    return {0, nullptr};
}

HS_FUNC void hs_print(hs_str_t str) { hs_print_full(str, stdout); }

static std::ostringstream capture;
static std::mutex captureLock;

HS_FUNC void hs_print_full(hs_str_t str, FILE *fo) {
    if ((hs_flags & HS_FLAG_CAPTURE_OUTPUT) && (fo == stdout || fo == stderr)) {
        captureLock.lock();
        capture.write(str.str, str.len);
        captureLock.unlock();
    } else {
        fwrite(str.str, 1, (size_t) str.len, fo);
    }
}

std::string hercules::runtime::getCapturedOutput() {
    std::string result = capture.str();
    capture.str("");
    return result;
}

HS_FUNC void *hs_stdin() { return stdin; }

HS_FUNC void *hs_stdout() { return stdout; }

HS_FUNC void *hs_stderr() { return stderr; }

/*
 * Threading
 */

HS_FUNC void *hs_lock_new() {
    return (void *) new(hs_alloc_atomic(sizeof(std::timed_mutex))) std::timed_mutex();
}

HS_FUNC bool hs_lock_acquire(void *lock, bool block, double timeout) {
    auto *m = (std::timed_mutex *) lock;
    if (timeout < 0.0) {
        if (block) {
            m->lock();
            return true;
        } else {
            return m->try_lock();
        }
    } else {
        return m->try_lock_for(std::chrono::duration<double>(timeout));
    }
}

HS_FUNC void hs_lock_release(void *lock) {
    auto *m = (std::timed_mutex *) lock;
    m->unlock();
}

HS_FUNC void *hs_rlock_new() {
    return (void *) new(hs_alloc_atomic(sizeof(std::recursive_timed_mutex)))
            std::recursive_timed_mutex();
}

HS_FUNC bool hs_rlock_acquire(void *lock, bool block, double timeout) {
    auto *m = (std::recursive_timed_mutex *) lock;
    if (timeout < 0.0) {
        if (block) {
            m->lock();
            return true;
        } else {
            return m->try_lock();
        }
    } else {
        return m->try_lock_for(std::chrono::duration<double>(timeout));
    }
}

HS_FUNC void hs_rlock_release(void *lock) {
    auto *m = (std::recursive_timed_mutex *) lock;
    m->unlock();
}
