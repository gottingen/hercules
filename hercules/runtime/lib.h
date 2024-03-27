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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <unwind.h>
#include "hercules/runtime/port.h"
#include "hercules/runtime/hs_str.h"

struct hs_time_t {
    int16_t year;
    int16_t yday;
    int8_t sec;
    int8_t min;
    int8_t hour;
    int8_t mday;
    int8_t mon;
    int8_t wday;
    int8_t isdst;
};

HS_FUNC int hs_flags;

HS_FUNC void hs_init(int flags);

HS_FUNC bool hs_is_macos();
HS_FUNC hs_int_t hs_pid();
HS_FUNC hs_int_t hs_time();
HS_FUNC hs_int_t hs_time_monotonic();
HS_FUNC hs_int_t hs_time_highres();
HS_FUNC bool hs_localtime(hs_int_t secs, hs_time_t *output);
HS_FUNC bool hs_gmtime(hs_int_t secs, hs_time_t *output);
HS_FUNC hs_int_t hs_mktime(hs_time_t *time);
HS_FUNC void hs_sleep(double secs);
HS_FUNC char **hs_env();
HS_FUNC void hs_assert_failed(hs_str_t file, hs_int_t line);

HS_FUNC void *hs_alloc(size_t n);
HS_FUNC void *hs_alloc_atomic(size_t n);
HS_FUNC void *hs_alloc_uncollectable(size_t n);
HS_FUNC void *hs_alloc_atomic_uncollectable(size_t n);
HS_FUNC void *hs_calloc(size_t m, size_t n);
HS_FUNC void *hs_calloc_atomic(size_t m, size_t n);
HS_FUNC void *hs_realloc(void *p, size_t newsize, size_t oldsize);
HS_FUNC void hs_free(void *p);
HS_FUNC void hs_register_finalizer(void *p, void (*f)(void *obj, void *data));

HS_FUNC void hs_gc_add_roots(void *start, void *end);
HS_FUNC void hs_gc_remove_roots(void *start, void *end);
HS_FUNC void hs_gc_clear_roots();
HS_FUNC void hs_gc_exclude_static_roots(void *start, void *end);

HS_FUNC void *hs_alloc_exc(int type, void *obj);
HS_FUNC void hs_throw(void *exc);
HS_FUNC _Unwind_Reason_Code hs_personality(int version, _Unwind_Action actions,
                                             uint64_t exceptionClass,
                                             _Unwind_Exception *exceptionObject,
                                             _Unwind_Context *context);
HS_FUNC int64_t hs_exc_offset();
HS_FUNC uint64_t hs_exc_class();

HS_FUNC hs_str_t hs_str_int(hs_int_t n, hs_str_t format, bool *error);
HS_FUNC hs_str_t hs_str_uint(hs_int_t n, hs_str_t format, bool *error);
HS_FUNC hs_str_t hs_str_float(double f, hs_str_t format, bool *error);
HS_FUNC hs_str_t hs_str_ptr(void *p, hs_str_t format, bool *error);
HS_FUNC hs_str_t hs_str_str(hs_str_t s, hs_str_t format, bool *error);

HS_FUNC void *hs_stdin();
HS_FUNC void *hs_stdout();
HS_FUNC void *hs_stderr();

HS_FUNC void hs_print(hs_str_t str);
HS_FUNC void hs_print_full(hs_str_t str, FILE *fo);

HS_FUNC void *hs_lock_new();
HS_FUNC bool hs_lock_acquire(void *lock, bool block, double timeout);
HS_FUNC void hs_lock_release(void *lock);
HS_FUNC void *hs_rlock_new();
HS_FUNC bool hs_rlock_acquire(void *lock, bool block, double timeout);
HS_FUNC void hs_rlock_release(void *lock);

namespace hercules::runtime {
    class JITError : public std::runtime_error {
    private:
        std::string output;
        std::string type;
        std::string file;
        int line;
        int col;
        std::vector<uintptr_t> backtrace;

    public:
        JITError(const std::string &output, const std::string &what, const std::string &type,
                 const std::string &file, int line, int col,
                 std::vector<uintptr_t> backtrace = {})
                : std::runtime_error(what), output(output), type(type), file(file), line(line),
                  col(col), backtrace(std::move(backtrace)) {}

        std::string getOutput() const { return output; }

        std::string getType() const { return type; }

        std::string getFile() const { return file; }

        int getLine() const { return line; }

        int getCol() const { return col; }

        std::vector<uintptr_t> getBacktrace() const { return backtrace; }
    };

    std::string makeBacktraceFrameString(uintptr_t pc, const std::string &func = "",
                                         const std::string &file = "", int line = 0,
                                         int col = 0);

    std::string getCapturedOutput();

    void setJITErrorCallback(std::function<void(const JITError &)> callback);
} // namespace hercules::runtime
