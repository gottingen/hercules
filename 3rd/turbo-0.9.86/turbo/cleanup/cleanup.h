// Copyright 2021 The Turbo Authors.
//
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
// -----------------------------------------------------------------------------
// File: cleanup.h
// -----------------------------------------------------------------------------
//
// `turbo::Cleanup` implements the scope guard idiom, invoking the contained
// callback's `operator()() &&` on scope exit.
//
// Example:
//
// ```
//   turbo::Status CopyGoodData(const char* source_path, const char* sink_path) {
//     FILE* source_file = fopen(source_path, "r");
//     if (source_file == nullptr) {
//       return turbo::not_found_error("No source file");  // No cleanups execute
//     }
//
//     // C++17 style cleanup using class template argument deduction
//     turbo::Cleanup source_closer = [source_file] { fclose(source_file); };
//
//     FILE* sink_file = fopen(sink_path, "w");
//     if (sink_file == nullptr) {
//       return turbo::not_found_error("No sink file");  // First cleanup executes
//     }
//
//     // C++11 style cleanup using the factory function
//     auto sink_closer = turbo::MakeCleanup([sink_file] { fclose(sink_file); });
//
//     Data data;
//     while (ReadData(source_file, &data)) {
//       if (!data.IsGood()) {
//         turbo::Status result = turbo::failed_precondition_error("Read bad data");
//         return result;  // Both cleanups execute
//       }
//       SaveData(sink_file, &data);
//     }
//
//     return turbo::ok_status();  // Both cleanups execute
//   }
// ```
//
// Methods:
//
// `std::move(cleanup).Cancel()` will prevent the callback from executing.
//
// `std::move(cleanup).Invoke()` will execute the callback early, before
// destruction, and prevent the callback from executing in the destructor.
//
// Usage:
//
// `turbo::Cleanup` is not an interface type. It is only intended to be used
// within the body of a function. It is not a value type and instead models a
// control flow construct. Check out `defer` in Golang for something similar.

#ifndef TURBO_CLEANUP_CLEANUP_H_
#define TURBO_CLEANUP_CLEANUP_H_

#include <utility>

#include "turbo/cleanup/internal/cleanup.h"
#include "turbo/platform/port.h"

namespace turbo {

    template<typename Arg, typename Callback = void()>
    class TURBO_MUST_USE_RESULT Cleanup final {
        static_assert(cleanup_internal::WasDeduced<Arg>(),
                      "Explicit template parameters are not supported.");

        static_assert(cleanup_internal::ReturnsVoid<Callback>(),
                      "Callbacks that return values are not supported.");

    public:
        Cleanup(Callback callback) : storage_(std::move(callback)) {}  // NOLINT

        Cleanup(Cleanup &&other) = default;

        void Cancel() &&{
            TURBO_HARDENING_ASSERT(storage_.IsCallbackEngaged());
            storage_.DestroyCallback();
        }

        void Invoke() &&{
            TURBO_HARDENING_ASSERT(storage_.IsCallbackEngaged());
            storage_.InvokeCallback();
            storage_.DestroyCallback();
        }

        ~Cleanup() {
            if (storage_.IsCallbackEngaged()) {
                storage_.InvokeCallback();
                storage_.DestroyCallback();
            }
        }

    private:
        cleanup_internal::Storage<Callback> storage_;
    };

// `turbo::Cleanup c = /* callback */;`
//
// C++17 type deduction API for creating an instance of `turbo::Cleanup`
#if defined(TURBO_HAVE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
    template<typename Callback>
    Cleanup(Callback callback) -> Cleanup<cleanup_internal::Tag, Callback>;
#endif  // defined(TURBO_HAVE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)

    // `auto c = turbo::MakeCleanup(/* callback */);`
    //
    // C++11 type deduction API for creating an instance of `turbo::Cleanup`
    template<typename... Args, typename Callback>
    turbo::Cleanup<cleanup_internal::Tag, Callback> MakeCleanup(Callback callback) {
        static_assert(cleanup_internal::WasDeduced<cleanup_internal::Tag, Args...>(),
                      "Explicit template parameters are not supported.");

        static_assert(cleanup_internal::ReturnsVoid<Callback>(),
                      "Callbacks that return values are not supported.");

        return {std::move(callback)};
    }

}  // namespace turbo

#endif  // TURBO_CLEANUP_CLEANUP_H_
