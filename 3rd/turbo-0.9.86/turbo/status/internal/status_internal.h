// Copyright 2022 The Turbo Authors.
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
#ifndef TURBO_BASE_INTERNAL_STATUS_INTERNAL_H_
#define TURBO_BASE_INTERNAL_STATUS_INTERNAL_H_

#include <memory>
#include <string>
#include <utility>

#include "turbo/container/inlined_vector.h"
#include "turbo/platform/port.h"
#include "turbo/strings/cord.h"

// Disabled for SWIG as it doesn't parse attributes correctly.
namespace turbo {

    /// do not mark this class nodiscard
    /// mark it as discard if you want to, during
    /// the function call, the return value can be ignored
    ///  Example:
    ///  [[nodiscard]] Status DoSomething();
    ///  void Test() {
    ///    DoSomething();  // This will trigger a warning.
    ///    [[maybe_unused]] Status s = DoSomething();  // This is OK.
    ///  }
    class Status;
}  // namespace turbo

namespace turbo {

    using StatusCode = int;

    namespace status_internal {

        // Container for status payloads.
        struct Payload {
            std::string type_url;
            turbo::Cord payload;
        };

        using Payloads = turbo::InlinedVector<Payload, 1>;

        // Reference-counted representation of Status data.
        struct StatusRep {
            StatusRep(unsigned short int index_arg, turbo::StatusCode code_arg, std::string_view message_arg,
                      std::unique_ptr<status_internal::Payloads> payloads_arg)
                    : ref(int32_t{1}),
                      code(code_arg),
                      index(index_arg),
                      message(message_arg),
                      payloads(std::move(payloads_arg)) {}

            std::atomic<int32_t> ref;
            turbo::StatusCode code;
            unsigned short int index;
            std::string message;
            std::unique_ptr<status_internal::Payloads> payloads;
        };

        turbo::StatusCode MapToLocalCode(int value);

        // Returns a pointer to a newly-allocated string with the given `prefix`,
        // suitable for output as an error message in assertion/`TURBO_CHECK()` failures.
        //
        // This is an internal implementation detail for Turbo logging.
        std::string *MakeCheckFailString(const turbo::Status *status,
                                         const char *prefix);

    }  // namespace status_internal

}  // namespace turbo

#endif  // TURBO_BASE_INTERNAL_STATUS_INTERNAL_H_
