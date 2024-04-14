// Copyright 2019 The Turbo Authors.
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
#ifndef TURBO_BASE_STATUS_PAYLOAD_PRINTER_H_
#define TURBO_BASE_STATUS_PAYLOAD_PRINTER_H_

#include <string>
#include <optional>

#include "turbo/strings/cord.h"
#include "turbo/strings/string_view.h"

namespace turbo::status_internal {

    // By default, `Status::ToString` and `operator<<(Status)` print a payload by
    // dumping the type URL and the raw bytes. To help debugging, we provide an
    // extension point, which is a global printer function that can be set by users
    // to specify how to print payloads. The function takes the type URL and the
    // payload as input, and should return a valid human-readable string on success
    // or `std::nullopt` on failure (in which case it falls back to the default
    // approach of printing the raw bytes).
    // NOTE: This is an internal API and the design is subject to change in the
    // future in a non-backward-compatible way. Since it's only meant for debugging
    // purpose, you should not rely on it in any critical logic.
    using StatusPayloadPrinter = std::optional<std::string> (*)(std::string_view,
                                                                const turbo::Cord &);

    // Sets the global payload printer. Only one printer should be set per process.
    // If multiple printers are set, it's undefined which one will be used.
    void SetStatusPayloadPrinter(StatusPayloadPrinter);

    // Returns the global payload printer if previously set, otherwise `nullptr`.
    StatusPayloadPrinter GetStatusPayloadPrinter();

}  // namespace turbo::status_internal

#endif  // TURBO_BASE_STATUS_PAYLOAD_PRINTER_H_
