// Copyright 2020 The Turbo Authors.
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
#include "turbo/status/result_status.h"

#include <utility>
#include "turbo/base/internal/raw_logging.h"
#include "turbo/status/status.h"

namespace turbo {

    BadResultStatusAccess::BadResultStatusAccess(turbo::Status status)
            : status_(std::move(status)) {}

    BadResultStatusAccess::BadResultStatusAccess(const BadResultStatusAccess &other)
            : status_(other.status_) {}

    BadResultStatusAccess &BadResultStatusAccess::operator=(
            const BadResultStatusAccess &other) {
        // Ensure assignment is correct regardless of whether this->InitWhat() has
        // already been called.
        other.InitWhat();
        status_ = other.status_;
        what_ = other.what_;
        return *this;
    }

    BadResultStatusAccess &BadResultStatusAccess::operator=(BadResultStatusAccess &&other) noexcept {
        // Ensure assignment is correct regardless of whether this->InitWhat() has
        // already been called.
        other.InitWhat();
        status_ = std::move(other.status_);
        what_ = std::move(other.what_);
        return *this;
    }

    BadResultStatusAccess::BadResultStatusAccess(BadResultStatusAccess &&other) noexcept
            : status_(std::move(other.status_)) {}

    const char *BadResultStatusAccess::what() const noexcept {
        InitWhat();
        return what_.c_str();
    }

    const turbo::Status &BadResultStatusAccess::status() const { return status_; }

    void BadResultStatusAccess::InitWhat() const {
        std::call_once(init_what_, [this] {
            what_ = turbo::format("Bad ResultStatus access: {}", status_.to_string());
        });
    }

    namespace result_status_internal {

        void Helper::HandleInvalidStatusCtorArg(turbo::Status *status) {
            const char *kMessage =
                    "An OK status is not a valid constructor argument to ResultStatus<T>";
#ifdef NDEBUG
            TURBO_INTERNAL_LOG(ERROR, kMessage);
#else
            TURBO_INTERNAL_LOG(FATAL, kMessage);
#endif
            // In optimized builds, we will fall back to internal_error.
            *status = turbo::internal_error(kMessage);
        }

        void Helper::Crash(const turbo::Status &status) {
            TURBO_INTERNAL_LOG(
                    FATAL,
                    turbo::format("Attempting to fetch value instead of handling error {}",
                                  status.to_string()));
        }

        void ThrowBadResultStatusAccess(turbo::Status status) {
#ifdef TURBO_HAVE_EXCEPTIONS
            throw turbo::BadResultStatusAccess(std::move(status));
#else
            TURBO_INTERNAL_LOG(
                FATAL,
                turbo::format("Attempting to fetch value instead of handling error {}",
                             status.ToString()));
            std::abort();
#endif
        }

    }  // namespace result_status_internal
}  // namespace turbo
