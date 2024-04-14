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
#include "turbo/status/status.h"

#include <errno.h>

#include <cassert>
#include <utility>

#include "status_payload_printer.h"
#include "turbo/base/internal/raw_logging.h"
//#include "turbo/base/internal/strerror.h"
#include "turbo/status/error.h"
#include "turbo/debugging/stacktrace.h"
#include "turbo/debugging/symbolize.h"
#include "turbo/platform/port.h"
#include "turbo/strings/escaping.h"
#include "turbo/strings/str_split.h"

namespace turbo {


    std::string status_code_to_string(StatusCode code) {
        return std::string(terror(code));
    }

    namespace status_internal {

        static std::optional<size_t> FindPayloadIndexByUrl(
                const Payloads *payloads,
                std::string_view type_url) {
            if (payloads == nullptr)
                return std::nullopt;

            for (size_t i = 0; i < payloads->size(); ++i) {
                if ((*payloads)[i].type_url == type_url) return i;
            }

            return std::nullopt;
        }

        // Convert canonical code to a value known to this binary.
        turbo::StatusCode MapToLocalCode(int value) {
            turbo::StatusCode code = static_cast<turbo::StatusCode>(value);
            switch (code) {
                case turbo::kOk:
                case turbo::kCancelled:
                case turbo::kUnknown:
                case turbo::kInvalidArgument:
                case turbo::kDeadlineExceeded:
                case turbo::kNotFound:
                case turbo::kAlreadyExists:
                case turbo::kPermissionDenied:
                case turbo::kResourceExhausted:
                case turbo::kFailedPrecondition:
                case turbo::kAborted:
                case turbo::kOutOfRange:
                case turbo::kUnimplemented:
                case turbo::kInternal:
                case turbo::kUnavailable:
                case turbo::kDataLoss:
                case turbo::kUnauthenticated:
                    return code;
                default:
                    return turbo::kUnknown;
            }
        }
    }  // namespace status_internal

    std::optional<turbo::Cord> Status::get_payload(
            std::string_view type_url) const {
        const auto *payloads = GetPayloads();
        std::optional<size_t> index =
                status_internal::FindPayloadIndexByUrl(payloads, type_url);
        if (index.has_value())
            return (*payloads)[index.value()].payload;

        return std::nullopt;
    }

    void Status::set_payload(std::string_view type_url, turbo::Cord payload) {
        if (ok()) return;

        PrepareToModify();

        status_internal::StatusRep *rep = RepToPointer(rep_);
        if (!rep->payloads) {
            rep->payloads = std::make_unique<status_internal::Payloads>();
        }

        std::optional<size_t> index =
                status_internal::FindPayloadIndexByUrl(rep->payloads.get(), type_url);
        if (index.has_value()) {
            (*rep->payloads)[index.value()].payload = std::move(payload);
            return;
        }

        rep->payloads->push_back({std::string(type_url), std::move(payload)});
    }

    bool Status::erase_payload(std::string_view type_url) {
        std::optional<size_t> index =
                status_internal::FindPayloadIndexByUrl(GetPayloads(), type_url);
        if (index.has_value()) {
            PrepareToModify();
            GetPayloads()->erase(GetPayloads()->begin() + index.value());
            if (GetPayloads()->empty() && message().empty()) {
                // Special case: If this can be represented inlined, it MUST be
                // inlined (EqualsSlow depends on this behavior).
                StatusCode c = static_cast<StatusCode>(code());
                Unref(rep_);
                rep_ = CodeToInlinedRep(c);
            }
            return true;
        }

        return false;
    }

    void Status::for_each_payload(
            turbo::FunctionRef<void(std::string_view, const turbo::Cord &)> visitor)
    const {
        if (auto *payloads = GetPayloads()) {
            bool in_reverse =
                    payloads->size() > 1 && reinterpret_cast<uintptr_t>(payloads) % 13 > 6;

            for (size_t index = 0; index < payloads->size(); ++index) {
                const auto &elem =
                        (*payloads)[in_reverse ? payloads->size() - 1 - index : index];

#ifdef NDEBUG
                visitor(elem.type_url, elem.payload);
#else
                // In debug mode invalidate the type url to prevent users from relying on
                // this string lifetime.

                // NOLINTNEXTLINE intentional extra conversion to force temporary.
                visitor(std::string(elem.type_url), elem.payload);
#endif  // NDEBUG
            }
        }
    }

    const std::string *Status::EmptyString() {
        static union EmptyString {
            std::string str;

            ~EmptyString() {}
        } empty = {{}};
        return &empty.str;
    }


    const std::string *Status::MovedFromString() {
        static std::string *moved_from_string = new std::string(kMovedFromString);
        return moved_from_string;
    }

    void Status::UnrefNonInlined(uintptr_t rep) {
        status_internal::StatusRep *r = RepToPointer(rep);
        // Fast path: if ref==1, there is no need for a RefCountDec (since
        // this is the only reference and therefore no other thread is
        // allowed to be mucking with r).
        if (r->ref.load(std::memory_order_acquire) == 1 ||
            r->ref.fetch_sub(1, std::memory_order_acq_rel) - 1 == 0) {
            delete r;
        }
    }

    Status::Status(turbo::StatusCode code, std::string_view msg)
            : rep_(CodeToInlinedRep(code)) {
        if (code != turbo::kOk && !msg.empty()) {
            rep_ = PointerToRep(new status_internal::StatusRep(kTurboModuleIndex, code, msg, nullptr));
        }
    }

    Status::Status(unsigned short int index, turbo::StatusCode code, std::string_view msg)
            : rep_(CodeToInlinedRep(index, code)) {
        if (code != turbo::kOk && !msg.empty()) {
            rep_ = PointerToRep(new status_internal::StatusRep(index, code, msg, nullptr));
        }
    }

    /*
    template <typename... Args>
    Status::Status(unsigned short int module_index, turbo::StatusCode code, const FormatSpec<Args...>& format,
           const Args&... args) {

    }
    */
    int Status::code() const {
        if (IsInlined(rep_)) {
            return static_cast<int>(InlinedRepToCode(rep_));
        }
        status_internal::StatusRep *rep = RepToPointer(rep_);
        return static_cast<int>(rep->code);
    }

    unsigned short int Status::index() const {
        if (IsInlined(rep_)) {
            return static_cast<unsigned short int>(InlinedRepToIndex(rep_));
        }
        status_internal::StatusRep *rep = RepToPointer(rep_);
        return static_cast<unsigned short int>(rep->index);
    }

    turbo::StatusCode Status::map_code() const {
        return errno_to_status_code(code());
        //return code();
    }

    void Status::PrepareToModify() {
        TURBO_RAW_CHECK(!ok(), "PrepareToModify shouldn't be called on OK status.");
        if (IsInlined(rep_)) {
            rep_ = PointerToRep(new status_internal::StatusRep(static_cast<unsigned short int>(index()),
                                                               static_cast<turbo::StatusCode>(code()),
                                                               std::string_view(),
                                                               nullptr));
            return;
        }

        uintptr_t rep_i = rep_;
        status_internal::StatusRep *rep = RepToPointer(rep_);
        if (rep->ref.load(std::memory_order_acquire) != 1) {
            std::unique_ptr<status_internal::Payloads> payloads;
            if (rep->payloads) {
                payloads = std::make_unique<status_internal::Payloads>(*rep->payloads);
            }
            status_internal::StatusRep *const new_rep = new status_internal::StatusRep(
                    rep->index,
                    rep->code, message(), std::move(payloads));
            rep_ = PointerToRep(new_rep);
            UnrefNonInlined(rep_i);
        }
    }

    bool Status::EqualsSlow(const turbo::Status &a, const turbo::Status &b) {
        if (IsInlined(a.rep_) != IsInlined(b.rep_)) return false;
        if (a.message() != b.message()) return false;
        if (a.code() != b.code()) return false;
        if (a.GetPayloads() == b.GetPayloads()) return true;

        const status_internal::Payloads no_payloads;
        const status_internal::Payloads *larger_payloads =
                a.GetPayloads() ? a.GetPayloads() : &no_payloads;
        const status_internal::Payloads *smaller_payloads =
                b.GetPayloads() ? b.GetPayloads() : &no_payloads;
        if (larger_payloads->size() < smaller_payloads->size()) {
            std::swap(larger_payloads, smaller_payloads);
        }
        if ((larger_payloads->size() - smaller_payloads->size()) > 1) return false;
        // Payloads can be ordered differently, so we can't just compare payload
        // vectors.
        for (const auto &payload: *larger_payloads) {

            bool found = false;
            for (const auto &other_payload: *smaller_payloads) {
                if (payload.type_url == other_payload.type_url) {
                    if (payload.payload != other_payload.payload) {
                        return false;
                    }
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    std::string Status::ToStringSlow(StatusToStringMode mode) const {
        std::string text;
        const bool with_module = (mode & StatusToStringMode::kWithModule) ==
                                 StatusToStringMode::kWithModule;

        if (with_module) {
            turbo::format_append(&text, "{}::{}: {}", TurboModule(index()), status_code_to_string(code()), message());
        } else {
            turbo::format_append(&text, "{}: {}", status_code_to_string(code()), message());
        }

        const bool with_payload = (mode & StatusToStringMode::kWithPayload) ==
                                  StatusToStringMode::kWithPayload;

        if (with_payload) {
            status_internal::StatusPayloadPrinter printer =
                    status_internal::GetStatusPayloadPrinter();
            this->for_each_payload([&](std::string_view type_url,
                                     const turbo::Cord &payload) {
                std::optional<std::string> result;
                if (printer) result = printer(type_url, payload);
                turbo::format_append(
                        &text, " [{}='{}']", type_url,
                        result.has_value() ? *result : turbo::c_hex_encode(std::string(payload)));
            });
        }

        return text;
    }

    std::ostream &operator<<(std::ostream &os, const Status &x) {
        os << x.to_string(StatusToStringMode::kWithEverything);
        return os;
    }

    bool is_aborted(const Status &status) {
        return status.map_code() == turbo::kAborted;
    }

    bool is_already_exists(const Status &status) {
        return status.map_code() == turbo::kAlreadyExists;
    }

    bool is_cancelled(const Status &status) {
        return status.map_code() == turbo::kCancelled;
    }

    bool is_data_loss(const Status &status) {
        return status.map_code() == turbo::kDataLoss;
    }

    bool is_deadline_exceeded(const Status &status) {
        return status.map_code() == turbo::kDeadlineExceeded;
    }

    bool is_failed_precondition(const Status &status) {
        return status.map_code() == turbo::kFailedPrecondition;
    }

    bool is_internal(const Status &status) {
        return status.map_code() == turbo::kInternal;
    }

    bool is_invalid_argument(const Status &status) {
        return status.map_code() == turbo::kInvalidArgument;
    }

    bool is_not_found(const Status &status) {
        return status.map_code() == turbo::kNotFound;
    }

    bool is_out_of_range(const Status &status) {
        return status.map_code() == turbo::kOutOfRange;
    }

    bool is_permission_denied(const Status &status) {
        return status.map_code() == turbo::kPermissionDenied;
    }

    bool is_resource_exhausted(const Status &status) {
        return status.map_code() == turbo::kResourceExhausted;
    }

    bool is_unauthenticated(const Status &status) {
        return status.map_code() == turbo::kUnauthenticated;
    }

    bool is_unavailable(const Status &status) {
        return status.map_code() == turbo::kUnavailable;
    }

    bool is_unimplemented(const Status &status) {
        return status.map_code() == turbo::kUnimplemented;
    }

    bool is_unknown(const Status &status) {
        return status.map_code() == turbo::kUnknown;
    }

    bool is_already_stop(const Status &status) {
        return status.map_code() == turbo::kAlreadyStop;
    }

    bool is_resource_busy(const Status &status) {
        return status.map_code() == turbo::kResourceBusy;
    }


    namespace {
        std::string MessageForErrnoToStatus(int error_number,
                                            std::string_view message) {
            return turbo::format("{}: {}", message,terror(error_number));
        }
    }  // namespace

    Status errno_to_status(int error_number, std::string_view message) {
        return Status(errno_to_status_code(error_number),
                      MessageForErrnoToStatus(error_number, message));
    }

    namespace status_internal {

        std::string *MakeCheckFailString(const turbo::Status *status,
                                         const char *prefix) {
            return new std::string(
                    turbo::format("{} ({})", prefix,
                                  status->to_string(StatusToStringMode::kWithEverything)));
        }

    }  // namespace status_internal

}  // namespace turbo
