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
//

#ifndef TURBO_STATUS_STATUS_H_
#define TURBO_STATUS_STATUS_H_

#include <ostream>
#include <string>
#include <utility>
#include <optional>

#include "turbo/meta/function_ref.h"
#include "turbo/status/internal/status_internal.h"
#include "turbo/status/status_code.h"
#include "turbo/strings/cord.h"
#include "turbo/strings/string_view.h"
#include "turbo/status/error.h"
#include "turbo/status/turbo_module.h"
#include "turbo/format/format.h"

namespace turbo {

    // turbo::StatusCode
    //
    // An `turbo::StatusCode` is an enumerated type indicating either no error ("OK")
    // or an error condition. In most cases, an `turbo::Status` indicates a
    // recoverable error, and the purpose of signalling an error is to indicate what
    // action to take in response to that error. These error codes map to the proto
    // RPC error codes indicated in https://cloud.google.com/apis/design/errors.
    //
    // The errors listed below are the canonical errors associated with
    // `turbo::Status` and are used throughout the codebase. As a result, these
    // error codes are somewhat generic.
    //
    // In general, try to return the most specific error that applies if more than
    // one error may pertain. For example, prefer `kOutOfRange` over
    // `kFailedPrecondition` if both codes apply. Similarly prefer `kNotFound` or
    // `kAlreadyExists` over `kFailedPrecondition`.
    //
    // Because these errors may cross RPC boundaries, these codes are tied to the
    // `google.rpc.Code` definitions within
    // https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
    // The string value of these RPC codes is denoted within each enum below.
    //
    // If your error handling code requires more context, you can attach payloads
    // to your status. See `turbo::Status::set_payload()` and
    // `turbo::Status::get_payload()` below.

    // status_code_to_string()
    //
    // Returns the name for the status code, or "" if it is an unknown value.
    std::string status_code_to_string(StatusCode code);

    // turbo::StatusToStringMode
    //
    // An `turbo::StatusToStringMode` is an enumerated type indicating how
    // `turbo::Status::ToString()` should construct the output string for a non-ok
    // status.
    enum class StatusToStringMode : int {
        // ToString will not contain any extra data (such as payloads). It will only
        // contain the error code and message, if any.
        kWithNoExtraData = 0,
        // ToString will contain the payloads.
        kWithPayload = 1 << 0,
        // ToString will contain the module name.
        kWithModule = 1 << 1,
        // ToString will include all the extra data this Status has.
        kWithEverything = ~kWithNoExtraData,
        // Default mode used by ToString. Its exact value might change in the future.
        kDefault = kWithPayload,
    };

    // turbo::StatusToStringMode is specified as a bitmask type, which means the
    // following operations must be provided:
    inline constexpr StatusToStringMode operator&(StatusToStringMode lhs,
                                                  StatusToStringMode rhs) {
        return static_cast<StatusToStringMode>(static_cast<int>(lhs) &
                                               static_cast<int>(rhs));
    }

    inline constexpr StatusToStringMode operator|(StatusToStringMode lhs,
                                                  StatusToStringMode rhs) {
        return static_cast<StatusToStringMode>(static_cast<int>(lhs) |
                                               static_cast<int>(rhs));
    }

    inline constexpr StatusToStringMode operator^(StatusToStringMode lhs,
                                                  StatusToStringMode rhs) {
        return static_cast<StatusToStringMode>(static_cast<int>(lhs) ^
                                               static_cast<int>(rhs));
    }

    inline constexpr StatusToStringMode operator~(StatusToStringMode arg) {
        return static_cast<StatusToStringMode>(~static_cast<int>(arg));
    }

    inline StatusToStringMode &operator&=(StatusToStringMode &lhs,
                                          StatusToStringMode rhs) {
        lhs = lhs & rhs;
        return lhs;
    }

    inline StatusToStringMode &operator|=(StatusToStringMode &lhs,
                                          StatusToStringMode rhs) {
        lhs = lhs | rhs;
        return lhs;
    }

    inline StatusToStringMode &operator^=(StatusToStringMode &lhs,
                                          StatusToStringMode rhs) {
        lhs = lhs ^ rhs;
        return lhs;
    }

    /**
     * @ingroup turbo_base_status
     * @brief The `turbo::Status` class is generally used to gracefully handle errors
     * across API boundaries (and in particular across RPC boundaries). Some of
     * these errors may be recoverable, but others may not. Most
     * functions which can produce a recoverable error should be designed to return
     * either an `turbo::Status` (or the similar `turbo::ResultStatus<T>`, which holds
     * either an object of type `T` or an error).
     *
     * API developers should construct their functions to return `turbo::ok_status()`
     * upon success, or an `turbo::StatusCode` upon another type of error (e.g
     * an `turbo::StatusCode::kInvalidArgument` error). The API provides convenience
     * functions to construct each status code.
     *
     * Example:
     * @code
     * turbo::Status myFunction(std::string_view fname, ...) {
     *   ...
     *   // encounter error
     *   if (error condition) {
     *     // Construct an turbo::StatusCode::kInvalidArgument error
     *     return turbo::invalid_argument_error("bad mode");
     *   }
     *   // else, return OK
     *   return turbo::ok_status();
     * }
     * @endcode
     * Users handling status error codes should prefer checking for an OK status
     * using the `ok()` member function. Handling multiple error codes may justify
     * use of switch statement, but only check for error codes you know how to
     * handle; do not try to exhaustively match against all canonical error codes.
     * Errors that cannot be handled should be logged and/or propagated for higher
     * levels to deal with. If you do use a switch statement, make sure that you
     * also provide a `default:` switch case, so that code does not break as other
     * canonical codes are added to the API.
     *
     * Example:
     * @code
     *      turbo::Status result = DoSomething();
     *      if (!result.ok()) {
     *          TLOG_ERROR("",result);
     *      }
     *
     *      // Provide a default if switching on multiple error codes
     *      switch (result.code()) {
     *          // The user hasn't authenticated. Ask them to reauth
     *          case turbo::StatusCode::kUnauthenticated:
     *              DoReAuth();
     *              break;
     *          // The user does not have permission. Log an error.
     *          case turbo::StatusCode::kPermissionDenied:
     *              TLOG_ERROR("",result);
     *               break;
     *              // Propagate the error otherwise.
     *          default:
     *          return true;
     *      }
     *     @endcode
     *     An `turbo::Status` can optionally include a payload with more information
     *     about the error. Typically, this payload serves one of several purposes:
     *     - It may provide more fine-grained semantic information about the error to
     *     facilitate actionable remedies.
     *     - It may provide human-readable contexual information that is more
     *     appropriate to display to an end user.
     *     For documentation see https://turbo-docs.readthedocs.io/en/latest/en/api/base_status.html
     *     Returned Status objects may not be ignored. status_internal.h has a forward
     *     declaration of the form
     *     class [[nodiscard]] Status;
     */
    class Status final {
    public:

        /**
         * @brief This default constructor creates an OK status with no message or payload.
         * Avoid this constructor and prefer explicit construction of an OK status
         * with `turbo::ok_status()`.
         */
        Status();

        /**
         * @brief Creates a status in the canonical error space with the specified
         * `      turbo::StatusCode` and error message.  If `code == turbo::StatusCode::kOk`,  // NOLINT
         *       `msg` is ignored and an object identical to an OK status is constructed.
         *
         *       The `msg` string must be in UTF-8. The implementation may complain (e.g.,  // NOLINT
         *       by printing a warning) if it is not.
         * @param code The canonical error code.
         * @param msg The error message.
         */
        Status(turbo::StatusCode code, std::string_view msg);

        /**
         * @brief Creates a status in the canonical error space with the specified
         * `      turbo::StatusCode` and error message.  If `code == turbo::StatusCode::kOk`,  // NOLINT
         *       `msg` is ignored and an object identical to an OK status is constructed.
         *
         *       The `msg` string must be in UTF-8. The implementation may complain (e.g.,  // NOLINT
         *       by printing a warning) if it is not.
         *
         *       addtional: index is the module index, which is used to identify the module
         *       that generates the error. The default value is 0, which means turbo self module.
         *       The index should be greater than 0. The index is defined in turbo_module.h.
         *       using turbo::ModuleIndex will define the index unqiue in the whole project.
         * @param index The module index.
         * @param code The canonical error code.
         * @param msg The error message.
         */
        Status(unsigned short int index, turbo::StatusCode code, std::string_view msg);

        /**
         * @brief copy constructor
         * @param x
         */
        Status(const Status &);

        /**
         * @brief copy assignment
         * @param x
         * @return Status&
         */
        Status &operator=(const Status &x);

        /**
         * @brief move constructor
         * @param x
         */
        Status(Status &&) noexcept;

        Status &operator=(Status &&);

        ~Status();

        /**
         * @brief Updates the existing status with `new_status` provided that `this->ok()`.
         *        If the existing status already contains a non-OK error, this update has no
         *        effect and preserves the current data. Note that this behavior may change
         *        in the future to augment a current non-ok status with additional
         *        information about `new_status`.
         *        `Update()` provides a convenient way of keeping track of the first error
         *        encountered.
         *        Example:
         *        @code
         *        // Instead of "if (overall_status.ok()) overall_status = new_status"
         *        overall_status.Update(new_status);
         *        @endcode
         * @param new_status
         */
        void Update(const Status &new_status);

        /**
         * @brief Updates the existing status with `new_status` provided that `this->ok()`.
         *        If the existing status already contains a non-OK error, this update has no
         *        effect and preserves the current data. Note that this behavior may change
         *        in the future to augment a current non-ok status with additional
         *        information about `new_status`.
         *        `Update()` provides a convenient way of keeping track of the first error
         *        encountered.
         *        Example:
         *        @code
         *        // Instead of "if (overall_status.ok()) overall_status = new_status"
         *        overall_status.Update(new_status);
         *        @endcode
         * @note This function will move the `new_status` to `this`. Thus, the `new_status`
         *       will be empty after this function.
         * @param new_status
         */
        void Update(Status &&new_status);

        /**
         * @brief Returns `true` if `this->code()` == `turbo::StatusCode::kOk`,
         *        indicating the absence of an error.
         *        Prefer checking for an OK status using this member function.
         * @return bool
         */
        [[nodiscard]] bool ok() const;

        // Status::map_code()
        //
        // Returns the canonical error code of type `turbo::StatusCode` of this status.
        turbo::StatusCode map_code() const;

        // Status::code()
        //

        int code() const;

        /**
         * @brief Returns the module index of this status.
         * @return unsigned short int
         */
        [[nodiscard]] unsigned short int index() const;

        /**
         * @brief Returns the error message associated with this error code, if available.
         *        Note that this message rarely describes the error code.  It is not unusual
         *        for the error message to be the empty string. As a result, prefer
         *        `operator<<` or `Status::ToString()` for debug logging.
         * @return std::string_view
         */
        [[nodiscard]]
        std::string_view message() const;

        friend bool operator==(const Status &, const Status &);

        friend bool operator!=(const Status &, const Status &);


        /**
         * @brief Returns a string based on the `mode`. By default, it returns combination of
         *        the error code name, the message and any associated payload messages. This
         *        string is designed simply to be human readable and its exact format should
         *        not be load bearing. Do not depend on the exact format of the result of
         *        `to_string()` which is subject to change.
         *        The printed code name and the message are generally substrings of the
         *        result, and the payloads to be printed use the status payload printer
         *        mechanism (which is internal).
         * @param mode
         * @return std::string
         */
        [[nodiscard]] std::string to_string(
                StatusToStringMode mode = StatusToStringMode::kDefault) const;

        /**
         * @brief Ignores any errors. This method does nothing except potentially suppress
         *        complaints from any tools that are checking that errors are not dropped on
         *        the floor.
         */
        void ignore_error() const;


        /**
         * @brief Swap the contents of one status with another.
         * @param a
         * @param b
         */
        friend void swap(Status &a, Status &b);

        /**
         * @brief Gets the payload of a status given its unique `type_url` key, if present.
         *        A payload may be attached to a status to provide additional context to an
         *        error that may not be satisfied by an existing `turbo::StatusCode`.
         *        Typically, this payload serves one of several purposes:
         *        * It may provide more fine-grained semantic information about the error
         *        to facilitate actionable remedies.
         *        * It may provide human-readable contexual information that is more
         *        appropriate to display to an end user.
         *        A payload consists of a [key,value] pair, where the key is a string
         *        referring to a unique "type URL" and the value is an object of type
         *        `turbo::Cord` to hold the contextual data.
         *        The "type URL" should be unique and follow the format of a URL
         *        (https://en.wikipedia.org/wiki/URL) and, ideally, provide some
         *        documentation or schema on how to interpret its associated data. For
         *        example, the default type URL for a protobuf message type is
         *        "type.googleapis.com/packagename.messagename". Other custom wire formats
         *        should define the format of type URL in a similar practice so as to
         *        minimize the chance of conflict between type URLs.
         *        Users should ensure that the type URL can be mapped to a concrete
         *        C++ type if they want to deserialize the payload and read it effectively.
         *        To attach a payload to a status object, call `Status::set_payload()`,
         *        passing it the type URL and an `turbo::Cord` of associated data. Similarly,
         *        to extract the payload from a status, call `Status::get_payload()`. You
         *        may attach multiple payloads (with differing type URLs) to any given
         *        status object, provided that the status is currently exhibiting an error
         *        code (i.e. is not OK).
         *        @code
         *        turbo::Status status = turbo::invalid_argument_error("bad mode");
         *        status.set_payload("type.googleapis.com/google.rpc.BadRequest",
         *        SerializeToCord(request));
         *        @endcode
         *        To retrieve the payload, call `Status::get_payload()` with the same
         *        `type_url` key.
         *        @code
         *        turbo::Cord payload = status.get_payload("type.googleapis.com/google.rpc.BadRequest");
         *        @endcode
         *        If the payload is not present, `Status::get_payload()` will return an empty
         *        `turbo::Cord`.
         *        @code
         *        turbo::Cord payload = status.get_payload("type.googleapis.com/google.rpc.DebugInfo");
         *        assert(payload.empty());
         *        @endcode
         *        If the status is OK, `Status::get_payload()` will return an empty
         *        `turbo::Cord`.
         *        @code
         *        turbo::Status status = turbo::ok_status();
         *        turbo::Cord payload = status.get_payload("type.googleapis.com/google.rpc.DebugInfo");
         *        assert(payload.empty());
         *        @endcode
         * @param type_url The unique type URL key.
         * @return std::optional<turbo::Cord>
         */
        std::optional<turbo::Cord> get_payload(std::string_view type_url) const;

        /**
         * @brief Sets the payload for a non-ok status using a `type_url` key, overwriting
         *        any existing payload for that `type_url`.
         * @note This function does nothing if the Status is ok.
         * @param type_url The unique type URL key.
         * @param payload The payload to set.
         */
        void set_payload(std::string_view type_url, turbo::Cord payload);

        /**
         * @brief Erases the payload corresponding to the `type_url` key.  Returns `true` if
         *        the payload was present.
         * @note This function does nothing if the Status is ok.
         * @param type_url The unique type URL key.
         * @return bool
         */
        bool erase_payload(std::string_view type_url);

        /**
         * @brief Iterates over the stored payloads and calls the
         *        `visitor(type_key, payload)` callable for each one.
         * @note The order of calls to `visitor()` is not specified and may change at
         *       any time.
         * @note Any mutation on the same 'turbo::Status' object during visitation is
         *       forbidden and could result in undefined behavior.
         * @param visitor
         */
        void for_each_payload(
                turbo::FunctionRef<void(std::string_view, const turbo::Cord &)> visitor)
        const;

    private:
        friend Status cancelled_error();

        // Creates a status in the canonical error space with the specified
        // code, and an empty error message.
        explicit Status(turbo::StatusCode code);

        static void UnrefNonInlined(uintptr_t rep);

        static void Ref(uintptr_t rep);

        static void Unref(uintptr_t rep);

        // REQUIRES: !ok()
        // Ensures rep_ is not shared with any other Status.
        void PrepareToModify();

        const status_internal::Payloads *GetPayloads() const;

        status_internal::Payloads *GetPayloads();

        static bool EqualsSlow(const turbo::Status &a, const turbo::Status &b);

        // MSVC 14.0 limitation requires the const.
        static constexpr const char kMovedFromString[] =
                "Status accessed after move.";

        static const std::string *EmptyString();

        static const std::string *MovedFromString();

        // Returns whether rep contains an inlined representation.
        // See rep_ for details.
        static bool IsInlined(uintptr_t rep);

        // Indicates whether this Status was the rhs of a move operation. See rep_
        // for details.
        static bool IsMovedFrom(uintptr_t rep);

        static uintptr_t MovedFromRep();

        // Convert between error::Code and the inlined uintptr_t representation used
        // by rep_. See rep_ for details.
        static uintptr_t CodeToInlinedRep(turbo::StatusCode code);

        static uintptr_t CodeToInlinedRep(unsigned short int module_index, turbo::StatusCode code);

        static turbo::StatusCode InlinedRepToCode(uintptr_t rep);

        static unsigned short int InlinedRepToIndex(uintptr_t rep);

        // Converts between StatusRep* and the external uintptr_t representation used
        // by rep_. See rep_ for details.
        static uintptr_t PointerToRep(status_internal::StatusRep *r);

        static status_internal::StatusRep *RepToPointer(uintptr_t r);

        std::string ToStringSlow(StatusToStringMode mode) const;

        // Status supports two different representations.
        //  - When the low bit is off it is an inlined representation.
        //    It uses the canonical error space, no message or payload.
        //    The error code is (rep_ >> 2).
        //    The (rep_ & 2) bit is the "moved from" indicator, used in IsMovedFrom().
        //  - When the low bit is on it is an external representation.
        //    In this case all the data comes from a heap allocated Rep object.
        //    (rep_ - 1) is a status_internal::StatusRep* pointer to that structure.
        uintptr_t rep_;
    };

    // ok_status()
    //
    // Returns an OK status, equivalent to a default constructed instance. Prefer
    // usage of `turbo::ok_status()` when constructing such an OK status.
    Status ok_status();

    // operator<<()
    //
    // Prints a human-readable representation of `x` to `os`.
    std::ostream &operator<<(std::ostream &os, const Status &x);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::KAbrted.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_aborted(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kAlreadyExists.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_already_exists(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kCancelled.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_cancelled(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kDataLoss.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_data_loss(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kDeadlineExceeded.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_deadline_exceeded(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kFailedPrecondition.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_failed_precondition(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kInternal.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_internal(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kInvalidArgument.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_invalid_argument(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kNotFound.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_not_found(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kOutOfRange.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_out_of_range(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kPermissionDenied.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_permission_denied(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kResourceExhausted.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_resource_exhausted(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kUnauthenticated.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_unauthenticated(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kUnavailable.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_unavailable(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kUnimplemented.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_unimplemented(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kUnknown.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_unknown(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kAlreadyStop.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_already_stop(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Returns `true` if `status` is turbo::kResourceBusy.
     * @param status The status to check.
     * @return bool
     */
    [[nodiscard]] bool is_resource_busy(const Status &status);

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::aborted_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::aborted_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status aborted_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kAborted, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::already_exists_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::already_exists_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status already_exists_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kAlreadyExists, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::cancelled_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::cancelled_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status cancelled_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kCancelled, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::data_loss_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::data_loss_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status data_loss_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kDataLoss, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::deadline_exceeded_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::deadline_exceeded_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status deadline_exceeded_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kDeadlineExceeded, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::failed_precondition_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::failed_precondition_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status failed_precondition_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kFailedPrecondition, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::internal_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::internal_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status internal_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kInternal, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::invalid_argument_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::invalid_argument_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status invalid_argument_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kInvalidArgument, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::not_found_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::not_found_error("bad mode: {}", mode);
     *        @endcode
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status not_found_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kNotFound, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::out_of_range_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::out_of_range_error("bad mode: {}", mode);
     *        @endcode
     *        @note This function is not recommended to use. Use `turbo::reach_file_end_error()` instead.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     * @deprecated
     */
    template<typename ...Args>
    Status out_of_range_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kOutOfRange, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message. This function is recommended to use instead of
     *        `turbo::out_of_range_error()`.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode: {}", mode);
     *        @endcode
     *        @note This function is not recommended to use. Use `turbo::reach_file_end_error()` instead.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @deprecated
     * @return Status
     */
    template<typename ...Args>
    Status permission_denied_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kPermissionDenied, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message. This function is recommended to use instead of
     *        `turbo::out_of_range_error()`.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode: {}", mode);
     *        @endcode
     *        @note This function is not recommended to use. Use `turbo::reach_file_end_error()` instead.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @deprecated
     * @return Status
     */
    template<typename ...Args>
    Status resource_exhausted_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kResourceExhausted, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message. This function is recommended to use instead of
     *        `turbo::out_of_range_error()`.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode: {}", mode);
     *        @endcode
     *        @note This function is not recommended to use. Use `turbo::reach_file_end_error()` instead.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @deprecated
     * @return Status
     */
    template<typename ...Args>
    Status unauthenticated_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kUnauthenticated, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message. This function is recommended to use instead of
     *        `turbo::out_of_range_error()`.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode: {}", mode);
     *        @endcode
     *        @note This function is not recommended to use. Use `turbo::reach_file_end_error()` instead.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @deprecated
     * @return Status
     */
    template<typename ...Args>
    Status unavailable_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kUnavailable, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name,
     *        using the fmt and args to format the error message. This function is recommended to use instead of
     *        `turbo::out_of_range_error()`.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode");
     *        @endcode
     *        @code
     *        turbo::Status status = turbo::reach_file_end_error("bad mode: {}", mode);
     *        @endcode
     *        @note This function is not recommended to use. Use `turbo::reach_file_end_error()` instead.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @deprecated
     * @return Status
     */
    template<typename ...Args>
    Status unimplemented_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kUnimplemented, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::unknown_error("");
     *        @endcode
     * @return Status
     */
    template<typename ...Args>
    Status unknown_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kUnknown, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::already_stop_error("");
     *        @endcode
     * @return Status
     */
    template<typename ...Args>
    Status already_stop_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kAlreadyStop, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::resource_busy_error("");
     *        @endcode
     * @return Status
     */
    template<typename ...Args>
    Status resource_busy_error(std::string_view fmt, Args &&...args) {
        return Status(turbo::kResourceBusy, format(fmt, std::forward<Args>(args)...));
    }
    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code and the error message.
     *       Example:
     *       @code
     *       turbo::Status status = turbo::make_status(turbo::kAborted, "bad mode");
     *       @endcode
     *       @code
     *       turbo::Status status = turbo::make_status(turbo::kAborted, "bad mode: {}", mode);
     *       @endcode
     * @param errcode The error code.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status make_status(int errcode, std::string_view fmt, Args &&...args) {
        return Status(errcode, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with defined module index, error code and the error message.
     *      Example:
     *      @code
     *      turbo::Status status = turbo::make_status(1, turbo::kAborted, "bad mode");
     *      @endcode
     *      @code
     *      turbo::Status status = turbo::make_status(1, turbo::kAborted, "bad mode: {}", mode);
     *      @endcode
     * @param module_index The module index.
     * @param errcode The error code.
     * @param fmt The format string. @see turbo::format()
     * @param args The arguments to format the error message.
     * @return Status
     */
    template<typename ...Args>
    Status make_status(short module_index, int errcode, std::string_view fmt, Args &&...args) {
        return Status(module_index, errcode, format(fmt, std::forward<Args>(args)...));
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code.
     *       Example:
     *       @code
     *       turbo::Status status = turbo::make_status(turbo::kAborted);
     *       @endcode
     * @note This function will create an `turbo::Status` object with an empty error message.
     * @param errcode The error code.
     * @return Status
     */
    TURBO_FORCE_INLINE Status make_status(int errcode) {
        return Status(errcode, "");
    }

    TURBO_FORCE_INLINE Status make_status() {
        return Status(errno,"");
    }

    TURBO_FORCE_INLINE Status make_status(std::error_code ec) {
        return Status(ec.value(), "");
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with defined module index and error code.
     *      Example:
     *      @code
     *      turbo::Status status = turbo::make_status(1, turbo::kAborted);
     *      @endcode
     * @note This function will create an `turbo::Status` object with an empty error message.
     * @param module_index The module index.
     * @param errcode The error code.
     * @return Status
     */
    TURBO_FORCE_INLINE Status make_status(short module_index, int errcode) {
        return Status(module_index, errcode, "");
    }

    /**
     * @ingroup turbo_base_status
     * @brief Creates an `turbo::Status` object with an error code as indicated by the associated function name.
     *        Example:
     *        @code
     *        turbo::Status status = turbo::errno_to_status(1);
     *        @endcode
     * @param error_number The error number.
     * @return Status
     */
    Status errno_to_status(int error_number, std::string_view message);

    //------------------------------------------------------------------------------
    // Implementation details follow
    //------------------------------------------------------------------------------

    inline Status::Status() : rep_(CodeToInlinedRep(turbo::kOk)) {}

    inline Status::Status(turbo::StatusCode code) : rep_(CodeToInlinedRep(code)) {}

    inline Status::Status(const Status &x) : rep_(x.rep_) { Ref(rep_); }

    inline Status &Status::operator=(const Status &x) {
        uintptr_t old_rep = rep_;
        if (x.rep_ != old_rep) {
            Ref(x.rep_);
            rep_ = x.rep_;
            Unref(old_rep);
        }
        return *this;
    }

    inline Status::Status(Status &&x) noexcept: rep_(x.rep_) {
        x.rep_ = MovedFromRep();
    }

    inline Status &Status::operator=(Status &&x) {
        uintptr_t old_rep = rep_;
        if (x.rep_ != old_rep) {
            rep_ = x.rep_;
            x.rep_ = MovedFromRep();
            Unref(old_rep);
        }
        return *this;
    }

    inline void Status::Update(const Status &new_status) {
        if (ok()) {
            *this = new_status;
        }
    }

    inline void Status::Update(Status &&new_status) {
        if (ok()) {
            *this = std::move(new_status);
        }
    }

    inline Status::~Status() { Unref(rep_); }

    inline bool Status::ok() const {
        return rep_ == CodeToInlinedRep(turbo::kOk);
    }

    inline std::string_view Status::message() const {
        return !IsInlined(rep_)
               ? RepToPointer(rep_)->message
               : (IsMovedFrom(rep_) ? std::string_view(kMovedFromString)
                                    : std::string_view());
    }

    inline bool operator==(const Status &lhs, const Status &rhs) {
        return lhs.rep_ == rhs.rep_ || Status::EqualsSlow(lhs, rhs);
    }

    inline bool operator!=(const Status &lhs, const Status &rhs) {
        return !(lhs == rhs);
    }

    inline std::string Status::to_string(StatusToStringMode mode) const {
        return ok() ? "OK" : ToStringSlow(mode);
    }

    inline void Status::ignore_error() const {
        // no-op
    }

    inline void swap(turbo::Status &a, turbo::Status &b) {
        using std::swap;
        swap(a.rep_, b.rep_);
    }

    inline const status_internal::Payloads *Status::GetPayloads() const {
        return IsInlined(rep_) ? nullptr : RepToPointer(rep_)->payloads.get();
    }

    inline status_internal::Payloads *Status::GetPayloads() {
        return IsInlined(rep_) ? nullptr : RepToPointer(rep_)->payloads.get();
    }

    inline bool Status::IsInlined(uintptr_t rep) { return (rep & 1) == 0; }

    inline bool Status::IsMovedFrom(uintptr_t rep) {
        return IsInlined(rep) && (rep & 2) != 0;
    }

    inline uintptr_t Status::MovedFromRep() {
        return CodeToInlinedRep(turbo::kInternal) | 2;
    }

    inline uintptr_t Status::CodeToInlinedRep(turbo::StatusCode code) {
        return static_cast<uintptr_t>(code) << 2;
    }

    inline uintptr_t Status::CodeToInlinedRep(unsigned short int module_index, turbo::StatusCode code) {
        uintptr_t ret = static_cast<uintptr_t>(module_index) << 32;
        ret |= static_cast<uintptr_t>(code);
        return ret << 2;
    }

    inline turbo::StatusCode Status::InlinedRepToCode(uintptr_t rep) {
        assert(IsInlined(rep));
        static constexpr uintptr_t kCodeMask = 0x00000000FFFFFFFF;
        return static_cast<turbo::StatusCode>((rep >> 2) & kCodeMask);
    }

    inline unsigned short int Status::InlinedRepToIndex(uintptr_t rep) {
        assert(IsInlined(rep));
        static constexpr uintptr_t kIndexMask = 0x000000000000FFFF;
        return static_cast<unsigned short int>((rep >> 34) & kIndexMask);
    }

    inline status_internal::StatusRep *Status::RepToPointer(uintptr_t rep) {
        assert(!IsInlined(rep));
        return reinterpret_cast<status_internal::StatusRep *>(rep - 1);
    }

    inline uintptr_t Status::PointerToRep(status_internal::StatusRep *rep) {
        return reinterpret_cast<uintptr_t>(rep) + 1;
    }

    inline void Status::Ref(uintptr_t rep) {
        if (!IsInlined(rep)) {
            RepToPointer(rep)->ref.fetch_add(1, std::memory_order_relaxed);
        }
    }

    inline void Status::Unref(uintptr_t rep) {
        if (!IsInlined(rep)) {
            UnrefNonInlined(rep);
        }
    }

    inline Status ok_status() { return Status(); }

    // Creates a `Status` object with the `turbo::StatusCode::kCancelled` error code
    // and an empty message. It is provided only for efficiency, given that
    // message-less kCancelled errors are common in the infrastructure.
    inline Status cancelled_error() { return Status(turbo::kCancelled); }

    template<>
    struct formatter<turbo::Status> {
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

        template<typename FormatContext>
        auto format(const turbo::Status &status, FormatContext &ctx) {
            return format_to(ctx.out(), "{}", status.to_string());
        }
    };
}  // namespace turbo

#endif  // TURBO_STATUS_STATUS_H_
