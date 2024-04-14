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
//
// -----------------------------------------------------------------------------
// File: result_status.h
// -----------------------------------------------------------------------------
//
// An `turbo::ResultStatus<T>` represents a union of an `turbo::Status` object
// and an object of type `T`. The `turbo::ResultStatus<T>` will either contain an
// object of type `T` (indicating a successful operation), or an error (of type
// `turbo::Status`) explaining why such a value is not present.
//
// In general, check the success of an operation returning an
// `turbo::ResultStatus<T>` like you would an `turbo::Status` by using the `ok()`
// member function.
//
// Example:
//
//   ResultStatus<Foo> result = Calculation();
//   if (result.ok()) {
//     result->DoSomethingCool();
//   } else {
//     TURBO_LOG(ERROR) << result.status();
//   }
#ifndef TURBO_STATUS_RESULT_STATUS_H_
#define TURBO_STATUS_RESULT_STATUS_H_

#include <exception>
#include <initializer_list>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include "turbo/base/internal/result_status_internal.h"
#include "turbo/status/status.h"
#include "turbo/meta/type_traits.h"
#include "turbo/meta/utility.h"
#include <variant>
#include "turbo/platform/port.h"

namespace turbo {

    // BadResultStatusAccess
    //
    // This class defines the type of object to throw (if exceptions are enabled),
    // when accessing the value of an `turbo::ResultStatus<T>` object that does not
    // contain a value. This behavior is analogous to that of
    // `std::bad_optional_access` in the case of accessing an invalid
    // `std::optional` value.
    //
    // Example:
    //
    // try {
    //   turbo::ResultStatus<int> v = FetchInt();
    //   DoWork(v.value());  // Accessing value() when not "OK" may throw
    // } catch (turbo::BadResultStatusAccess& ex) {
    //   TURBO_LOG(ERROR) << ex.status();
    // }
    class BadResultStatusAccess : public std::exception {
    public:
        explicit BadResultStatusAccess(turbo::Status status);

        ~BadResultStatusAccess() override = default;

        BadResultStatusAccess(const BadResultStatusAccess &other);

        BadResultStatusAccess &operator=(const BadResultStatusAccess &other);

        BadResultStatusAccess(BadResultStatusAccess &&other) noexcept;

        BadResultStatusAccess &operator=(BadResultStatusAccess &&other) noexcept;

        // BadResultStatusAccess::what()
        //
        // Returns the associated explanatory string of the `turbo::ResultStatus<T>`
        // object's error code. This function contains information about the failing
        // status, but its exact formatting may change and should not be depended on.
        //
        // The pointer of this string is guaranteed to be valid until any non-const
        // function is invoked on the exception object.
        const char *what() const noexcept override;

        // BadResultStatusAccess::status()
        //
        // Returns the associated `turbo::Status` of the `turbo::ResultStatus<T>` object's
        // error.
        const turbo::Status &status() const;

    private:
        void InitWhat() const;

        turbo::Status status_;
        mutable std::once_flag init_what_;
        mutable std::string what_;
    };

// Returned ResultStatus objects may not be ignored.
    template<typename T>
#if TURBO_HAVE_CPP_ATTRIBUTE(nodiscard)
    // TODO(b/176172494): TURBO_MUST_USE_RESULT should expand to the more strict
    // [[nodiscard]]. For now, just use [[nodiscard]] directly when it is available.
    class [[nodiscard]] ResultStatus;
#else
    class TURBO_MUST_USE_RESULT ResultStatus;

#endif  // TURBO_HAVE_CPP_ATTRIBUTE(nodiscard)

// turbo::ResultStatus<T>
//
// The `turbo::ResultStatus<T>` class template is a union of an `turbo::Status` object
// and an object of type `T`. The `turbo::ResultStatus<T>` models an object that is
// either a usable object, or an error (of type `turbo::Status`) explaining why
// such an object is not present. An `turbo::ResultStatus<T>` is typically the return
// value of a function which may fail.
//
// An `turbo::ResultStatus<T>` can never hold an "OK" status (an
// `turbo::StatusCode::kOk` value); instead, the presence of an object of type
// `T` indicates success. Instead of checking for a `kOk` value, use the
// `turbo::ResultStatus<T>::ok()` member function. (It is for this reason, and code
// readability, that using the `ok()` function is preferred for `turbo::Status`
// as well.)
//
// Example:
//
//   ResultStatus<Foo> result = DoBigCalculationThatCouldFail();
//   if (result.ok()) {
//     result->DoSomethingCool();
//   } else {
//     TURBO_LOG(ERROR) << result.status();
//   }
//
// Accessing the object held by an `turbo::ResultStatus<T>` should be performed via
// `operator*` or `operator->`, after a call to `ok()` confirms that the
// `turbo::ResultStatus<T>` holds an object of type `T`:
//
// Example:
//
//   turbo::ResultStatus<int> i = GetCount();
//   if (i.ok()) {
//     updated_total += *i
//   }
//
// NOTE: using `turbo::ResultStatus<T>::value()` when no valid value is present will
// throw an exception if exceptions are enabled or terminate the process when
// exceptions are not enabled.
//
// Example:
//
//   ResultStatus<Foo> result = DoBigCalculationThatCouldFail();
//   const Foo& foo = result.value();    // Crash/exception if no value present
//   foo.DoSomethingCool();
//
// A `turbo::ResultStatus<T*>` can be constructed from a null pointer like any other
// pointer value, and the result will be that `ok()` returns `true` and
// `value()` returns `nullptr`. Checking the value of pointer in an
// `turbo::ResultStatus<T*>` generally requires a bit more care, to ensure both that
// a value is present and that value is not null:
//
//  ResultStatus<std::unique_ptr<Foo>> result = FooFactory::MakeNewFoo(arg);
//  if (!result.ok()) {
//    TURBO_LOG(ERROR) << result.status();
//  } else if (*result == nullptr) {
//    TURBO_LOG(ERROR) << "Unexpected null pointer";
//  } else {
//    (*result)->DoSomethingCool();
//  }
//
// Example factory implementation returning ResultStatus<T>:
//
//  ResultStatus<Foo> FooFactory::MakeFoo(int arg) {
//    if (arg <= 0) {
//      return turbo::Status(turbo::StatusCode::kInvalidArgument,
//                          "Arg must be positive");
//    }
//    return Foo(arg);
//  }
    template<typename T>
    class ResultStatus : private result_status_internal::ResultStatusData<T>,
                         private result_status_internal::CopyCtorBase<T>,
                         private result_status_internal::MoveCtorBase<T>,
                         private result_status_internal::CopyAssignBase<T>,
                         private result_status_internal::MoveAssignBase<T> {
        template<typename U>
        friend
        class ResultStatus;

        typedef result_status_internal::ResultStatusData<T> Base;

    public:
        // ResultStatus<T>::value_type
        //
        // This instance data provides a generic `value_type` member for use within
        // generic programming. This usage is analogous to that of
        // `optional::value_type` in the case of `std::optional`.
        typedef T value_type;

        // Constructors

        // Constructs a new `turbo::ResultStatus` with an `turbo::StatusCode::kUnknown`
        // status. This constructor is marked 'explicit' to prevent usages in return
        // values such as 'return {};', under the misconception that
        // `turbo::ResultStatus<std::vector<int>>` will be initialized with an empty
        // vector, instead of an `turbo::StatusCode::kUnknown` error code.
        explicit ResultStatus();

        // `ResultStatus<T>` is copy constructible if `T` is copy constructible.
        ResultStatus(const ResultStatus &) = default;

        // `ResultStatus<T>` is copy assignable if `T` is copy constructible and copy
        // assignable.
        ResultStatus &operator=(const ResultStatus &) = default;

        // `ResultStatus<T>` is move constructible if `T` is move constructible.
        ResultStatus(ResultStatus &&) = default;

        // `ResultStatus<T>` is moveAssignable if `T` is move constructible and move
        // assignable.
        ResultStatus &operator=(ResultStatus &&) = default;

        // Converting Constructors

        // Constructs a new `turbo::ResultStatus<T>` from an `turbo::ResultStatus<U>`, when `T`
        // is constructible from `U`. To avoid ambiguity, these constructors are
        // disabled if `T` is also constructible from `ResultStatus<U>.`. This constructor
        // is explicit if and only if the corresponding construction of `T` from `U`
        // is explicit. (This constructor inherits its explicitness from the
        // underlying constructor.)
        template<
                typename U,
                std::enable_if_t<
                        std::conjunction<
                                std::negation<std::is_same<T, U>>,
                                std::is_constructible<T, const U &>,
                                std::is_convertible<const U &, T>,
                                std::negation<
                                        result_status_internal::IsConstructibleOrConvertibleFromResultStatus<
                                                T, U>>>::value,
                        int> = 0>
        ResultStatus(const ResultStatus<U> &other)  // NOLINT
                : Base(static_cast<const typename ResultStatus<U>::Base &>(other)) {}

        template<
                typename U,
                std::enable_if_t<
                        std::conjunction<
                                std::negation<std::is_same<T, U>>,
                                std::is_constructible<T, const U &>,
                                std::negation<std::is_convertible<const U &, T>>,
                                std::negation<
                                        result_status_internal::IsConstructibleOrConvertibleFromResultStatus<
                                                T, U>>>::value,
                        int> = 0>
        explicit ResultStatus(const ResultStatus<U> &other)
                : Base(static_cast<const typename ResultStatus<U>::Base &>(other)) {}

        template<
                typename U,
                std::enable_if_t<
                        std::conjunction<
                                std::negation<std::is_same<T, U>>, std::is_constructible<T, U &&>,
                                std::is_convertible<U &&, T>,
                                std::negation<
                                        result_status_internal::IsConstructibleOrConvertibleFromResultStatus<
                                                T, U>>>::value,
                        int> = 0>
        ResultStatus(ResultStatus<U> &&other)  // NOLINT
                : Base(static_cast<typename ResultStatus<U>::Base &&>(other)) {}

        template<
                typename U,
                std::enable_if_t<
                        std::conjunction<
                                std::negation<std::is_same<T, U>>, std::is_constructible<T, U &&>,
                                std::negation<std::is_convertible<U &&, T>>,
                                std::negation<
                                        result_status_internal::IsConstructibleOrConvertibleFromResultStatus<
                                                T, U>>>::value,
                        int> = 0>
        explicit ResultStatus(ResultStatus<U> &&other)
                : Base(static_cast<typename ResultStatus<U>::Base &&>(other)) {}

        // Converting Assignment Operators

        // Creates an `turbo::ResultStatus<T>` through assignment from an
        // `turbo::ResultStatus<U>` when:
        //
        //   * Both `turbo::ResultStatus<T>` and `turbo::ResultStatus<U>` are OK by assigning
        //     `U` to `T` directly.
        //   * `turbo::ResultStatus<T>` is OK and `turbo::ResultStatus<U>` contains an error
        //      code by destroying `turbo::ResultStatus<T>`'s value and assigning from
        //      `turbo::ResultStatus<U>'
        //   * `turbo::ResultStatus<T>` contains an error code and `turbo::ResultStatus<U>` is
        //      OK by directly initializing `T` from `U`.
        //   * Both `turbo::ResultStatus<T>` and `turbo::ResultStatus<U>` contain an error
        //     code by assigning the `Status` in `turbo::ResultStatus<U>` to
        //     `turbo::ResultStatus<T>`
        //
        // These overloads only apply if `turbo::ResultStatus<T>` is constructible and
        // assignable from `turbo::ResultStatus<U>` and `ResultStatus<T>` cannot be directly
        // assigned from `ResultStatus<U>`.
        template<
                typename U,
                std::enable_if_t<
                        std::conjunction<
                                std::negation<std::is_same<T, U>>,
                                std::is_constructible<T, const U &>,
                                std::is_assignable<T, const U &>,
                                std::negation<
                                        result_status_internal::
                                        IsConstructibleOrConvertibleOrAssignableFromResultStatus<
                                                T, U>>>::value,
                        int> = 0>
        ResultStatus &operator=(const ResultStatus<U> &other) {
            this->Assign(other);
            return *this;
        }

        template<
                typename U,
                std::enable_if_t<
                        std::conjunction<
                                std::negation<std::is_same<T, U>>, std::is_constructible<T, U &&>,
                                std::is_assignable<T, U &&>,
                                std::negation<
                                        result_status_internal::
                                        IsConstructibleOrConvertibleOrAssignableFromResultStatus<
                                                T, U>>>::value,
                        int> = 0>
        ResultStatus &operator=(ResultStatus<U> &&other) {
            this->Assign(std::move(other));
            return *this;
        }

        // Constructs a new `turbo::ResultStatus<T>` with a non-ok status. After calling
        // this constructor, `this->ok()` will be `false` and calls to `value()` will
        // crash, or produce an exception if exceptions are enabled.
        //
        // The constructor also takes any type `U` that is convertible to
        // `turbo::Status`. This constructor is explicit if an only if `U` is not of
        // type `turbo::Status` and the conversion from `U` to `Status` is explicit.
        //
        // REQUIRES: !Status(std::forward<U>(v)).ok(). This requirement is DCHECKed.
        // In optimized builds, passing turbo::ok_status() here will have the effect
        // of passing turbo::StatusCode::kInternal as a fallback.
        template<
                typename U = turbo::Status,
                std::enable_if_t<
                        std::conjunction<
                                std::is_convertible<U &&, turbo::Status>,
                                std::is_constructible<turbo::Status, U &&>,
                                std::negation<std::is_same<std::decay_t<U>, turbo::ResultStatus<T>>>,
                                std::negation<std::is_same<std::decay_t<U>, T>>,
                                std::negation<std::is_same<std::decay_t<U>, std::in_place_t>>,
                                std::negation<result_status_internal::HasConversionOperatorToResultStatus<
                                        T, U &&>>>::value,
                        int> = 0>
        ResultStatus(U &&v) noexcept : Base(std::forward<U>(v)) {}

        template<
                typename U = turbo::Status,
                std::enable_if_t<
                        std::conjunction<
                                std::negation<std::is_convertible<U &&, turbo::Status>>,
                                std::is_constructible<turbo::Status, U &&>,
                                std::negation<std::is_same<std::decay_t<U>, turbo::ResultStatus<T>>>,
                                std::negation<std::is_same<std::decay_t<U>, T>>,
                                std::negation<std::is_same<std::decay_t<U>, std::in_place_t>>,
                                std::negation<result_status_internal::HasConversionOperatorToResultStatus<
                                        T, U &&>>>::value,
                        int> = 0>
        explicit ResultStatus(U &&v) : Base(std::forward<U>(v)) {}

        template<
                typename U = turbo::Status,
                std::enable_if_t<
                        std::conjunction<
                                std::is_convertible<U &&, turbo::Status>,
                                std::is_constructible<turbo::Status, U &&>,
                                std::negation<std::is_same<std::decay_t<U>, turbo::ResultStatus<T>>>,
                                std::negation<std::is_same<std::decay_t<U>, T>>,
                                std::negation<std::is_same<std::decay_t<U>, std::in_place_t>>,
                                std::negation<result_status_internal::HasConversionOperatorToResultStatus<
                                        T, U &&>>>::value,
                        int> = 0>
        ResultStatus &operator=(U &&v) {
            this->AssignStatus(std::forward<U>(v));
            return *this;
        }

        // Perfect-forwarding value assignment operator.

        // If `*this` contains a `T` value before the call, the contained value is
        // assigned from `std::forward<U>(v)`; Otherwise, it is directly-initialized
        // from `std::forward<U>(v)`.
        // This function does not participate in overload unless:
        // 1. `std::is_constructible_v<T, U>` is true,
        // 2. `std::is_assignable_v<T&, U>` is true.
        // 3. `std::is_same_v<ResultStatus<T>, std::remove_cvref_t<U>>` is false.
        // 4. Assigning `U` to `T` is not ambiguous:
        //  If `U` is `ResultStatus<V>` and `T` is constructible and assignable from
        //  both `ResultStatus<V>` and `V`, the assignment is considered bug-prone and
        //  ambiguous thus will fail to compile. For example:
        //    ResultStatus<bool> s1 = true;  // s1.ok() && *s1 == true
        //    ResultStatus<bool> s2 = false;  // s2.ok() && *s2 == false
        //    s1 = s2;  // ambiguous, `s1 = *s2` or `s1 = bool(s2)`?
        template<
                typename U = T,
                typename = typename std::enable_if<std::conjunction<
                        std::is_constructible<T, U &&>, std::is_assignable<T &, U &&>,
                        std::disjunction<
                                std::is_same<std::remove_cv_t<std::remove_reference_t<U>>, T>,
                                std::conjunction<
                                        std::negation<std::is_convertible<U &&, turbo::Status>>,
                                        std::negation<result_status_internal::
                                        HasConversionOperatorToResultStatus<T, U &&>>>>,
                        result_status_internal::IsForwardingAssignmentValid<T, U &&>>::value>::type>
        ResultStatus &operator=(U &&v) {
            this->Assign(std::forward<U>(v));
            return *this;
        }

        // Constructs the inner value `T` in-place using the provided args, using the
        // `T(args...)` constructor.
        template<typename... Args>
        explicit ResultStatus(std::in_place_t, Args &&... args);

        template<typename U, typename... Args>
        explicit ResultStatus(std::in_place_t, std::initializer_list<U> ilist,
                              Args &&... args);

        // Constructs the inner value `T` in-place using the provided args, using the
        // `T(U)` (direct-initialization) constructor. This constructor is only valid
        // if `T` can be constructed from a `U`. Can accept move or copy constructors.
        //
        // This constructor is explicit if `U` is not convertible to `T`. To avoid
        // ambiguity, this constructor is disabled if `U` is a `ResultStatus<J>`, where
        // `J` is convertible to `T`.
        template<
                typename U = T,
                std::enable_if_t<
                        std::conjunction<
                                result_status_internal::IsDirectInitializationValid<T, U &&>,
                                std::is_constructible<T, U &&>, std::is_convertible<U &&, T>,
                                std::disjunction<
                                        std::is_same<std::remove_cv_t<std::remove_reference_t<U>>,
                                                T>,
                                        std::conjunction<
                                                std::negation<std::is_convertible<U &&, turbo::Status>>,
                                                std::negation<
                                                        result_status_internal::HasConversionOperatorToResultStatus<
                                                                T, U &&>>>>>::value,
                        int> = 0>
        ResultStatus(U &&u)  // NOLINT
                : ResultStatus(std::in_place, std::forward<U>(u)) {}

        template<
                typename U = T,
                std::enable_if_t<
                        std::conjunction<
                                result_status_internal::IsDirectInitializationValid<T, U &&>,
                                std::disjunction<
                                        std::is_same<std::remove_cv_t<std::remove_reference_t<U>>,
                                                T>,
                                        std::conjunction<
                                                std::negation<std::is_constructible<turbo::Status, U &&>>,
                                                std::negation<
                                                        result_status_internal::HasConversionOperatorToResultStatus<
                                                                T, U &&>>>>,
                                std::is_constructible<T, U &&>,
                                std::negation<std::is_convertible<U &&, T>>>::value,
                        int> = 0>
        explicit ResultStatus(U &&u)  // NOLINT
                : ResultStatus(std::in_place, std::forward<U>(u)) {}

        // ResultStatus<T>::ok()
        //
        // Returns whether or not this `turbo::ResultStatus<T>` holds a `T` value. This
        // member function is analogous to `turbo::Status::ok()` and should be used
        // similarly to check the status of return values.
        //
        // Example:
        //
        // ResultStatus<Foo> result = DoBigCalculationThatCouldFail();
        // if (result.ok()) {
        //    // Handle result
        // else {
        //    // Handle error
        // }
        TURBO_MUST_USE_RESULT bool ok() const noexcept { return this->status_.ok(); }

        // ResultStatus<T>::status()
        //
        // Returns a reference to the current `turbo::Status` contained within the
        // `turbo::ResultStatus<T>`. If `turbo::ResultStatus<T>` contains a `T`, then this
        // function returns `turbo::ok_status()`.
        const Status &status() const &;

        Status status() &&;

        // ResultStatus<T>::value()
        //
        // Returns a reference to the held value if `this->ok()`. Otherwise, throws
        // `turbo::BadResultStatusAccess` if exceptions are enabled, or is guaranteed to
        // terminate the process if exceptions are disabled.
        //
        // If you have already checked the status using `this->ok()`, you probably
        // want to use `operator*()` or `operator->()` to access the value instead of
        // `value`.
        //
        // Note: for value types that are cheap to copy, prefer simple code:
        //
        //   T value = statusor.value();
        //
        // Otherwise, if the value type is expensive to copy, but can be left
        // in the ResultStatus, simply assign to a reference:
        //
        //   T& value = statusor.value();  // or `const T&`
        //
        // Otherwise, if the value type supports an efficient move, it can be
        // used as follows:
        //
        //   T value = std::move(statusor).value();
        //
        // The `std::move` on statusor instead of on the whole expression enables
        // warnings about possible uses of the statusor object after the move.
        const T &value() const & TURBO_ATTRIBUTE_LIFETIME_BOUND;

        T &value() & TURBO_ATTRIBUTE_LIFETIME_BOUND;

        const T &&value() const && TURBO_ATTRIBUTE_LIFETIME_BOUND;

        T &&value() && TURBO_ATTRIBUTE_LIFETIME_BOUND;

        // ResultStatus<T>:: operator*()
        //
        // Returns a reference to the current value.
        //
        // REQUIRES: `this->ok() == true`, otherwise the behavior is undefined.
        //
        // Use `this->ok()` to verify that there is a current value within the
        // `turbo::ResultStatus<T>`. Alternatively, see the `value()` member function for a
        // similar API that guarantees crashing or throwing an exception if there is
        // no current value.
        const T &operator*() const & TURBO_ATTRIBUTE_LIFETIME_BOUND;

        T &operator*() & TURBO_ATTRIBUTE_LIFETIME_BOUND;

        const T &&operator*() const && TURBO_ATTRIBUTE_LIFETIME_BOUND;

        T &&operator*() && TURBO_ATTRIBUTE_LIFETIME_BOUND;

        // ResultStatus<T>::operator->()
        //
        // Returns a pointer to the current value.
        //
        // REQUIRES: `this->ok() == true`, otherwise the behavior is undefined.
        //
        // Use `this->ok()` to verify that there is a current value.
        const T *operator->() const TURBO_ATTRIBUTE_LIFETIME_BOUND;

        T *operator->() TURBO_ATTRIBUTE_LIFETIME_BOUND;

        // ResultStatus<T>::value_or()
        //
        // Returns the current value if `this->ok() == true`. Otherwise constructs a
        // value using the provided `default_value`.
        //
        // Unlike `value`, this function returns by value, copying the current value
        // if necessary. If the value type supports an efficient move, it can be used
        // as follows:
        //
        //   T value = std::move(statusor).value_or(def);
        //
        // Unlike with `value`, calling `std::move()` on the result of `value_or` will
        // still trigger a copy.
        template<typename U>
        T value_or(U &&default_value) const &;

        template<typename U>
        T value_or(U &&default_value) &&;

        // ResultStatus<T>::IgnoreError()
        //
        // Ignores any errors. This method does nothing except potentially suppress
        // complaints from any tools that are checking that errors are not dropped on
        // the floor.
        void IgnoreError() const;

        // ResultStatus<T>::emplace()
        //
        // Reconstructs the inner value T in-place using the provided args, using the
        // T(args...) constructor. Returns reference to the reconstructed `T`.
        template<typename... Args>
        T &emplace(Args &&... args) {
            if (ok()) {
                this->Clear();
                this->MakeValue(std::forward<Args>(args)...);
            } else {
                this->MakeValue(std::forward<Args>(args)...);
                this->status_ = turbo::ok_status();
            }
            return this->data_;
        }

        template<
                typename U, typename... Args,
                std::enable_if_t<
                        std::is_constructible<T, std::initializer_list<U> &, Args &&...>::value,
                        int> = 0>
        T &emplace(std::initializer_list<U> ilist, Args &&... args) {
            if (ok()) {
                this->Clear();
                this->MakeValue(ilist, std::forward<Args>(args)...);
            } else {
                this->MakeValue(ilist, std::forward<Args>(args)...);
                this->status_ = turbo::ok_status();
            }
            return this->data_;
        }

    private:
        using result_status_internal::ResultStatusData<T>::Assign;

        template<typename U>
        void Assign(const turbo::ResultStatus<U> &other);

        template<typename U>
        void Assign(turbo::ResultStatus<U> &&other);
    };

// operator==()
//
// This operator checks the equality of two `turbo::ResultStatus<T>` objects.
    template<typename T>
    bool operator==(const ResultStatus<T> &lhs, const ResultStatus<T> &rhs) {
        if (lhs.ok() && rhs.ok()) return *lhs == *rhs;
        return lhs.status() == rhs.status();
    }

// operator!=()
//
// This operator checks the inequality of two `turbo::ResultStatus<T>` objects.
    template<typename T>
    bool operator!=(const ResultStatus<T> &lhs, const ResultStatus<T> &rhs) {
        return !(lhs == rhs);
    }

//------------------------------------------------------------------------------
// Implementation details for ResultStatus<T>
//------------------------------------------------------------------------------

// TODO(sbenza): avoid the string here completely.
    template<typename T>
    ResultStatus<T>::ResultStatus() : Base(Status(turbo::kUnknown, "")) {}

    template<typename T>
    template<typename U>
    inline void ResultStatus<T>::Assign(const ResultStatus<U> &other) {
        if (other.ok()) {
            this->Assign(*other);
        } else {
            this->AssignStatus(other.status());
        }
    }

    template<typename T>
    template<typename U>
    inline void ResultStatus<T>::Assign(ResultStatus<U> &&other) {
        if (other.ok()) {
            this->Assign(*std::move(other));
        } else {
            this->AssignStatus(std::move(other).status());
        }
    }

    template<typename T>
    template<typename... Args>
    ResultStatus<T>::ResultStatus(std::in_place_t, Args &&... args)
            : Base(std::in_place, std::forward<Args>(args)...) {}

    template<typename T>
    template<typename U, typename... Args>
    ResultStatus<T>::ResultStatus(std::in_place_t, std::initializer_list<U> ilist,
                                  Args &&... args)
            : Base(std::in_place, ilist, std::forward<Args>(args)...) {}

    template<typename T>
    const Status &ResultStatus<T>::status() const &{
        return this->status_;
    }

    template<typename T>
    Status ResultStatus<T>::status() &&{
        return ok() ? ok_status() : std::move(this->status_);
    }

    template<typename T>
    const T &ResultStatus<T>::value() const &{
        if (!this->ok()) result_status_internal::ThrowBadResultStatusAccess(this->status_);
        return this->data_;
    }

    template<typename T>
    T &ResultStatus<T>::value() &{
        if (!this->ok()) result_status_internal::ThrowBadResultStatusAccess(this->status_);
        return this->data_;
    }

    template<typename T>
    const T &&ResultStatus<T>::value() const &&{
        if (!this->ok()) {
            result_status_internal::ThrowBadResultStatusAccess(std::move(this->status_));
        }
        return std::move(this->data_);
    }

    template<typename T>
    T &&ResultStatus<T>::value() &&{
        if (!this->ok()) {
            result_status_internal::ThrowBadResultStatusAccess(std::move(this->status_));
        }
        return std::move(this->data_);
    }

    template<typename T>
    const T &ResultStatus<T>::operator*() const &{
        this->EnsureOk();
        return this->data_;
    }

    template<typename T>
    T &ResultStatus<T>::operator*() &{
        this->EnsureOk();
        return this->data_;
    }

    template<typename T>
    const T &&ResultStatus<T>::operator*() const &&{
        this->EnsureOk();
        return std::move(this->data_);
    }

    template<typename T>
    T &&ResultStatus<T>::operator*() &&{
        this->EnsureOk();
        return std::move(this->data_);
    }

    template<typename T>
    const T *ResultStatus<T>::operator->() const {
        this->EnsureOk();
        return &this->data_;
    }

    template<typename T>
    T *ResultStatus<T>::operator->() {
        this->EnsureOk();
        return &this->data_;
    }

    template<typename T>
    template<typename U>
    T ResultStatus<T>::value_or(U &&default_value) const &{
        if (ok()) {
            return this->data_;
        }
        return std::forward<U>(default_value);
    }

    template<typename T>
    template<typename U>
    T ResultStatus<T>::value_or(U &&default_value) &&{
        if (ok()) {
            return std::move(this->data_);
        }
        return std::forward<U>(default_value);
    }

    template<typename T>
    void ResultStatus<T>::IgnoreError() const {
        // no-op
    }

}  // namespace turbo

#endif  // TURBO_STATUS_RESULT_STATUS_H_
