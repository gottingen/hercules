// Copyright 2018 The Turbo Authors.
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
// File: bind_front.h
// -----------------------------------------------------------------------------
//
// `turbo::bind_front()` returns a functor by binding a number of arguments to
// the front of a provided (usually more generic) functor. Unlike `std::bind`,
// it does not require the use of argument placeholders. The simpler syntax of
// `turbo::bind_front()` allows you to avoid known misuses with `std::bind()`.
//
// `turbo::bind_front()` is meant as a drop-in replacement for C++20's upcoming
// `std::bind_front()`, which similarly resolves these issues with
// `std::bind()`. Both `bind_front()` alternatives, unlike `std::bind()`, allow
// partial function application. (See
// https://en.wikipedia.org/wiki/Partial_application).

#ifndef TURBO_FUNCTIONAL_BIND_FRONT_H_
#define TURBO_FUNCTIONAL_BIND_FRONT_H_

#if defined(__cpp_lib_bind_front) && __cpp_lib_bind_front >= 201907L
#include <functional>  // For std::bind_front.
#endif  // defined(__cpp_lib_bind_front) && __cpp_lib_bind_front >= 201907L

#include "turbo/meta/internal/front_binder.h"
#include "turbo/meta/utility.h"

namespace turbo {

    // bind_front()
    //
    // Binds the first N arguments of an invocable object and stores them by value.
    //
    // Like `std::bind()`, `turbo::bind_front()` is implicitly convertible to
    // `std::function`.  In particular, it may be used as a simpler replacement for
    // `std::bind()` in most cases, as it does not require  placeholders to be
    // specified. More importantly, it provides more reliable correctness guarantees
    // than `std::bind()`; while `std::bind()` will silently ignore passing more
    // parameters than expected, for example, `turbo::bind_front()` will report such
    // mis-uses as errors. In C++20, `turbo::bind_front` is replaced by
    // `std::bind_front`.
    //
    // turbo::bind_front(a...) can be seen as storing the results of
    // std::make_tuple(a...).
    //
    // Example: Binding a free function.
    //
    //   int Minus(int a, int b) { return a - b; }
    //
    //   assert(turbo::bind_front(Minus)(3, 2) == 3 - 2);
    //   assert(turbo::bind_front(Minus, 3)(2) == 3 - 2);
    //   assert(turbo::bind_front(Minus, 3, 2)() == 3 - 2);
    //
    // Example: Binding a member function.
    //
    //   struct Math {
    //     int Double(int a) const { return 2 * a; }
    //   };
    //
    //   Math math;
    //
    //   assert(turbo::bind_front(&Math::Double)(&math, 3) == 2 * 3);
    //   // Stores a pointer to math inside the functor.
    //   assert(turbo::bind_front(&Math::Double, &math)(3) == 2 * 3);
    //   // Stores a copy of math inside the functor.
    //   assert(turbo::bind_front(&Math::Double, math)(3) == 2 * 3);
    //   // Stores std::unique_ptr<Math> inside the functor.
    //   assert(turbo::bind_front(&Math::Double,
    //                           std::unique_ptr<Math>(new Math))(3) == 2 * 3);
    //
    // Example: Using `turbo::bind_front()`, instead of `std::bind()`, with
    //          `std::function`.
    //
    //   class FileReader {
    //    public:
    //     void ReadFileAsync(const std::string& filename, std::string* content,
    //                        const std::function<void()>& done) {
    //       // Calls Executor::Schedule(std::function<void()>).
    //       Executor::DefaultExecutor()->Schedule(
    //           turbo::bind_front(&FileReader::BlockingRead, this,
    //                            filename, content, done));
    //     }
    //
    //    private:
    //     void BlockingRead(const std::string& filename, std::string* content,
    //                       const std::function<void()>& done) {
    //       CHECK_OK(file::GetContents(filename, content, {}));
    //       done();
    //     }
    //   };
    //
    // `turbo::bind_front()` stores bound arguments explicitly using the type passed
    // rather than implicitly based on the type accepted by its functor.
    //
    // Example: Binding arguments explicitly.
    //
    //   void LogStringView(std::string_view sv) {
    //     LOG(INFO) << sv;
    //   }
    //
    //   Executor* e = Executor::DefaultExecutor();
    //   std::string s = "hello";
    //   std::string_view sv = s;
    //
    //   // turbo::bind_front(LogStringView, arg) makes a copy of arg and stores it.
    //   e->Schedule(turbo::bind_front(LogStringView, sv)); // ERROR: dangling
    //                                                     // std::string_view.
    //
    //   e->Schedule(turbo::bind_front(LogStringView, s));  // OK: stores a copy of
    //                                                     // s.
    //
    // To store some of the arguments passed to `turbo::bind_front()` by reference,
    //  use std::ref()` and `std::cref()`.
    //
    // Example: Storing some of the bound arguments by reference.
    //
    //   class Service {
    //    public:
    //     void Serve(const Request& req, std::function<void()>* done) {
    //       // The request protocol buffer won't be deleted until done is called.
    //       // It's safe to store a reference to it inside the functor.
    //       Executor::DefaultExecutor()->Schedule(
    //           turbo::bind_front(&Service::BlockingServe, this, std::cref(req),
    //           done));
    //     }
    //
    //    private:
    //     void BlockingServe(const Request& req, std::function<void()>* done);
    //   };
    //
    // Example: Storing bound arguments by reference.
    //
    //   void Print(const std::string& a, const std::string& b) {
    //     std::cerr << a << b;
    //   }
    //
    //   std::string hi = "Hello, ";
    //   std::vector<std::string> names = {"Chuk", "Gek"};
    //   // Doesn't copy hi.
    //   for_each(names.begin(), names.end(),
    //            turbo::bind_front(Print, std::ref(hi)));
    //
    //   // DO NOT DO THIS: the functor may outlive "hi", resulting in
    //   // dangling references.
    //   foo->DoInFuture(turbo::bind_front(Print, std::ref(hi), "Guest"));  // BAD!
    //   auto f = turbo::bind_front(Print, std::ref(hi), "Guest"); // BAD!
    //
    // Example: Storing reference-like types.
    //
    //   void Print(std::string_view a, const std::string& b) {
    //     std::cerr << a << b;
    //   }
    //
    //   std::string hi = "Hello, ";
    //   // Copies "hi".
    //   turbo::bind_front(Print, hi)("Chuk");
    //
    //   // Compile error: std::reference_wrapper<const string> is not implicitly
    //   // convertible to std::string_view.
    //   // turbo::bind_front(Print, std::cref(hi))("Chuk");
    //
    //   // Doesn't copy "hi".
    //   turbo::bind_front(Print, std::string_view(hi))("Chuk");
    //
#if defined(__cpp_lib_bind_front) && __cpp_lib_bind_front >= 201907L
    using std::bind_front;
#else   // defined(__cpp_lib_bind_front) && __cpp_lib_bind_front >= 201907L

    template<class F, class... BoundArgs>
    constexpr functional_internal::bind_front_t<F, BoundArgs...> bind_front(
            F &&func, BoundArgs &&... args) {
        return functional_internal::bind_front_t<F, BoundArgs...>(
                std::in_place, std::forward<F>(func),
                std::forward<BoundArgs>(args)...);
    }

#endif  // defined(__cpp_lib_bind_front) && __cpp_lib_bind_front >= 201907L

}  // namespace turbo

#endif  // TURBO_FUNCTIONAL_BIND_FRONT_H_
