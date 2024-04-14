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
// File: thread_annotations.h
// -----------------------------------------------------------------------------
//
// This header file contains macro definitions for thread safety annotations
// that allow developers to document the locking policies of multi-threaded
// code. The annotations can also help program analysis tools to identify
// potential thread safety issues.
//
// These annotations are implemented using compiler attributes. Using the macros
// defined here instead of raw attributes allow for portability and future
// compatibility.
//
// When referring to mutexes in the arguments of the attributes, you should
// use variable names or more complex expressions (e.g. my_object->mutex_)
// that evaluate to a concrete mutex object whenever possible. If the mutex
// you want to refer to is not in scope, you may use a member pointer
// (e.g. &MyClass::mutex_) to refer to a mutex in some (unknown) object.

#ifndef TURBO_PLATFORM_THREAD_ANNOTATIONS_H_
#define TURBO_PLATFORM_THREAD_ANNOTATIONS_H_

#include "turbo/platform/port.h"
#include "turbo/platform/internal/thread_annotations.h"  // IWYU pragma: export

// TURBO_GUARDED_BY()
//
// Documents if a shared field or global variable needs to be protected by a
// mutex. TURBO_GUARDED_BY() allows the user to specify a particular mutex that
// should be held when accessing the annotated variable.
//
// Although this annotation (and TURBO_PT_GUARDED_BY, below) cannot be applied to
// local variables, a local variable and its associated mutex can often be
// combined into a small class or struct, thereby allowing the annotation.
//
// Example:
//
//   class Foo {
//     Mutex mu_;
//     int p1_ TURBO_GUARDED_BY(mu_);
//     ...
//   };
#if TURBO_HAVE_ATTRIBUTE(guarded_by)
#define TURBO_GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define TURBO_GUARDED_BY(x)
#endif

// TURBO_PT_GUARDED_BY()
//
// Documents if the memory location pointed to by a pointer should be guarded
// by a mutex when dereferencing the pointer.
//
// Example:
//   class Foo {
//     Mutex mu_;
//     int *p1_ TURBO_PT_GUARDED_BY(mu_);
//     ...
//   };
//
// Note that a pointer variable to a shared memory location could itself be a
// shared variable.
//
// Example:
//
//   // `q_`, guarded by `mu1_`, points to a shared memory location that is
//   // guarded by `mu2_`:
//   int *q_ TURBO_GUARDED_BY(mu1_) TURBO_PT_GUARDED_BY(mu2_);
#if TURBO_HAVE_ATTRIBUTE(pt_guarded_by)
#define TURBO_PT_GUARDED_BY(x) __attribute__((pt_guarded_by(x)))
#else
#define TURBO_PT_GUARDED_BY(x)
#endif

// TURBO_ACQUIRED_AFTER() / TURBO_ACQUIRED_BEFORE()
//
// Documents the acquisition order between locks that can be held
// simultaneously by a thread. For any two locks that need to be annotated
// to establish an acquisition order, only one of them needs the annotation.
// (i.e. You don't have to annotate both locks with both TURBO_ACQUIRED_AFTER
// and TURBO_ACQUIRED_BEFORE.)
//
// As with TURBO_GUARDED_BY, this is only applicable to mutexes that are shared
// fields or global variables.
//
// Example:
//
//   Mutex m1_;
//   Mutex m2_ TURBO_ACQUIRED_AFTER(m1_);
#if TURBO_HAVE_ATTRIBUTE(acquired_after)
#define TURBO_ACQUIRED_AFTER(...) __attribute__((acquired_after(__VA_ARGS__)))
#else
#define TURBO_ACQUIRED_AFTER(...)
#endif

#if TURBO_HAVE_ATTRIBUTE(acquired_before)
#define TURBO_ACQUIRED_BEFORE(...) __attribute__((acquired_before(__VA_ARGS__)))
#else
#define TURBO_ACQUIRED_BEFORE(...)
#endif

// TURBO_EXCLUSIVE_LOCKS_REQUIRED() / TURBO_SHARED_LOCKS_REQUIRED()
//
// Documents a function that expects a mutex to be held prior to entry.
// The mutex is expected to be held both on entry to, and exit from, the
// function.
//
// An exclusive lock allows read-write access to the guarded data member(s), and
// only one thread can acquire a lock exclusively at any one time. A shared lock
// allows read-only access, and any number of threads can acquire a shared lock
// concurrently.
//
// Generally, non-const methods should be annotated with
// TURBO_EXCLUSIVE_LOCKS_REQUIRED, while const methods should be annotated with
// TURBO_SHARED_LOCKS_REQUIRED.
//
// Example:
//
//   Mutex mu1, mu2;
//   int a TURBO_GUARDED_BY(mu1);
//   int b TURBO_GUARDED_BY(mu2);
//
//   void foo() TURBO_EXCLUSIVE_LOCKS_REQUIRED(mu1, mu2) { ... }
//   void bar() const TURBO_SHARED_LOCKS_REQUIRED(mu1, mu2) { ... }
#if TURBO_HAVE_ATTRIBUTE(exclusive_locks_required)
#define TURBO_EXCLUSIVE_LOCKS_REQUIRED(...) \
  __attribute__((exclusive_locks_required(__VA_ARGS__)))
#else
#define TURBO_EXCLUSIVE_LOCKS_REQUIRED(...)
#endif

#if TURBO_HAVE_ATTRIBUTE(shared_locks_required)
#define TURBO_SHARED_LOCKS_REQUIRED(...) \
  __attribute__((shared_locks_required(__VA_ARGS__)))
#else
#define TURBO_SHARED_LOCKS_REQUIRED(...)
#endif

// TURBO_LOCKS_EXCLUDED()
//
// Documents the locks that cannot be held by callers of this function, as they
// might be acquired by this function (Turbo's `Mutex` locks are
// non-reentrant).
#if TURBO_HAVE_ATTRIBUTE(locks_excluded)
#define TURBO_LOCKS_EXCLUDED(...) __attribute__((locks_excluded(__VA_ARGS__)))
#else
#define TURBO_LOCKS_EXCLUDED(...)
#endif

// TURBO_LOCK_RETURNED()
//
// Documents a function that returns a mutex without acquiring it.  For example,
// a public getter method that returns a pointer to a private mutex should
// be annotated with TURBO_LOCK_RETURNED.
#if TURBO_HAVE_ATTRIBUTE(lock_returned)
#define TURBO_LOCK_RETURNED(x) __attribute__((lock_returned(x)))
#else
#define TURBO_LOCK_RETURNED(x)
#endif

// TURBO_LOCKABLE
//
// Documents if a class/type is a lockable type (such as the `Mutex` class).
#if TURBO_HAVE_ATTRIBUTE(lockable)
#define TURBO_LOCKABLE __attribute__((lockable))
#else
#define TURBO_LOCKABLE
#endif

// TURBO_SCOPED_LOCKABLE
//
// Documents if a class does RAII locking (such as the `MutexLock` class).
// The constructor should use `LOCK_FUNCTION()` to specify the mutex that is
// acquired, and the destructor should use `UNLOCK_FUNCTION()` with no
// arguments; the analysis will assume that the destructor unlocks whatever the
// constructor locked.
#if TURBO_HAVE_ATTRIBUTE(scoped_lockable)
#define TURBO_SCOPED_LOCKABLE __attribute__((scoped_lockable))
#else
#define TURBO_SCOPED_LOCKABLE
#endif

// TURBO_EXCLUSIVE_LOCK_FUNCTION()
//
// Documents functions that acquire a lock in the body of a function, and do
// not release it.
#if TURBO_HAVE_ATTRIBUTE(exclusive_lock_function)
#define TURBO_EXCLUSIVE_LOCK_FUNCTION(...) \
  __attribute__((exclusive_lock_function(__VA_ARGS__)))
#else
#define TURBO_EXCLUSIVE_LOCK_FUNCTION(...)
#endif

// TURBO_SHARED_LOCK_FUNCTION()
//
// Documents functions that acquire a shared (reader) lock in the body of a
// function, and do not release it.
#if TURBO_HAVE_ATTRIBUTE(shared_lock_function)
#define TURBO_SHARED_LOCK_FUNCTION(...) \
  __attribute__((shared_lock_function(__VA_ARGS__)))
#else
#define TURBO_SHARED_LOCK_FUNCTION(...)
#endif

// TURBO_UNLOCK_FUNCTION()
//
// Documents functions that expect a lock to be held on entry to the function,
// and release it in the body of the function.
#if TURBO_HAVE_ATTRIBUTE(unlock_function)
#define TURBO_UNLOCK_FUNCTION(...) __attribute__((unlock_function(__VA_ARGS__)))
#else
#define TURBO_UNLOCK_FUNCTION(...)
#endif

// TURBO_EXCLUSIVE_TRYLOCK_FUNCTION() / TURBO_SHARED_TRYLOCK_FUNCTION()
//
// Documents functions that try to acquire a lock, and return success or failure
// (or a non-boolean value that can be interpreted as a boolean).
// The first argument should be `true` for functions that return `true` on
// success, or `false` for functions that return `false` on success. The second
// argument specifies the mutex that is locked on success. If unspecified, this
// mutex is assumed to be `this`.
#if TURBO_HAVE_ATTRIBUTE(exclusive_trylock_function)
#define TURBO_EXCLUSIVE_TRYLOCK_FUNCTION(...) \
  __attribute__((exclusive_trylock_function(__VA_ARGS__)))
#else
#define TURBO_EXCLUSIVE_TRYLOCK_FUNCTION(...)
#endif

#if TURBO_HAVE_ATTRIBUTE(shared_trylock_function)
#define TURBO_SHARED_TRYLOCK_FUNCTION(...) \
  __attribute__((shared_trylock_function(__VA_ARGS__)))
#else
#define TURBO_SHARED_TRYLOCK_FUNCTION(...)
#endif

// TURBO_ASSERT_EXCLUSIVE_LOCK() / TURBO_ASSERT_SHARED_LOCK()
//
// Documents functions that dynamically check to see if a lock is held, and fail
// if it is not held.
#if TURBO_HAVE_ATTRIBUTE(assert_exclusive_lock)
#define TURBO_ASSERT_EXCLUSIVE_LOCK(...) \
  __attribute__((assert_exclusive_lock(__VA_ARGS__)))
#else
#define TURBO_ASSERT_EXCLUSIVE_LOCK(...)
#endif

#if TURBO_HAVE_ATTRIBUTE(assert_shared_lock)
#define TURBO_ASSERT_SHARED_LOCK(...) \
  __attribute__((assert_shared_lock(__VA_ARGS__)))
#else
#define TURBO_ASSERT_SHARED_LOCK(...)
#endif

// TURBO_NO_THREAD_SAFETY_ANALYSIS
//
// Turns off thread safety checking within the body of a particular function.
// This annotation is used to mark functions that are known to be correct, but
// the locking behavior is more complicated than the analyzer can handle.
#if TURBO_HAVE_ATTRIBUTE(no_thread_safety_analysis)
#define TURBO_NO_THREAD_SAFETY_ANALYSIS \
  __attribute__((no_thread_safety_analysis))
#else
#define TURBO_NO_THREAD_SAFETY_ANALYSIS
#endif

//------------------------------------------------------------------------------
// Tool-Supplied Annotations
//------------------------------------------------------------------------------

// TURBO_TS_UNCHECKED should be placed around lock expressions that are not valid
// C++ syntax, but which are present for documentation purposes.  These
// annotations will be ignored by the analysis.
#define TURBO_TS_UNCHECKED(x) ""

// TURBO_TS_FIXME is used to mark lock expressions that are not valid C++ syntax.
// It is used by automated tools to mark and disable invalid expressions.
// The annotation should either be fixed, or changed to TURBO_TS_UNCHECKED.
#define TURBO_TS_FIXME(x) ""

// Like TURBO_NO_THREAD_SAFETY_ANALYSIS, this turns off checking within the body
// of a particular function.  However, this attribute is used to mark functions
// that are incorrect and need to be fixed.  It is used by automated tools to
// avoid breaking the build when the analysis is updated.
// Code owners are expected to eventually fix the routine.
#define TURBO_NO_THREAD_SAFETY_ANALYSIS_FIXME TURBO_NO_THREAD_SAFETY_ANALYSIS

// Similar to TURBO_NO_THREAD_SAFETY_ANALYSIS_FIXME, this macro marks a
// TURBO_GUARDED_BY annotation that needs to be fixed, because it is producing
// thread safety warning. It disables the TURBO_GUARDED_BY.
#define TURBO_GUARDED_BY_FIXME(x)

// Disables warnings for a single read operation.  This can be used to avoid
// warnings when it is known that the read is not actually involved in a race,
// but the compiler cannot confirm that.
#define TURBO_TS_UNCHECKED_READ(x) turbo::base_internal::ts_unchecked_read(x)

namespace turbo::base_internal {

    // Takes a reference to a guarded data member, and returns an unguarded
    // reference.
    // Do not use this function directly, use TURBO_TS_UNCHECKED_READ instead.
    template<typename T>
    inline const T &ts_unchecked_read(const T &v) TURBO_NO_THREAD_SAFETY_ANALYSIS {
        return v;
    }

    template<typename T>
    inline T &ts_unchecked_read(T &v) TURBO_NO_THREAD_SAFETY_ANALYSIS {
        return v;
    }

}  // namespace turbo::base_internal

#endif  // TURBO_PLATFORM_THREAD_ANNOTATIONS_H_
