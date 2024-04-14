// Copyright 2023 The titan-search Authors.
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

#ifndef TURBO_FLAGS_FLAG_H_
#define TURBO_FLAGS_FLAG_H_

#include <string>
#include <type_traits>
#include "turbo/platform/port.h"
#include "turbo/flags/config.h"
#include "turbo/flags/internal/flag.h"
#include "turbo/flags/internal/registry.h"
#include "turbo/strings/string_view.h"

namespace turbo {


// Flag
//
// An `turbo::Flag` holds a command-line flag value, providing a runtime
// parameter to a binary. Such flags should be defined in the global namespace
// and (preferably) in the module containing the binary's `main()` function.
//
// You should not construct and cannot use the `turbo::Flag` type directly;
// instead, you should declare flags using the `TURBO_DECLARE_FLAG()` macro
// within a header file, and define your flag using `TURBO_FLAG()` within your
// header's associated `.cc` file. Such flags will be named `FLAGS_name`.
//
// Example:
//
//    .h file
//
//      // Declares usage of a flag named "FLAGS_count"
//      TURBO_DECLARE_FLAG(int, count);
//
//    .cc file
//
//      // Defines a flag named "FLAGS_count" with a default `int` value of 0.
//      TURBO_FLAG(int, count, 0, "Count of items to process");
//
// No public methods of `turbo::Flag<T>` are part of the Turbo Flags API.
//
// For type support of Turbo Flags, see the marshalling.h header file, which
// discusses supported standard types, optional flags, and additional turbo
// type support.
#if !defined(_MSC_VER) || defined(__clang__)
    template<typename T>
    using Flag = flags_internal::Flag<T>;
#else
#include "turbo/flags/internal/flag_msvc.inc"
#endif

    // get_flag()
    //
    // Returns the value (of type `T`) of an `turbo::Flag<T>` instance, by value. Do
    // not construct an `turbo::Flag<T>` directly and call `turbo::get_flag()`;
    // instead, refer to flag's constructed variable name (e.g. `FLAGS_name`).
    // Because this function returns by value and not by reference, it is
    // thread-safe, but note that the operation may be expensive; as a result, avoid
    // `turbo::get_flag()` within any tight loops.
    //
    // Example:
    //
    //   // FLAGS_count is a Flag of type `int`
    //   int my_count = turbo::get_flag(FLAGS_count);
    //
    //   // FLAGS_firstname is a Flag of type `std::string`
    //   std::string first_name = turbo::get_flag(FLAGS_firstname);
    template<typename T>
    [[nodiscard]] T get_flag(const turbo::Flag<T> &flag) {
        return flags_internal::FlagImplPeer::InvokeGet<T>(flag);
    }

    // set_flag()
    //
    // Sets the value of an `turbo::Flag` to the value `v`. Do not construct an
    // `turbo::Flag<T>` directly and call `turbo::set_flag()`; instead, use the
    // flag's variable name (e.g. `FLAGS_name`). This function is
    // thread-safe, but is potentially expensive. Avoid setting flags in general,
    // but especially within performance-critical code.
    template<typename T>
    void set_flag(turbo::Flag<T> *flag, const T &v) {
        flags_internal::FlagImplPeer::InvokeSet(*flag, v);
    }

    // Overload of `set_flag()` to allow callers to pass in a value that is
    // convertible to `T`. E.g., use this overload to pass a "const char*" when `T`
    // is `std::string`.
    template<typename T, typename V>
    void set_flag(turbo::Flag<T> *flag, const V &v) {
        T value(v);
        flags_internal::FlagImplPeer::InvokeSet(*flag, value);
    }

    // get_flag_reflection_handle()
    //
    // Returns the reflection handle corresponding to specified turbo Flag
    // instance. Use this handle to access flag's reflection information, like name,
    // location, default value etc.
    //
    // Example:
    //
    //   std::string = turbo::get_flag_reflection_handle(FLAGS_count).default_value();

    template<typename T>
    const CommandLineFlag &get_flag_reflection_handle(const turbo::Flag<T> &f) {
        return flags_internal::FlagImplPeer::InvokeReflect(f);
    }


}  // namespace turbo


// TURBO_FLAG()
//
// This macro defines an `turbo::Flag<T>` instance of a specified type `T`:
//
//   TURBO_FLAG(T, name, default_value, help);
//
// where:
//
//   * `T` is a supported flag type (see the list of types in `marshalling.h`),
//   * `name` designates the name of the flag (as a global variable
//     `FLAGS_name`),
//   * `default_value` is an expression holding the default value for this flag
//     (which must be implicitly convertible to `T`),
//   * `help` is the help text, which can also be an expression.
//
// This macro expands to a flag named 'FLAGS_name' of type 'T':
//
//   turbo::Flag<T> FLAGS_name = ...;
//
// Note that all such instances are created as global variables.
//
// For `TURBO_FLAG()` values that you wish to expose to other translation units,
// it is recommended to define those flags within the `.cc` file associated with
// the header where the flag is declared.
//
// Note: do not construct objects of type `turbo::Flag<T>` directly. Only use the
// `TURBO_FLAG()` macro for such construction.
#define TURBO_FLAG(Type, name, default_value, help) \
  TURBO_FLAG_IMPL(Type, name, default_value, help)

// TURBO_FLAG().on_update()
//
// Defines a flag of type `T` with a callback attached:
//
//   TURBO_FLAG(T, name, default_value, help).on_update(callback);
//
// `callback` should be convertible to `void (*)()`.
//
// After any setting of the flag value, the callback will be called at least
// once. A rapid sequence of changes may be merged together into the same
// callback. No concurrent calls to the callback will be made for the same
// flag. Callbacks are allowed to read the current value of the flag but must
// not mutate that flag.
//
// The update mechanism guarantees "eventual consistency"; if the callback
// derives an auxiliary data structure from the flag value, it is guaranteed
// that eventually the flag value and the derived data structure will be
// consistent.
//
// Note: TURBO_FLAG.on_update() does not have a public definition. Hence, this
// comment serves as its API documentation.

// -----------------------------------------------------------------------------
// Implementation details below this section
// -----------------------------------------------------------------------------

// TURBO_FLAG_IMPL macro definition conditional on TURBO_FLAGS_STRIP_NAMES
#if !defined(_MSC_VER) || defined(__clang__)
#define TURBO_FLAG_IMPL_FLAG_PTR(flag) flag
#define TURBO_FLAG_IMPL_HELP_ARG(name)                      \
  turbo::flags_internal::HelpArg<TurboFlagHelpGenFor##name>( \
      FLAGS_help_storage_##name)
#define TURBO_FLAG_IMPL_DEFAULT_ARG(Type, name) \
  turbo::flags_internal::DefaultArg<Type, TurboFlagDefaultGenFor##name>(0)
#else
#define TURBO_FLAG_IMPL_FLAG_PTR(flag) flag.GetImpl()
#define TURBO_FLAG_IMPL_HELP_ARG(name) &TurboFlagHelpGenFor##name::NonConst
#define TURBO_FLAG_IMPL_DEFAULT_ARG(Type, name) &TurboFlagDefaultGenFor##name::Gen
#endif

#if TURBO_FLAGS_STRIP_NAMES
#define TURBO_FLAG_IMPL_FLAGNAME(txt) ""
#define TURBO_FLAG_IMPL_FILENAME() ""
#define TURBO_FLAG_IMPL_REGISTRAR(T, flag)                                      \
  turbo::flags_internal::FlagRegistrar<T, false>(TURBO_FLAG_IMPL_FLAG_PTR(flag), \
                                                nullptr)
#else
#define TURBO_FLAG_IMPL_FLAGNAME(txt) txt
#define TURBO_FLAG_IMPL_FILENAME() __FILE__
#define TURBO_FLAG_IMPL_REGISTRAR(T, flag)                                     \
  turbo::flags_internal::FlagRegistrar<T, true>(TURBO_FLAG_IMPL_FLAG_PTR(flag), \
                                               __FILE__)
#endif

// TURBO_FLAG_IMPL macro definition conditional on TURBO_FLAGS_STRIP_HELP

#if TURBO_FLAGS_STRIP_HELP
#define TURBO_FLAG_IMPL_FLAGHELP(txt) turbo::flags_internal::kStrippedFlagHelp
#else
#define TURBO_FLAG_IMPL_FLAGHELP(txt) txt
#endif

// TurboFlagHelpGenFor##name is used to encapsulate both immediate (method Const)
// and lazy (method NonConst) evaluation of help message expression. We choose
// between the two via the call to HelpArg in turbo::Flag instantiation below.
// If help message expression is constexpr evaluable compiler will optimize
// away this whole struct.
// TODO(rogeeff): place these generated structs into local namespace and apply
// TURBO_INTERNAL_UNIQUE_SHORT_NAME.
// TODO(rogeeff): Apply __attribute__((nodebug)) to FLAGS_help_storage_##name
#define TURBO_FLAG_IMPL_DECLARE_HELP_WRAPPER(name, txt)                       \
  struct TurboFlagHelpGenFor##name {                                          \
    /* The expression is run in the caller as part of the   */               \
    /* default value argument. That keeps temporaries alive */               \
    /* long enough for NonConst to work correctly.          */               \
    static constexpr std::string_view Value(                                \
        std::string_view turbo_flag_help = TURBO_FLAG_IMPL_FLAGHELP(txt)) {   \
      return turbo_flag_help;                                                 \
    }                                                                        \
    static std::string NonConst() { return std::string(Value()); }           \
  };                                                                         \
  constexpr auto FLAGS_help_storage_##name TURBO_INTERNAL_UNIQUE_SMALL_NAME() \
      TURBO_ATTRIBUTE_SECTION_VARIABLE(flags_help_cold) =                     \
          turbo::flags_internal::HelpStringAsArray<TurboFlagHelpGenFor##name>( \
              0);

#define TURBO_FLAG_IMPL_DECLARE_DEF_VAL_WRAPPER(name, Type, default_value)     \
  struct TurboFlagDefaultGenFor##name {                                        \
    Type value = turbo::flags_internal::InitDefaultValue<Type>(default_value); \
    static void Gen(void* turbo_flag_default_loc) {                            \
      new (turbo_flag_default_loc) Type(TurboFlagDefaultGenFor##name{}.value);  \
    }                                                                         \
  };

// TURBO_FLAG_IMPL
//
// Note: Name of registrar object is not arbitrary. It is used to "grab"
// global name for FLAGS_no<flag_name> symbol, thus preventing the possibility
// of defining two flags with names foo and nofoo.
#define TURBO_FLAG_IMPL(Type, name, default_value, help)                       \
  extern ::turbo::Flag<Type> FLAGS_##name;                                     \
  namespace turbo /* block flags in namespaces */ {}                           \
  TURBO_FLAG_IMPL_DECLARE_DEF_VAL_WRAPPER(name, Type, default_value)           \
  TURBO_FLAG_IMPL_DECLARE_HELP_WRAPPER(name, help)                             \
  TURBO_CONST_INIT turbo::Flag<Type> FLAGS_##name{                              \
      TURBO_FLAG_IMPL_FLAGNAME(#name), TURBO_FLAG_IMPL_FILENAME(),              \
      TURBO_FLAG_IMPL_HELP_ARG(name), TURBO_FLAG_IMPL_DEFAULT_ARG(Type, name)}; \
  extern turbo::flags_internal::FlagRegistrarEmpty FLAGS_no##name;             \
  turbo::flags_internal::FlagRegistrarEmpty FLAGS_no##name =                   \
      TURBO_FLAG_IMPL_REGISTRAR(Type, FLAGS_##name)

// TURBO_RETIRED_FLAG
//
// Designates the flag (which is usually pre-existing) as "retired." A retired
// flag is a flag that is now unused by the program, but may still be passed on
// the command line, usually by production scripts. A retired flag is ignored
// and code can't access it at runtime.
//
// This macro registers a retired flag with given name and type, with a name
// identical to the name of the original flag you are retiring. The retired
// flag's type can change over time, so that you can retire code to support a
// custom flag type.
//
// This macro has the same signature as `TURBO_FLAG`. To retire a flag, simply
// replace an `TURBO_FLAG` definition with `TURBO_RETIRED_FLAG`, leaving the
// arguments unchanged (unless of course you actually want to retire the flag
// type at this time as well).
//
// `default_value` is only used as a double check on the type. `explanation` is
// unused.
// TODO(rogeeff): replace RETIRED_FLAGS with FLAGS once forward declarations of
// retired flags are cleaned up.
#define TURBO_RETIRED_FLAG(type, name, default_value, explanation)      \
  static turbo::flags_internal::RetiredFlag<type> RETIRED_FLAGS_##name; \
  TURBO_MAYBE_UNUSED static const auto RETIRED_FLAGS_REG_##name =   \
      (RETIRED_FLAGS_##name.Retire(#name),                             \
       ::turbo::flags_internal::FlagRegistrarEmpty{})

#endif  // TURBO_FLAGS_FLAG_H_
