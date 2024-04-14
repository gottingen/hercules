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
//

#ifndef TURBO_PLATFORM_OPTIONS_H_
#define TURBO_PLATFORM_OPTIONS_H_


#define TURBO_OPTION_USE_INLINE_NAMESPACE 0
#define TURBO_OPTION_INLINE_NAMESPACE_NAME head

// TURBO_OPTION_HARDENED
//
// This option enables a "hardened" build in release mode (in this context,
// release mode is defined as a build where the `NDEBUG` macro is defined).
//
// A value of 0 means that "hardened" mode is not enabled.
//
// A value of 1 means that "hardened" mode is enabled.
//
// Hardened builds have additional security checks enabled when `NDEBUG` is
// defined. Defining `NDEBUG` is normally used to turn `assert()` macro into a
// no-op, as well as disabling other bespoke program consistency checks. By
// defining TURBO_OPTION_HARDENED to 1, a select set of checks remain enabled in
// release mode. These checks guard against programming errors that may lead to
// security vulnerabilities. In release mode, when one of these programming
// errors is encountered, the program will immediately abort, possibly without
// any attempt at logging.
//
// The checks enabled by this option are not free; they do incur runtime cost.
//
// The checks enabled by this option are always active when `NDEBUG` is not
// defined, even in the case when TURBO_OPTION_HARDENED is defined to 0. The
// checks enabled by this option may abort the program in a different way and
// log additional information when `NDEBUG` is not defined.
#ifndef TURBO_OPTION_HARDENED
#define TURBO_OPTION_HARDENED 0
#endif

#ifndef TURBO_OPTION_DEBUG
#define TURBO_OPTION_DEBUG 0
#endif

#ifndef TURBO_OPTION_LOGGING_NO_SOURCE_LOC
#define TLOG_SOURCE_LOC 1
#else
#define TLOG_SOURCE_LOC 0
#endif
#endif  // TURBO_PLATFORM_OPTIONS_H_
