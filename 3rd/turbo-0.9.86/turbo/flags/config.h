// Copyright 2023 The Turbo Authors.
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
#ifndef TURBO_FLAGS_CONFIG_H_
#define TURBO_FLAGS_CONFIG_H_

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>


#include "turbo/flags/app.h"
#include "turbo/flags/config_fwd.h"
#include "turbo/flags/string_tools.h"

// Determine if we should strip string literals from the Flag objects.
// By default we strip string literals on mobile platforms.
#if !defined(TURBO_FLAGS_STRIP_NAMES)

#if defined(__ANDROID__)
#define TURBO_FLAGS_STRIP_NAMES 1

#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
#define TURBO_FLAGS_STRIP_NAMES 1
#elif defined(TARGET_OS_EMBEDDED) && TARGET_OS_EMBEDDED
#define TURBO_FLAGS_STRIP_NAMES 1
#endif  // TARGET_OS_*
#endif

#endif  // !defined(TURBO_FLAGS_STRIP_NAMES)

#if !defined(TURBO_FLAGS_STRIP_NAMES)
// If TURBO_FLAGS_STRIP_NAMES wasn't set on the command line or above,
// the default is not to strip.
#define TURBO_FLAGS_STRIP_NAMES 0
#endif

#if !defined(TURBO_FLAGS_STRIP_HELP)
// By default, if we strip names, we also strip help.
#define TURBO_FLAGS_STRIP_HELP TURBO_FLAGS_STRIP_NAMES
#endif

// These macros represent the "source of truth" for the list of supported
// built-in types.
#define TURBO_FLAGS_INTERNAL_BUILTIN_TYPES(A) \
  A(bool, bool)                              \
  A(short, short)                            \
  A(unsigned short, unsigned_short)          \
  A(int, int)                                \
  A(unsigned int, unsigned_int)              \
  A(long, long)                              \
  A(unsigned long, unsigned_long)            \
  A(long long, long_long)                    \
  A(unsigned long long, unsigned_long_long)  \
  A(double, double)                          \
  A(float, float)

#define TURBO_FLAGS_INTERNAL_SUPPORTED_TYPES(A) \
  TURBO_FLAGS_INTERNAL_BUILTIN_TYPES(A)         \
  A(std::string, std_string)                   \
  A(std::vector<std::string>, std_vector_of_string)

namespace turbo::detail {

    std::string convert_arg_for_ini(const std::string &arg, char stringQuote = '"', char characterQuote = '\'');

    /// Comma separated join, adds quotes if needed
    std::string ini_join(const std::vector<std::string> &args,
                         char sepChar = ',',
                         char arrayStart = '[',
                         char arrayEnd = ']',
                         char stringQuote = '"',
                         char characterQuote = '\'');

    std::vector<std::string> generate_parents(const std::string &section, std::string &name, char parentSeparator);

    /// assuming non default segments do a check on the close and open of the segments in a configItem structure
    void
    checkParentSegments(std::vector<ConfigItem> &output, const std::string &currentSection, char parentSeparator);

}  // namespace turbo::detail

#endif  // TURBO_FLAGS_CONFIG_H_