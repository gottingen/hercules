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

#ifndef TURBO_FLAGS_INTERNAL_PROGRAM_NAME_H_
#define TURBO_FLAGS_INTERNAL_PROGRAM_NAME_H_

#include <string>

#include "turbo/platform/port.h"
#include "turbo/strings/string_view.h"

namespace turbo::flags_internal {

    // Returns program invocation name or "UNKNOWN" if `set_program_invocation_name()`
    // is never called. At the moment this is always set to argv[0] as part of
    // library initialization.
    std::string program_invocation_name();

    // Returns base name for program invocation name. For example, if
    //   program_invocation_name() == "a/b/mybinary"
    // then
    //   short_program_invocation_name() == "mybinary"
    std::string short_program_invocation_name();

    // Sets program invocation name to a new value. Should only be called once
    // during program initialization, before any threads are spawned.
    void set_program_invocation_name(std::string_view prog_name_str);

}  // namespace turbo::flags_internal

#endif  // TURBO_FLAGS_INTERNAL_PROGRAM_NAME_H_
