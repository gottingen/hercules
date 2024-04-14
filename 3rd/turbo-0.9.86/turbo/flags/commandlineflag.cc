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
#include "turbo/flags/commandlineflag.h"

#include <string>

#include "turbo/platform/port.h"
#include "turbo/flags/internal/commandlineflag.h"
#include "turbo/strings/string_view.h"

namespace turbo {


    bool CommandLineFlag::is_retired() const { return false; }

    bool CommandLineFlag::parse_from(std::string_view value, std::string *error) {
        return parse_from(value, flags_internal::SET_FLAGS_VALUE,
                         flags_internal::kProgrammaticChange, *error);
    }


}  // namespace turbo
