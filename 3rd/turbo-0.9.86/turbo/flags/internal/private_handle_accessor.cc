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

#include "turbo/flags/internal/private_handle_accessor.h"

#include <memory>
#include <string>
#include "turbo/platform/port.h"
#include "turbo/flags/commandlineflag.h"
#include "turbo/flags/internal/commandlineflag.h"
#include "turbo/strings/string_view.h"

namespace turbo::flags_internal {
    FlagFastTypeId PrivateHandleAccessor::type_id(const CommandLineFlag &flag) {
        return flag.type_id();
    }

    std::unique_ptr<FlagStateInterface> PrivateHandleAccessor::save_state(
            CommandLineFlag &flag) {
        return flag.save_state();
    }

    bool PrivateHandleAccessor::is_specified_on_command_line(
            const CommandLineFlag &flag) {
        return flag.is_specified_on_command_line();
    }

    bool PrivateHandleAccessor::validate_input_value(const CommandLineFlag &flag,
                                                   std::string_view value) {
        return flag.validate_input_value(value);
    }

    void PrivateHandleAccessor::check_default_value_parsing_roundtrip(
            const CommandLineFlag &flag) {
        flag.check_default_value_parsing_roundtrip();
    }

    bool PrivateHandleAccessor::parse_from(CommandLineFlag &flag,
                                          std::string_view value,
                                          flags_internal::FlagSettingMode set_mode,
                                          flags_internal::ValueSource source,
                                          std::string &error) {
        return flag.parse_from(value, set_mode, source, error);
    }

}  // namespace turbo::flags_internal

