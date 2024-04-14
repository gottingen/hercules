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

#include "turbo/flags/internal/program_name.h"

#include <string>
#include <mutex>
#include "turbo/platform/port.h"
#include "turbo/base/const_init.h"
#include "turbo/platform/thread_annotations.h"
#include "turbo/flags/internal/path_util.h"
#include "turbo/strings/string_view.h"

namespace turbo {

    namespace flags_internal {

        static std::mutex program_name_guard;
        TURBO_CONST_INIT static std::string *program_name
                TURBO_GUARDED_BY(program_name_guard) = nullptr;

        std::string program_invocation_name() {
            std::unique_lock l(program_name_guard);

            return program_name ? *program_name : "UNKNOWN";
        }

        std::string short_program_invocation_name() {
            std::unique_lock l(program_name_guard);

            return program_name ? std::string(flags_internal::Basename(*program_name))
                                : "UNKNOWN";
        }

        void set_program_invocation_name(std::string_view prog_name_str) {
            std::unique_lock l(program_name_guard);

            if (!program_name)
                program_name = new std::string(prog_name_str);
            else
                program_name->assign(prog_name_str.data(), prog_name_str.size());
        }

    }  // namespace flags_internal

}  // namespace turbo
