// Copyright 2023 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#pragma once

#include <turbo/log/cfg/helpers.h>
#include "turbo/log/details/registry.h"
#include "turbo/log/details/os.h"
#include "turbo/system/env.h"

//
// Init levels and patterns from env variables TLOG_LEVEL
// Inspired from Rust's "env_logger" crate (https://crates.io/crates/env_logger).
// Note - fallback to "info" level on unrecognized levels
//
// Examples:
//
// set global level to debug:
// export TLOG_LEVEL=debug
//
// turn off all logging except for logger1:
// export TLOG_LEVEL="*=off,logger1=debug"
//

// turn off all logging except for logger1 and logger2:
// export TLOG_LEVEL="off,logger1=debug,logger2=info"

namespace turbo::tlog {
    namespace cfg {
        inline void load_env_levels() {
            auto env_val = turbo::get_env("TLOG_LEVEL");
            if (!env_val.empty()) {
                helpers::load_levels(env_val);
            }
        }

    } // namespace cfg
} // namespace turbo::tlog
