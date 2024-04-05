// Copyright 2024 The Elastic-AI Authors.
// part of Elastic AI Search
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
#include <collie/log/cfg/helpers.h>
#include <collie/log/details/os.h>
#include <collie/log/details/registry.h>

//
// Init levels and patterns from env variables CLOG_LEVEL
// Inspired from Rust's "env_logger" crate (https://crates.io/crates/env_logger).
// Note - fallback to "info" level on unrecognized levels
//
// Examples:
//
// set global level to debug:
// export CLOG_LEVEL=debug
//
// turn off all logging except for logger1:
// export CLOG_LEVEL="*=off,logger1=debug"
//

// turn off all logging except for logger1 and logger2:
// export CLOG_LEVEL="off,logger1=debug,logger2=info"

namespace clog {
namespace cfg {
inline void load_env_levels() {
    auto env_val = details::os::getenv("CLOG_LEVEL");
    if (!env_val.empty()) {
        helpers::load_levels(env_val);
    }
}

}  // namespace cfg
}  // namespace clog
