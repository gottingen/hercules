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

#include "turbo/log/common.h"
#include <unordered_map>

namespace turbo::tlog {
namespace cfg {
namespace helpers {
//
// Init levels from given string
//
// Examples:
//
// set global level to debug: "debug"
// turn off all logging except for logger1: "off,logger1=debug"
// turn off all logging except for logger1 and logger2: "off,logger1=debug,logger2=info"
//
TURBO_DLL void load_levels(const std::string &txt);
} // namespace helpers

} // namespace cfg
} // namespace turbo::tlog
