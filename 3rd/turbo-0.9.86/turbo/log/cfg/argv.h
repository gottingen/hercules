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

//
// Init log levels using each argv entry that starts with "TLOG_LEVEL="
//
// set all loggers to debug level:
// example.exe "TLOG_LEVEL=debug"

// set logger1 to trace level
// example.exe "TLOG_LEVEL=logger1=trace"

// turn off all logging except for logger1 and logger2:
// example.exe "TLOG_LEVEL=off,logger1=debug,logger2=info"

namespace turbo::tlog {
namespace cfg {

// search for TLOG_LEVEL= in the args and use it to init the levels
inline void load_argv_levels(int argc, const char **argv)
{
    const std::string tlog_level_prefix = "TLOG_LEVEL=";
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find(tlog_level_prefix) == 0)
        {
            auto levels_string = arg.substr(tlog_level_prefix.size());
            helpers::load_levels(levels_string);
        }
    }
}

inline void load_argv_levels(int argc, char **argv)
{
    load_argv_levels(argc, const_cast<const char **>(argv));
}

} // namespace cfg
} // namespace turbo::tlog
