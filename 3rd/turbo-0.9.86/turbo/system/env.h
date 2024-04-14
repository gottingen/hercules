// Copyright 2023 The Elastic-AI Authors.
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
//
// Created by jeff on 24-1-5.
//

#ifndef TURBO_SYSTEM_ENV_H_
#define TURBO_SYSTEM_ENV_H_

#include <cstdlib>
#include <cstdio>
#include <string>

namespace turbo {

    // Function: get_env
    std::string get_env(const std::string &str);

    // Function: has_env
    bool has_env(const std::string &str);

    class ScopedSetEnv {
    public:
        ScopedSetEnv(const char *var_name, const char *new_value);

        ~ScopedSetEnv();

    private:
        std::string var_name_;
        std::string old_value_;

        // True if the environment variable was initially not set.
        bool was_unset_;
    };

}  // namespace turbo

#endif  // TURBO_SYSTEM_ENV_H_
