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


#ifndef TURBO_TOOLS_CONTEXT_H_
#define TURBO_TOOLS_CONTEXT_H_

#include <string>
#include "turbo/hash/hash_engine.h"

namespace turbo::tools {

    struct Context {
        static Context &get_instance() {
            static Context instance;
            return instance;
        }

        bool verbose = false;
        bool version = false;
        std::string hash_string;
        turbo::hash_engine_type engine{turbo::hash_engine_type::bytes_hash};
    };
}  // namespace turbo::tools
#endif  // TURBO_TOOLS_CONTEXT_H_
