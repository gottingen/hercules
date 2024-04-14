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
#include "turbo/hash/hash_engine.h"

namespace turbo {

    std::map<std::string, hash_engine_type> engine_map = std::map<std::string, hash_engine_type>{
            {"bytes", hash_engine_type::bytes_hash},
            {"m3", hash_engine_type::m3_hash},
            {"xx", hash_engine_type::xx_hash}
    };
}  // namespace turbo