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
// Created by jeff on 24-2-27.
//

#include <hercules/runtime/hs_str.h>
#include <hercules/runtime/lib.h>
#include <cstring>

hs_str_t hs_string_conv(const std::string &s) {
    auto n = s.size();
    if(n == 0) {
        return {0, nullptr};
    }
    auto *p = (char *) hs_alloc_atomic(n);
    ::memcpy(p, s.data(), n);
    return {(hs_int_t) n, p};
}