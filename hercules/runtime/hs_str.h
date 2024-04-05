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

#ifndef HERCULES_HS_STR_H_
#define HERCULES_HS_STR_H_

#include <cstdint>
#include <string>

typedef int64_t hs_int_t;

struct hs_str_t {
    hs_int_t len;
    char *str;
};

hs_str_t hs_string_conv(const std::string &s);

#endif // HERCULES_HS_STR_H_
