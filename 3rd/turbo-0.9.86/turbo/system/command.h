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

#ifndef TURBO_SYSTEM_COMMAND_H_
#define TURBO_SYSTEM_COMMAND_H_

#include <iostream>
#include <string>

namespace turbo {

    int read_command_output(std::ostream& os, const char* cmd);

    bool self_command_line(std::string &cmd, bool with_args, size_t max_len = 1024);
}  // namespace turbo

#endif // TURBO_SYSTEM_COMMAND_H_
