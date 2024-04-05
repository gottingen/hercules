// Copyright 2024 The EA Authors.
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

#include <string>
#include <vector>

namespace hercules::jit {

    class JIT;

    struct JITResult {
        void *result;
        std::string message;

        operator bool() const { return message.empty(); }

        static JITResult success(void *result) { return {result, ""}; }

        static JITResult error(const std::string &message) { return {nullptr, message}; }
    };

    JIT *jitInit(const std::string &name);

    JITResult jitExecutePython(JIT *jit, const std::string &name,
                               const std::vector<std::string> &types,
                               const std::string &pyModule,
                               const std::vector<std::string> &pyVars, void *arg,
                               bool debug);

    JITResult jitExecuteSafe(JIT *jit, const std::string &code, const std::string &file,
                             int line, bool debug);

    std::string getJITLibrary();

} // namespace hercules::jit
