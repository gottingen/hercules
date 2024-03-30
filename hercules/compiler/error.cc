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

#include <hercules/compiler/error.h>

namespace hercules::error {

    char ParserErrorInfo::ID = 0;

    char RuntimeErrorInfo::ID = 0;

    char PluginErrorInfo::ID = 0;

    char IOErrorInfo::ID = 0;

    void raise_error(const char *format) { throw exc::ParserException(format); }

    void raise_error(int e, const ::hercules::SrcInfo &info, const char *format) {
        throw exc::ParserException(e, format, info);
    }

    void raise_error(int e, const ::hercules::SrcInfo &info, const std::string &format) {
        throw exc::ParserException(e, format, info);
    }

} // namespace hercules::error
