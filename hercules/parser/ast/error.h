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

#include <stdexcept>
#include <string>
#include <vector>

/**
 * WARNING: do not include anything else in this file, especially format.h
 * peglib.h uses this file. However, it is not compatible with format.h
 * (and possibly some other includes). Their inclusion will result in a succesful
 * compilation but extremely weird behaviour and hard-to-debug crashes (it seems that
 * some parts of peglib conflict with format.h in a weird way---further investigation
 * needed).
 */

namespace hercules {
    struct SrcInfo {
        std::string file;
        int line;
        int col;
        int len;
        int id; /// used to differentiate different instances

        SrcInfo(std::string file, int line, int col, int len)
                : file(std::move(file)), line(line), col(col), len(len), id(0) {
            static int nextId = 0;
            id = nextId++;
        };

        SrcInfo() : SrcInfo("", 0, 0, 0) {}

        bool operator==(const SrcInfo &src) const { return id == src.id; }
    };

} // namespace hercules

namespace hercules::exc {

    /**
     * Parser error exception.
     * Used for parsing, transformation and type-checking errors.
     */
    class ParserException : public std::runtime_error {
    public:
        /// These vectors (stacks) store an error stack-trace.
        std::vector<SrcInfo> locations;
        std::vector<std::string> messages;
        int errorCode = -1;

    public:
        ParserException(int errorCode, const std::string &msg, const SrcInfo &info) noexcept
                : std::runtime_error(msg), errorCode(errorCode) {
            messages.push_back(msg);
            locations.push_back(info);
        }

        ParserException() noexcept: std::runtime_error("") {}

        ParserException(int errorCode, const std::string &msg) noexcept
                : ParserException(errorCode, msg, {}) {}

        explicit ParserException(const std::string &msg) noexcept
                : ParserException(-1, msg, {}) {}

        ParserException(const ParserException &e) noexcept
                : std::runtime_error(e), locations(e.locations), messages(e.messages),
                  errorCode(e.errorCode) {};

        /// Add an error message to the current stack trace
        void trackRealize(const std::string &msg, const SrcInfo &info) {
            locations.push_back(info);
            messages.push_back("during the realization of " + msg);
        }

        /// Add an error message to the current stack trace
        void track(const std::string &msg, const SrcInfo &info) {
            locations.push_back(info);
            messages.push_back(msg);
        }
    };

} // namespace hercules::exc
