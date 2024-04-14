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
// Created by jeff on 24-1-9.
//

#ifndef TURBO_FILES_INTERNAL_FILE_WRITER_H_
#define TURBO_FILES_INTERNAL_FILE_WRITER_H_

#include "turbo/files/internal/filesystem.h"
#include "turbo/files/internal/fwd.h"
#include "turbo/status/result_status.h"
#include "turbo/files/file_option.h"
#include "turbo/system/io.h"
#include "turbo/format/format.h"
#include <string_view>


namespace turbo {

    class SequentialFileWriter {
    public:

        virtual  ~SequentialFileWriter() = default;

        [[nodiscard]] virtual turbo::Status open(const turbo::filesystem::path &path,const turbo::OpenOption &option) noexcept = 0;

        [[nodiscard]] virtual  turbo::Status write(const void *buff, size_t len) = 0;

        [[nodiscard]] virtual turbo::Status flush() = 0;

        [[nodiscard]] virtual  turbo::Status write(std::string_view buff) {
            return turbo::make_status(kENOSYS);
        }

        [[nodiscard]] virtual  turbo::Status write(const turbo::IOBuf &buff) {
            return turbo::make_status(kENOSYS);
        }

        [[nodiscard]] virtual turbo::Status truncate(size_t size) = 0;

        template<typename ...Args>
        [[nodiscard]] turbo::Status write_format(off_t offset, const char *fmt, const Args&...args) {
            std::string_view content = turbo::format(fmt, args...);
            return write(content.data(), content.size());
        }

        [[nodiscard]] virtual ResultStatus<size_t> size() const = 0;

        virtual void close() = 0;

    };

    class RandomFileWriter {
    public:

        virtual ~RandomFileWriter() = default;

        [[nodiscard]] virtual turbo::Status open(const turbo::filesystem::path &path, const turbo::OpenOption &option) noexcept = 0;

        [[nodiscard]] virtual  turbo::Status
        write(off_t offset, const void *buff, size_t len, bool truncate = false) = 0;

        [[nodiscard]] virtual turbo::Status flush() = 0;

        [[nodiscard]] virtual turbo::Status
        write(off_t offset, std::string_view buff, bool truncate = false) {
            return turbo::make_status(kENOSYS);
        }

        [[nodiscard]] virtual turbo::Status
        write(off_t offset, const turbo::IOBuf &buff, bool truncate = false) {
            return turbo::make_status(kENOSYS);
        }

        template<typename ...Args>
        [[nodiscard]] turbo::Status write_format(off_t offset, const char *fmt, const Args&...args) {
            std::string_view content = turbo::format(fmt, args...);
            return write(offset, content.data(), content.size());
        }

        [[nodiscard]] virtual turbo::Status truncate(size_t size) = 0;

        virtual ResultStatus<size_t> size() const = 0;

        virtual void close() = 0;

    };

    static constexpr std::string_view kDefaultTempFilePrefix = "temp_file_";

    class TempFileWriter {
    public:

        virtual ~TempFileWriter() = default;

        [[nodiscard]] virtual turbo::Status open(std::string_view prefix = kDefaultTempFilePrefix, std::string_view ext ="", size_t bits = 6) noexcept = 0;

        [[nodiscard]] virtual turbo::Status write(const void *buff, size_t len) = 0;

        [[nodiscard]] virtual turbo::Status flush() = 0;

        [[nodiscard]] virtual turbo::Status truncate(size_t size) = 0;

        [[nodiscard]] virtual turbo::Status write(std::string_view buff) {
            return turbo::make_status(kENOSYS);
        }

        [[nodiscard]] virtual turbo::Status write(const turbo::IOBuf &buff) {
            return turbo::make_status(kENOSYS);
        }

        template<typename ...Args>
        [[nodiscard]] turbo::Status write_format(off_t offset, const char *fmt, const Args&...args) {
            std::string_view content = turbo::format(fmt, args...);
            return write(content.data(), content.size());
        }


        [[nodiscard]] virtual std::string path() const = 0;

        virtual ResultStatus<size_t> size() const = 0;

        virtual void close() = 0;
    };
}  // namespace turbo

#endif  // TURBO_FILES_INTERNAL_FILE_WRITER_H_
