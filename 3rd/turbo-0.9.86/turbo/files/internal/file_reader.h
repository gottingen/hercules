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

#ifndef TURBO_FILE_INTERNAL_FILE_READER_H_
#define TURBO_FILE_INTERNAL_FILE_READER_H_

#include "turbo/files/internal/filesystem.h"
#include "turbo/files/internal/fwd.h"
#include "turbo/status/result_status.h"
#include "turbo/system/io.h"
#include <string>

namespace turbo {

    class SequentialFileReader {
    public:

        virtual  ~SequentialFileReader()   = default;

        [[nodiscard]] virtual turbo::Status open(const turbo::filesystem::path &path, const turbo::OpenOption &option = kDefaultReadOption) noexcept = 0;

        [[nodiscard]] virtual turbo::Status skip(off_t n) = 0;

        [[nodiscard]] virtual turbo::ResultStatus<size_t> read(void *buff, size_t len) = 0;

        [[nodiscard]] virtual turbo::ResultStatus<size_t> read(std::string *result, size_t len = kInfiniteFileSize) {
            return turbo::make_status(kENOSYS);
        }

        [[nodiscard]] virtual turbo::ResultStatus<size_t> read(turbo::IOBuf *result, size_t len = kInfiniteFileSize) {
            return turbo::make_status(kENOSYS);
        }

        virtual void close() = 0;

        virtual size_t position() const = 0;

        [[nodiscard]] virtual turbo::ResultStatus<bool> is_eof() const = 0;

    };

    class RandomAccessFileReader {
    public:

            virtual  ~RandomAccessFileReader()   = default;

            [[nodiscard]] virtual turbo::Status open(const turbo::filesystem::path &path, const turbo::OpenOption &option = kDefaultReadOption) noexcept = 0;

            [[nodiscard]] virtual turbo::ResultStatus<size_t> read(off_t offset, void *buff, size_t len) = 0;

            [[nodiscard]] virtual turbo::ResultStatus<size_t> read(off_t offset, std::string *result, size_t len = kInfiniteFileSize) {
                return turbo::make_status(kENOSYS);
            }

            [[nodiscard]] virtual turbo::ResultStatus<size_t> read(off_t offset, turbo::IOBuf *result, size_t len = kInfiniteFileSize) {
                return turbo::make_status(kENOSYS);
            }

            virtual void close() = 0;
    };

}  // namespace turbo

#endif  // TURBO_FILE_INTERNAL_FILE_READER_H_
