// Copyright 2022 The Turbo Authors.
//
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

#ifndef TURBO_FILES_SYS_RANDOM_READ_FILE_H_
#define TURBO_FILES_SYS_RANDOM_READ_FILE_H_

#include "turbo/status/result_status.h"
#include "turbo/files/internal/filesystem.h"
#include "turbo/platform/port.h"
#include "turbo/files/file_event_listener.h"
#include "turbo/files/file_option.h"
#include "turbo/files/fwd.h"

namespace turbo {

    /**
     * @ingroup turbo_files_read_file
     * @brief RandomReadFile is a file io utility class.
     * eg.
     * @code
     *    RandomReadFile file;
     *    auto rs = file.open("test.txt");
     *    if (!rs.ok()) {
     *        std::cout << rs.status().message() << std::endl;
     *        return;
     *        // or throw exception.
     *    }
     *    std::string content;
     *    // read all content.
     *    rs = file.read(0, &content);
     *    if (!rs.ok()) {
     *        std::cout << rs.status().message() << std::endl;
     *        // or throw exception.
     *        return;
     *    }
     *    std::cout << content << std::endl;
     *    // read 10 bytes from offset 10.
     *    rs = file.read(10, &content, 10);
     *    if (!rs.ok()) {
     *        std::cout << rs.status().message() << std::endl;
     *        // or throw exception.
     *        return;
     *    }
     *    std::cout << content << std::endl;
     *    // close file or use RAII.
     *    file.close();
     * @endcode
     */
    class RandomReadFile :public  RandomAccessFileReader {
    public:
        RandomReadFile() = default;

        explicit RandomReadFile(const FileEventListener &listener);

        ~RandomReadFile() override;

        /**
         * @brief open file with path and option specified by user.
         *        If the file does not exist, it will be returned with an error.
         * @param path file path
         * @param option file option
         */
        [[nodiscard]] turbo::Status open(const turbo::filesystem::path &path, const turbo::OpenOption &option = kDefaultReadOption) noexcept override;

        /**
         * @brief read file content from offset to the specified length.
         * @param offset [input] file offset
         * @param content [output] file content, can not be nullptr.
         * @param n [input] read length, default is npos, which means read all. If the length is greater than the file
         *          size, the file content will be read from offset to the end of the file.
         * @return the length of the file content read and the status of the operation.
         */
        [[nodiscard]] turbo::ResultStatus<size_t> read(off_t offset, std::string *content, size_t n = kInfiniteFileSize) override;

        /**
         * @brief read file content from offset to the specified length.
         * @param offset [input] file offset
         * @param buf [output] file content, can not be nullptr.
         * @param n [input] read length, default is npos, which means read all. If the length is greater than the file
         *          size, the file content will be read from offset to the end of the file.
         * @return the length of the file content read and the status of the operation.
         */
        [[nodiscard]] turbo::ResultStatus<size_t> read(off_t offset, turbo::IOBuf *buf, size_t n = kInfiniteFileSize) override;

        /**
         * @brief read file content from offset to the specified length.
         * @param offset [input] file offset
         * @param buff [output] file content, can not be nullptr.
         * @param len [input] read length, and buff size must be greater than len. if from offset to the end of the file
         *            is less than len, the file content will be read from offset to the end of the file. the size of read
         *            content is the minimum of len and the file size from offset to the end of the file and will be returned
         *            in the result.
         * @return the length of the file content read and the status of the operation.
         */
        [[nodiscard]] turbo::ResultStatus<size_t> read(off_t offset, void *buff, size_t len) override;

        /**
         * @brief close file.
         */
        void close() override;

    private:
        // no lint
        TURBO_NON_COPYABLE(RandomReadFile);
        int        _fd;
        turbo::filesystem::path _file_path;
        turbo::OpenOption _option;
        FileEventListener _listener;
    };

} // namespace turbo

#endif  // TURBO_FILES_SYS_RANDOM_READ_FILE_H_
