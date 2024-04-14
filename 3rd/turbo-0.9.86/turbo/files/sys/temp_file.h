// Copyright 2023 The titan-search Authors.
// by jeff.li
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

#ifndef TURBO_FILES_TEMP_FILE_H_
#define TURBO_FILES_TEMP_FILE_H_

#include "turbo/platform/port.h"
#include "turbo/files/internal/filesystem.h"
#include <string_view>
#include <string>
#include "turbo/format/format.h"
#include "turbo/status/result_status.h"
#include "turbo/files/fwd.h"
#include "turbo/files/sys/sequential_write_file.h"

namespace turbo {

    // Create a temporary file in current directory, which will be deleted when
    // corresponding temp_file object destructs, typically for unit testing.
    //
    // Usage:
    //   {
    //      temp_file tmpfile;           // A temporay file shall be created
    //      tmpfile.save("some text");  // Write into the temporary file
    //   }
    //   // The temporary file shall be removed due to destruction of tmpfile

    class TempFile : public TempFileWriter {
    public:

        // Create a temporary file in current directory. If |ext| is given,
        // filename will be temp_file_XXXXXX.|ext|, temp_file_XXXXXX otherwise.
        // If temporary file cannot be created, all save*() functions will
        // return -1. If |ext| is too long, filename will be truncated.
        TempFile() = default;

        explicit TempFile(const FileEventListener &listener);

        // The temporary file is removed in destructor.
        ~TempFile() {
            _file.close();
        }

        [[nodiscard]] virtual turbo::Status open(std::string_view prefix = kDefaultTempFilePrefix, std::string_view ext ="", size_t bits = 6) noexcept override;

        turbo::Status write(const void *buf, size_t count) override;

        // Save binary data |buf| (|count| bytes) to file, overwriting existing file.
        // Returns 0 when successful, -1 otherwise.

        // Get name of the temporary file.
        std::string path() const override { return _file_path; }

        [[nodiscard]] turbo::Status write(std::string_view buff) override;

        [[nodiscard]] turbo::Status write(const turbo::IOBuf &buff) override;

        [[nodiscard]] turbo::Status flush() override { return _file.flush(); }

        [[nodiscard]] turbo::Status truncate(size_t size) override { return _file.truncate(size); }

        [[nodiscard]] ResultStatus<size_t> size() const override { return _file.size(); }

        void close() override { _file.close(); }


    private:
        // temp_file is associated with file, copying makes no sense.
        TURBO_NON_COPYABLE(TempFile);

        std::string generate_temp_file_name(std::string_view prefix, std::string_view ext, size_t bits);
        std::string _file_path;
        SequentialWriteFile _file;
        bool         _ever_opened{false};
    };

    /// inlined implementations

    [[nodiscard]] inline turbo::Status TempFile::write(std::string_view buff) {
        return write(buff.data(), buff.size());
    }

     [[nodiscard]] inline turbo::Status TempFile::write(const turbo::IOBuf &buff)  {
        return _file.write(buff);
    }

} // namespace turbo

#endif  // TURBO_FILES_TEMP_FILE_H_
