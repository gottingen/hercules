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
// Created by jeff on 24-1-8.
//

#ifndef TURBO_FILES_FILESYSTEM_H_
#define TURBO_FILES_FILESYSTEM_H_

#include "turbo/files/internal/filesystem.h"
#include "turbo/files/fwd.h"
#include "turbo/files/sys/sys_adapter.h"

namespace turbo {

    //
    // return file path and its extension:
    //
    // "mylog.txt" => ("mylog", ".txt")
    // "mylog" => ("mylog", "")
    // "mylog." => ("mylog.", "")
    // "/dir1/dir2/mylog.txt" => ("/dir1/dir2/mylog", ".txt")
    //
    // the starting dot in filenames is ignored (hidden files):
    //
    // ".mylog" => (".mylog". "")
    // "my_folder/.mylog" => ("my_folder/.mylog", "")
    // "my_folder/.mylog.txt" => ("my_folder/.mylog", ".txt")

    std::tuple<std::string, std::string> split_by_extension(const std::string &fname);

    /**
         * @ingroup turbo_files_utility
         * @brief get file md5 sum.
         * @param path [input] file path
         * @param size [output] file size that is calculated.
         * @return md5 sum of the file and the status of the operation.
         * @note use the result of this function should check the status of the operation first.
         *      if the status is not ok, the result is invalid.
         * @code
         *     auto rs = FileUtility::md5_sum_file("test.txt");
         *     if (rs.ok()) {
         *          std::cout << rs.value() << std::endl;
         *          // do something else.
         *      } else {
         *           std::cout << rs.status().message() << std::endl;
         *           // do something else.
         *      }
         * @endcode
         */
    turbo::ResultStatus<std::string> md5_sum_file(const std::string_view &path, int64_t *size = nullptr);

    /**
     * @ingroup turbo_files_utility
     * @brief get file sha1 sum.
     * @param path [input] file path
     * @param size [output] file size that is calculated.
     * @return sha1 sum of the file and the status of the operation.
     * @note use the result of this function should check the status of the operation first.
     *      if the status is not ok, the result is invalid.
     * @code
     *     auto rs = FileUtility::sha1_sum_file("test.txt");
     *     if (rs.ok()) {
     *          std::cout << rs.value() << std::endl;
     *          // do something else.
     *      } else {
     *           std::cout << rs.status().message() << std::endl;
     *           // do something else.
     *      }
     * @endcode
     */
    turbo::ResultStatus<std::string> sha1_sum_file(const std::string_view &path, int64_t *size = nullptr);

    template<typename Tag = sys_adapter>
    struct Filesystem {

        static SequentialFileReader* create_sequential_file_reader();

        static RandomAccessFileReader* create_random_file_reader();

        static SequentialFileWriter* create_sequential_file_writer();

        static RandomFileWriter* create_random_file_writer();

        static TempFileWriter* create_temp_file_writer();

        static turbo::Status read_file(const std::string_view &file_path, std::string &result, bool append = false) noexcept;

        static turbo::Status write_file(const std::string_view &file_path, const std::string_view &content, bool truncate = true) noexcept;

        static turbo::Status list_files(const std::string_view &root_path,std::vector<std::string> &result, bool full_path = true) noexcept;

        static turbo::Status list_directories(const std::string_view &root_path,std::vector<std::string> &result, bool full_path = true) noexcept;

        static turbo::ResultStatus<size_t> file_size(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::ResultStatus<bool> exists(const std::string_view &dir_path) noexcept;

        static turbo::Status remove(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status remove_all(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status rename(const turbo::filesystem::path &old_path, const turbo::filesystem::path &new_path) noexcept;

        static turbo::Status create_directory(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status create_directories(const turbo::filesystem::path &dir_path) noexcept;

        static turbo::Status resize_file(const turbo::filesystem::path &dir_path, size_t size) noexcept;
    };

    template<typename Tag>
    inline SequentialFileReader* Filesystem<Tag>::create_sequential_file_reader() {
        return FileAdapter<Tag>::create_sequential_file_reader();
    }

    template<typename Tag>
    inline RandomAccessFileReader* Filesystem<Tag>::create_random_file_reader() {
        return FileAdapter<Tag>::create_random_file_reader();
    }

    template<typename Tag>
    inline SequentialFileWriter* Filesystem<Tag>::create_sequential_file_writer() {
        return FileAdapter<Tag>::create_sequential_file_writer();
    }

    template<typename Tag>
    inline RandomFileWriter* Filesystem<Tag>::create_random_file_writer() {
        return FileAdapter<Tag>::create_random_file_writer();
    }

    template<typename Tag>
    inline TempFileWriter* Filesystem<Tag>::create_temp_file_writer() {
        return FileAdapter<Tag>::create_temp_file_writer();
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::read_file(const std::string_view &file_path, std::string &result, bool append) noexcept {
        return FileAdapter<Tag>::read_file(file_path, result, append);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::write_file(const std::string_view &file_path, const std::string_view &content, bool truncate) noexcept {
        return FileAdapter<Tag>::write_file(file_path, content, truncate);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::list_files(const std::string_view &root_path,std::vector<std::string> &result, bool full_path) noexcept {
        return FileAdapter<Tag>::list_files(root_path, result, full_path);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::list_directories(const std::string_view &root_path,std::vector<std::string> &result, bool full_path) noexcept {
        return FileAdapter<Tag>::list_directories(root_path, result, full_path);
    }

    template<typename Tag>
    inline turbo::ResultStatus<size_t> Filesystem<Tag>::file_size(const turbo::filesystem::path &dir_path) noexcept {
        return FileAdapter<Tag>::file_size(dir_path);
    }

    template<typename Tag>
    inline turbo::ResultStatus<bool> Filesystem<Tag>::exists(const std::string_view &dir_path) noexcept {
        return FileAdapter<Tag>::exists(dir_path);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::remove(const turbo::filesystem::path &dir_path) noexcept {
        return FileAdapter<Tag>::remove(dir_path);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::remove_all(const turbo::filesystem::path &dir_path) noexcept {
        return FileAdapter<Tag>::remove_all(dir_path);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::rename(const turbo::filesystem::path &old_path, const turbo::filesystem::path &new_path) noexcept {
        return FileAdapter<Tag>::rename(old_path, new_path);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::create_directory(const turbo::filesystem::path &dir_path) noexcept {
        return FileAdapter<Tag>::create_directory(dir_path);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::create_directories(const turbo::filesystem::path &dir_path) noexcept {
        return FileAdapter<Tag>::create_directories(dir_path);
    }

    template<typename Tag>
    inline turbo::Status Filesystem<Tag>::resize_file(const turbo::filesystem::path &dir_path, size_t size) noexcept {
        return FileAdapter<Tag>::resize_file(dir_path, size);
    }

}  // namespace turbo


namespace turbo {

    namespace fmt_detail {

        template<typename Char>
        void write_escaped_path(basic_memory_buffer<Char> &quoted,
                                const turbo::filesystem::path &p) {
            write_escaped_string<Char>(std::back_inserter(quoted), p.string<Char>());
        }

#  ifdef _WIN32
        template <>
        inline void write_escaped_path<char>(memory_buffer& quoted,
                                             const std::filesystem::path& p) {
          auto buf = basic_memory_buffer<wchar_t>();
          write_escaped_string<wchar_t>(std::back_inserter(buf), p.native());
          // Convert UTF-16 to UTF-8.
          if (!unicode_to_utf8<wchar_t>::convert(quoted, {buf.data(), buf.size()}))
            FMT_THROW(std::runtime_error("invalid utf16"));
        }
#  endif

        template<>
        inline void write_escaped_path<turbo::filesystem::path::value_type>(
                basic_memory_buffer<turbo::filesystem::path::value_type> &quoted,
                const turbo::filesystem::path &p) {
            write_escaped_string<turbo::filesystem::path::value_type>(
                    std::back_inserter(quoted), p.native());
        }

    }  // namespace fmt_detail

    template<typename Char>
    struct formatter<turbo::filesystem::path, Char>
            : formatter<std::basic_string_view<Char>> {
        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) {
            auto out = formatter<std::basic_string_view<Char>>::parse(ctx);
            this->set_debug_format(false);
            return out;
        }

        template<typename FormatContext>
        auto format(const turbo::filesystem::path &p, FormatContext &ctx) const ->
        typename FormatContext::iterator {
            auto quoted = basic_memory_buffer<Char>();
            fmt_detail::write_escaped_path(quoted, p);
            return formatter<std::basic_string_view<Char>>::format(
                    std::basic_string_view<Char>(quoted.data(), quoted.size()), ctx);
        }
    };

}  // namespace turbo
#endif  // TURBO_FILES_FILESYSTEM_H_
