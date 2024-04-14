// Formatting library for C++ - std::ostream support
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_OSTREAM_H_
#define FMT_OSTREAM_H_

#include <fstream>  // std::filebuf

#if defined(_WIN32) && defined(__GLIBCXX__)
#  include <ext/stdio_filebuf.h>
#  include <ext/stdio_sync_filebuf.h>
#elif defined(_WIN32) && defined(_LIBCPP_VERSION)
#  include <__std_stream>
#endif

#include "format.h"

namespace turbo {

    namespace fmt_detail {

        // Generate a unique explicit instantion in every translation unit using a tag
        // type in an anonymous namespace.
        namespace {
            struct file_access_tag {
            };
        }  // namespace
        template<typename Tag, typename BufType, FILE *BufType::*FileMemberPtr>
        class file_access {
            friend auto get_file(BufType &obj) -> FILE * { return obj.*FileMemberPtr; }
        };

#if TURBO_MSC_VERSION
        template class file_access<file_access_tag, std::filebuf,
                                   &std::filebuf::_Myfile>;
        auto get_file(std::filebuf&) -> FILE*;
#elif defined(_WIN32) && defined(_LIBCPP_VERSION)
        template class file_access<file_access_tag, std::__stdoutbuf<char>,
                                   &std::__stdoutbuf<char>::__file_>;
        auto get_file(std::__stdoutbuf<char>&) -> FILE*;
#endif

        inline bool write_ostream_unicode(std::ostream &os, std::string_view data) {
#if TURBO_MSC_VERSION
            if (auto* buf = dynamic_cast<std::filebuf*>(os.rdbuf()))
              if (FILE* f = get_file(*buf)) return write_console(f, data);
#elif defined(_WIN32) && defined(__GLIBCXX__)
            auto* rdbuf = os.rdbuf();
            FILE* c_file;
            if (auto* sfbuf = dynamic_cast<__gnu_cxx::stdio_sync_filebuf<char>*>(rdbuf))
              c_file = sfbuf->file();
            else if (auto* fbuf = dynamic_cast<__gnu_cxx::stdio_filebuf<char>*>(rdbuf))
              c_file = fbuf->file();
            else
              return false;
            if (c_file) return write_console(c_file, data);
#elif defined(_WIN32) && defined(_LIBCPP_VERSION)
            if (auto* buf = dynamic_cast<std::__stdoutbuf<char>*>(os.rdbuf()))
              if (FILE* f = get_file(*buf)) return write_console(f, data);
#else
            TURBO_UNUSED(os);
            TURBO_UNUSED(data);
#endif
            return false;
        }

        inline bool write_ostream_unicode(std::wostream &,
                                          std::basic_string_view<wchar_t>) {
            return false;
        }

        // Write the content of buf to os.
        // It is a separate function rather than a part of vprint to simplify testing.
        template<typename Char>
        void write_buffer(std::basic_ostream<Char> &os, buffer<Char> &buf) {
            const Char *buf_data = buf.data();
            using unsigned_streamsize = std::make_unsigned<std::streamsize>::type;
            unsigned_streamsize size = buf.size();
            unsigned_streamsize max_size = turbo::to_unsigned(max_value<std::streamsize>());
            do {
                unsigned_streamsize n = size <= max_size ? size : max_size;
                os.write(buf_data, static_cast<std::streamsize>(n));
                buf_data += n;
                size -= n;
            } while (size != 0);
        }

        template<typename Char, typename T>
        void format_value(buffer<Char> &buf, const T &value,
                          locale_ref loc = locale_ref()) {
            auto &&format_buf = formatbuf<std::basic_streambuf<Char>>(buf);
            auto &&output = std::basic_ostream<Char>(&format_buf);
#if !defined(FMT_STATIC_THOUSANDS_SEPARATOR)
            if (loc) output.imbue(loc.get<std::locale>());
#endif
            output << value;
            output.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        }

        template<typename T>
        struct streamed_view {
            const T &value;
        };

    }  // namespace fmt_detail

    // Formats an object of type T that has an overloaded ostream operator<<.
    template<typename Char>
    struct basic_ostream_formatter : formatter<std::basic_string_view<Char>, Char> {
        void set_debug_format() = delete;

        template<typename T, typename OutputIt>
        auto format(const T &value, basic_format_context<OutputIt, Char> &ctx) const
        -> OutputIt {
            auto buffer = basic_memory_buffer<Char>();
            fmt_detail::format_value(buffer, value, ctx.locale());
            return formatter<std::basic_string_view<Char>, Char>::format(
                    {buffer.data(), buffer.size()}, ctx);
        }
    };

    using ostream_formatter = basic_ostream_formatter<char>;

    template<typename T, typename Char>
    struct formatter<fmt_detail::streamed_view<T>, Char>
            : basic_ostream_formatter<Char> {
        template<typename OutputIt>
        auto format(fmt_detail::streamed_view<T> view,
                    basic_format_context<OutputIt, Char> &ctx) const -> OutputIt {
            return basic_ostream_formatter<Char>::format(view.value, ctx);
        }
    };

    /**
      \rst
      Returns a view that formats `value` via an ostream ``operator<<``.

      **Example**::

        turbo::print("Current thread id: {}\n",
                   turbo::streamed(std::this_thread::get_id()));
      \endrst
     */
    template<typename T>
    auto streamed(const T &value) -> fmt_detail::streamed_view<T> {
        return {value};
    }

    namespace fmt_detail {

        inline void vprint_directly(std::ostream &os, std::string_view format_str,
                                    format_args args) {
            auto buffer = memory_buffer();
            fmt_detail::vformat_to(buffer, format_str, args);
            fmt_detail::write_buffer(os, buffer);
        }

    }  // namespace fmt_detail

    template<typename Char>
    void vprint(std::basic_ostream<Char> &os,
                std::basic_string_view<type_identity_t<Char>> format_str,
                basic_format_args<buffer_context<type_identity_t<Char>>> args) {
        auto buffer = basic_memory_buffer<Char>();
        fmt_detail::vformat_to(buffer, format_str, args);
        if (fmt_detail::write_ostream_unicode(os, {buffer.data(), buffer.size()})) return;
        fmt_detail::write_buffer(os, buffer);
    }

}  // namespace turbo

#endif  // FMT_OSTREAM_H_
