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

#ifndef TURBO_SYSTEM_IO_IOBUF_PRINTER_H_
#define TURBO_SYSTEM_IO_IOBUF_PRINTER_H_

#include "turbo/format/format.h"
#include "turbo/system/io/iobuf.h"
#include <string_view>


namespace turbo {

    using files_internal::IOBuf;

    class ToPrintable {
    public:
        static const size_t DEFAULT_MAX_LENGTH = 64;

        ToPrintable(const IOBuf& b, size_t max_length = DEFAULT_MAX_LENGTH)
                : _iobuf(&b), _max_length(max_length) {}

        ToPrintable(const std::string_view str, size_t max_length = DEFAULT_MAX_LENGTH)
                : _iobuf(nullptr), _str(str), _max_length(max_length) {}

        ToPrintable(const void* data, size_t n, size_t max_length = DEFAULT_MAX_LENGTH)
                : _iobuf(nullptr), _str((const char*)data, n), _max_length(max_length) {}

        void print(std::ostream& os) const;

        std::string to_string() const;

    private:
        const IOBuf* _iobuf;
        std::string_view _str;
        size_t _max_length;
    };

    inline std::ostream& operator<<(std::ostream& os, const ToPrintable& p) {
        p.print(os);
        return os;
    }

    // Convert binary data to a printable string.
    std::string to_printable_string(const IOBuf& data,
                                  size_t max_length = ToPrintable::DEFAULT_MAX_LENGTH);
    std::string to_printable_string(const std::string_view& data,
                                  size_t max_length = ToPrintable::DEFAULT_MAX_LENGTH);
    std::string to_printable_string(const void* data, size_t n,
                                  size_t max_length = ToPrintable::DEFAULT_MAX_LENGTH);

    template<>
    struct formatter<ToPrintable> : public formatter<std::string_view> {
        template<typename FormatContext>
        auto format(const ToPrintable& p, FormatContext& ctx) {
            auto str = p.to_string();
            return formatter<std::string_view>::format(str, ctx);
        }
    };

    template<>
    struct formatter<IOBuf> : public formatter<std::string_view> {
        template<typename FormatContext>
        auto format(const IOBuf& buf, FormatContext& ctx) {
            auto str = buf.to_string();
            return formatter<std::string_view>::format(str, ctx);
        }
    };

}
#endif  // TURBO_SYSTEM_IO_IOBUF_PRINTER_H_
