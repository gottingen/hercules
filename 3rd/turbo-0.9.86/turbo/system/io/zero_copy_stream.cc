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

#include "turbo/system/io/zero_copy_stream.h"
#include "turbo/meta/span.h"
#include "turbo/strings/cord.h"
#include "turbo/log/logging.h"

namespace turbo {


    bool ZeroCopyInputStream::read_cord(turbo::Cord* cord, int count) {
        if (count <= 0) return true;

        turbo::CordBuffer cord_buffer = cord->get_append_buffer(count);
        turbo::Span<char> out = cord_buffer.available_up_to(count);

        auto FetchNextChunk = [&]() -> turbo::Span<const char> {
            const void* buffer;
            int size;
            if (!next(&buffer, &size)) return {};

            if (size > count) {
                back_up(size - count);
                size = count;
            }
            return turbo::MakeConstSpan(static_cast<const char*>(buffer), size);
        };

        auto AppendFullBuffer = [&]() -> turbo::Span<char> {
            cord->append(std::move(cord_buffer));
            cord_buffer = turbo::CordBuffer::CreateWithDefaultLimit(count);
            return cord_buffer.available_up_to(count);
        };

        auto CopyBytes = [&](turbo::Span<char>& dst, turbo::Span<const char>& src,
                             size_t bytes) {
            memcpy(dst.data(), src.data(), bytes);
            dst.remove_prefix(bytes);
            src.remove_prefix(bytes);
            count -= bytes;
            cord_buffer.IncreaseLengthBy(bytes);
        };

        do {
            turbo::Span<const char> in = FetchNextChunk();
            if (in.empty()) {
                // Append whatever we have pending so far.
                cord->append(std::move(cord_buffer));
                return false;
            }

            if (out.empty()) out = AppendFullBuffer();

            while (in.size() > out.size()) {
                CopyBytes(out, in, out.size());
                out = AppendFullBuffer();
            }

            CopyBytes(out, in, in.size());
        } while (count > 0);

        cord->append(std::move(cord_buffer));
        return true;
    }

    bool ZeroCopyOutputStream::write_cord(const turbo::Cord& cord) {
        if (cord.empty()) return true;

        void* buffer;
        int buffer_size = 0;
        if (!next(&buffer, &buffer_size)) return false;

        for (turbo::string_view fragment : cord.chunks()) {
            while (fragment.size() > static_cast<size_t>(buffer_size)) {
                std::memcpy(buffer, fragment.data(), buffer_size);

                fragment.remove_prefix(buffer_size);

                if (!next(&buffer, &buffer_size)) return false;
            }
            std::memcpy(buffer, fragment.data(), fragment.size());

            // Advance the buffer.
            buffer = static_cast<char*>(buffer) + fragment.size();
            buffer_size -= static_cast<int>(fragment.size());
        }
        back_up(buffer_size);
        return true;
    }


    bool ZeroCopyOutputStream::write_aliased_raw(const void* /* data */,
                                               int /* size */) {
        TLOG_CHECK(false, "WriteAliasedRaw() not implemented for this stream type");
        return false;
    }


    static_assert(sizeof(std::streambuf::char_type) == sizeof(char),"only support char");

    int ZeroCopyStreamAsStreamBuf::overflow(int ch) {
        if (ch == std::streambuf::traits_type::eof()) {
            return ch;
        }
        void* block = nullptr;
        int size = 0;
        if (_zero_copy_stream->next(&block, &size)) {
            setp((char*)block, (char*)block + size);
            // if size == 0, this function will call overflow again.
            return sputc(ch);
        } else {
            setp(nullptr, nullptr);
            return std::streambuf::traits_type::eof();
        }
    }

    int ZeroCopyStreamAsStreamBuf::sync() {
        // data are already in IOBuf.
        return 0;
    }

    ZeroCopyStreamAsStreamBuf::~ZeroCopyStreamAsStreamBuf() {
        shrink();
    }

    void ZeroCopyStreamAsStreamBuf::shrink() {
        if (pbase() != nullptr) {
            _zero_copy_stream->back_up(epptr() - pptr());
            setp(nullptr, nullptr);
        }
    }

    std::streampos ZeroCopyStreamAsStreamBuf::seekoff(
            std::streamoff off,
            std::ios_base::seekdir way,
            std::ios_base::openmode which) {
        if (off == 0 && way == std::ios_base::cur) {
            return _zero_copy_stream->byte_count() - (epptr() - pptr());
        }
        return (std::streampos)(std::streamoff)-1;
    }

}  // namespace turbo
