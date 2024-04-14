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

#ifndef TURBO_FILES_IO_ZERO_COPY_STREAM_H_
#define TURBO_FILES_IO_ZERO_COPY_STREAM_H_

#include <string>
#include "turbo/platform/port.h"
#include "turbo/strings/cord.h"

namespace turbo {


    class TURBO_DLL ZeroCopyInputStream {
    public:
        ZeroCopyInputStream() = default;

        virtual ~ZeroCopyInputStream() = default;

        ZeroCopyInputStream(const ZeroCopyInputStream &) = delete;

        ZeroCopyInputStream &operator=(const ZeroCopyInputStream &) = delete;

        ZeroCopyInputStream(ZeroCopyInputStream &&) = delete;

        ZeroCopyInputStream &operator=(ZeroCopyInputStream &&) = delete;

        // Obtains a chunk of data from the stream.
        //
        // Preconditions:
        // * "size" and "data" are not NULL.
        //
        // Postconditions:
        // * If the returned value is false, there is no more data to return or
        //   an error occurred.  All errors are permanent.
        // * Otherwise, "size" points to the actual number of bytes read and "data"
        //   points to a pointer to a buffer containing these bytes.
        // * Ownership of this buffer remains with the stream, and the buffer
        //   remains valid only until some other method of the stream is called
        //   or the stream is destroyed.
        // * It is legal for the returned buffer to have zero size, as long
        //   as repeatedly calling Next() eventually yields a buffer with non-zero
        //   size.
        virtual bool next(const void **data, int *size) = 0;

        // Backs up a number of bytes, so that the next call to Next() returns
        // data again that was already returned by the last call to Next().  This
        // is useful when writing procedures that are only supposed to read up
        // to a certain point in the input, then return.  If Next() returns a
        // buffer that goes beyond what you wanted to read, you can use BackUp()
        // to return to the point where you intended to finish.
        //
        // Preconditions:
        // * The last method called must have been Next().
        // * count must be less than or equal to the size of the last buffer
        //   returned by Next().
        //
        // Postconditions:
        // * The last "count" bytes of the last buffer returned by Next() will be
        //   pushed back into the stream.  Subsequent calls to Next() will return
        //   the same data again before producing new data.
        virtual void back_up(int count) = 0;

        // Skips `count` number of bytes.
        // Returns true on success, or false if some input error occurred, or `count`
        // exceeds the end of the stream. This function may skip up to `count - 1`
        // bytes in case of failure.
        //
        // Preconditions:
        // * `count` is non-negative.
        //
        virtual bool skip(int count) = 0;

        // Returns the total number of bytes read since this object was created.
        virtual size_t byte_count() const = 0;

        // Read the next `count` bytes and append it to the given Cord.
        //
        // In the case of a read error, the method reads as much data as possible into
        // the cord before returning false. The default implementation iterates over
        // the buffers and appends up to `count` bytes of data into `cord` using the
        // `turbo::CordBuffer` API.
        //
        // Some streams may implement this in a way that avoids copying by sharing or
        // reference counting existing data managed by the stream implementation.
        //
        virtual bool read_cord(turbo::Cord *cord, int count);

    };

    // Abstract interface similar to an output stream but designed to minimize
    // copying.
    class TURBO_DLL ZeroCopyOutputStream {
    public:
        ZeroCopyOutputStream() {}

        ZeroCopyOutputStream(const ZeroCopyOutputStream &) = delete;

        ZeroCopyOutputStream &operator=(const ZeroCopyOutputStream &) = delete;

        virtual ~ZeroCopyOutputStream() {}

        // Obtains a buffer into which data can be written.  Any data written
        // into this buffer will eventually (maybe instantly, maybe later on)
        // be written to the output.
        //
        // Preconditions:
        // * "size" and "data" are not NULL.
        //
        // Postconditions:
        // * If the returned value is false, an error occurred.  All errors are
        //   permanent.
        // * Otherwise, "size" points to the actual number of bytes in the buffer
        //   and "data" points to the buffer.
        // * Ownership of this buffer remains with the stream, and the buffer
        //   remains valid only until some other method of the stream is called
        //   or the stream is destroyed.
        // * Any data which the caller stores in this buffer will eventually be
        //   written to the output (unless BackUp() is called).
        // * It is legal for the returned buffer to have zero size, as long
        //   as repeatedly calling Next() eventually yields a buffer with non-zero
        //   size.
        virtual bool next(void **data, int *size) = 0;

        // Backs up a number of bytes, so that the end of the last buffer returned
        // by Next() is not actually written.  This is needed when you finish
        // writing all the data you want to write, but the last buffer was bigger
        // than you needed.  You don't want to write a bunch of garbage after the
        // end of your data, so you use BackUp() to back up.
        //
        // This method can be called with `count = 0` to finalize (flush) any
        // previously returned buffer. For example, a file output stream can
        // flush buffers returned from a previous call to Next() upon such
        // BackUp(0) invocations. ZeroCopyOutputStream callers should always
        // invoke BackUp() after a final Next() call, even if there is no
        // excess buffer data to be backed up to indicate a flush point.
        //
        // Preconditions:
        // * The last method called must have been Next().
        // * count must be less than or equal to the size of the last buffer
        //   returned by Next().
        // * The caller must not have written anything to the last "count" bytes
        //   of that buffer.
        //
        // Postconditions:
        // * The last "count" bytes of the last buffer returned by Next() will be
        //   ignored.
        virtual void back_up(int count) = 0;

        // Returns the total number of bytes written since this object was created.
        virtual size_t byte_count() const = 0;

        // Write a given chunk of data to the output.  Some output streams may
        // implement this in a way that avoids copying. Check AllowsAliasing() before
        // calling WriteAliasedRaw(). It will TURBO_CHECK fail if WriteAliasedRaw() is
        // called on a stream that does not allow aliasing.
        //
        // NOTE: It is caller's responsibility to ensure that the chunk of memory
        // remains live until all of the data has been consumed from the stream.
        virtual bool write_aliased_raw(const void *data, int size);

        virtual bool allows_aliasing() const { return false; }

        // Writes the given Cord to the output.
        //
        // The default implementation iterates over all Cord chunks copying all cord
        // data into the buffer(s) returned by the stream's `Next()` method.
        //
        // Some streams may implement this in a way that avoids copying the cord
        // data by copying and managing a copy of the provided cord instead.
        virtual bool write_cord(const turbo::Cord &cord);

    };

    class ZeroCopyStreamAsStreamBuf : public std::streambuf {
    public:
        ZeroCopyStreamAsStreamBuf(ZeroCopyOutputStream *stream)
                : _zero_copy_stream(stream) {}

        virtual ~ZeroCopyStreamAsStreamBuf();

        // BackUp() unused bytes. Automatically called in destructor.
        void shrink();

    protected:
        int overflow(int ch) override;

        int sync() override;

        std::streampos seekoff(std::streamoff off,
                               std::ios_base::seekdir way,
                               std::ios_base::openmode which) override;

    private:
        ZeroCopyOutputStream *_zero_copy_stream;
    };

}  // namespace turbo

#endif  // TURBO_FILES_IO_ZERO_COPY_STREAM_H_
