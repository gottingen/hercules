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

#ifndef TURBO_SYSTEM_IO_IOBUF_H_
#define TURBO_SYSTEM_IO_IOBUF_H_

#include <sys/uio.h>                             // iovec
#include <stdint.h>                              // uint32_t
#include <string>                                // std::string
#include <ostream>                               // std::ostream
#include <string_view>
#include "turbo/system/io/writer.h"
#include "turbo/system/io/reader.h"
#include "turbo/system/io/zero_copy_stream.h"
#include "turbo/status/result_status.h"

extern "C" {
struct const_iovec {
    const void *iov_base;
    size_t iov_len;
};
}

namespace turbo::files_internal {

    // IOBuf is a non-continuous buffer that can be cut and combined w/o copying
    // payload. It can be read from or flushed into file descriptors as well.
    // IOBuf is [thread-compatible]. Namely using different IOBuf in different
    // threads simultaneously is safe, and reading a static IOBuf from different
    // threads is safe as well.
    // IOBuf is [NOT thread-safe]. Modifying a same IOBuf from different threads
    // simultaneously is unsafe and likely to crash.
    class IOBuf {
        friend class IOBufAsZeroCopyInputStream;

        friend class IOBufAsZeroCopyOutputStream;

        friend class IOBufBytesIterator;

        friend class IOBufCutter;

    public:
        static const size_t DEFAULT_BLOCK_SIZE = 8192;
        static const size_t INITIAL_CAP = 32; // must be power of 2

        struct Block;

        // can't directly use `struct iovec' here because we also need to access the
        // reference counter(nshared) in Block*
        struct BlockRef {
            // NOTICE: first bit of `offset' is shared with BigView::start
            uint32_t offset;
            uint32_t length;
            Block *block;
        };

        // IOBuf is essentially a tiny queue of BlockRefs.
        struct SmallView {
            BlockRef refs[2];
        };

        struct BigView {
            int32_t magic;
            uint32_t start;
            BlockRef *refs;
            uint32_t nref;
            uint32_t cap_mask;
            size_t nbytes;

            const BlockRef &ref_at(uint32_t i) const { return refs[(start + i) & cap_mask]; }

            BlockRef &ref_at(uint32_t i) { return refs[(start + i) & cap_mask]; }

            uint32_t capacity() const { return cap_mask + 1; }
        };

        struct Movable {
            explicit Movable(IOBuf &v) : _v(&v) {}

            IOBuf &value() const { return *_v; }

        private:
            IOBuf *_v;
        };

        typedef uint64_t Area;
        static const Area INVALID_AREA = 0;

        IOBuf();

        IOBuf(const IOBuf &);

        IOBuf(IOBuf &&);

        IOBuf(const Movable &);

        ~IOBuf() { clear(); }

        void operator=(const IOBuf &);

        void operator=(IOBuf &&);
        void operator=(const Movable &);

        void operator=(const char *);

        void operator=(const std::string &);

        // Exchange internal fields with another IOBuf.
        void swap(IOBuf &);

        // Pop n bytes from front side
        // If n == 0, nothing popped; if n >= length(), all bytes are popped
        // Returns bytes popped.
        size_t pop_front(size_t n);

        // Pop n bytes from back side
        // If n == 0, nothing popped; if n >= length(), all bytes are popped
        // Returns bytes popped.
        size_t pop_back(size_t n);

        // Cut off n bytes from front side and APPEND to `out'
        // If n == 0, nothing cut; if n >= length(), all bytes are cut
        // Returns bytes cut.
        size_t cutn(IOBuf *out, size_t n);

        size_t cutn(void *out, size_t n);

        size_t cutn(std::string *out, size_t n);

        // Cut off 1 byte from the front side and set to *c
        // Return true on cut, false otherwise.
        bool cut1(void *c);

        // Cut from front side until the characters matches `delim', append
        // data before the matched characters to `out'.
        // Returns 0 on success, -1 when there's no match (including empty `delim')
        // or other errors.
        int cut_until(IOBuf *out, char const *delim);

        // std::string version, `delim' could be binary
        int cut_until(IOBuf *out, const std::string &delim);

        // Cut at most `size_hint' bytes(approximately) into the writer
        // Returns bytes cut on success, -1 otherwise and errno is set.
        ssize_t cut_into_writer(IWriter *writer, size_t size_hint = 1024 * 1024);

        // Cut at most `size_hint' bytes(approximately) into the file descriptor
        // Returns bytes cut on success, -1 otherwise and errno is set.
        turbo::ResultStatus<ssize_t> cut_into_file_descriptor(int fd, size_t size_hint = 1024 * 1024);

        // Cut at most `size_hint' bytes(approximately) into the file descriptor at
        // a given offset(from the start of the file). The file offset is not changed.
        // If `offset' is negative, does exactly what cut_into_file_descriptor does.
        // Returns bytes cut on success, -1 otherwise and errno is set.
        //
        // NOTE: POSIX requires that a file open with the O_APPEND flag should
        // not affect pwrite(). However, on Linux, if |fd| is open with O_APPEND,
        // pwrite() appends data to the end of the file, regardless of the value
        // of |offset|.
        turbo::ResultStatus<ssize_t> pcut_into_file_descriptor(int fd, off_t offset /*NOTE*/,
                                          size_t size_hint = 1024 * 1024);

        // Cut `count' number of `pieces' into the writer.
        // Returns bytes cut on success, -1 otherwise and errno is set.
        static ResultStatus<ssize_t> cut_multiple_into_writer(
                IWriter *writer, IOBuf *const *pieces, size_t count);

        // Cut `count' number of `pieces' into the file descriptor.
        // Returns bytes cut on success, -1 otherwise and errno is set.
        static turbo::ResultStatus<ssize_t> cut_multiple_into_file_descriptor(
                int fd, IOBuf *const *pieces, size_t count);

        // Cut `count' number of `pieces' into file descriptor `fd' at a given
        // offset. The file offset is not changed.
        // If `offset' is negative, does exactly what cut_multiple_into_file_descriptor
        // does.
        // Read NOTE of pcut_into_file_descriptor.
        // Returns bytes cut on success, -1 otherwise and errno is set.
        static turbo::ResultStatus<ssize_t> pcut_multiple_into_file_descriptor(
                int fd, off_t offset, IOBuf *const *pieces, size_t count);

        // Append another IOBuf to back side, payload of the IOBuf is shared
        // rather than copied.
        void append(const IOBuf &other);

        void append(IOBuf &&other);

        // Append content of `other' to self and clear `other'.
        void append(const Movable &other);

        // ===================================================================
        // Following push_back()/append() are just implemented for convenience
        // and occasional usages, they're relatively slow because of the overhead
        // of frequent BlockRef-management and reference-countings. If you get
        // a lot of push_back/append to do, you should use IOBufAppender or
        // IOBufBuilder instead, which reduce overhead by owning IOBuf::Block.
        // ===================================================================

        // Append a character to back side. (with copying)
        // Returns 0 on success, -1 otherwise.
        int push_back(char c);

        // Append `data' with `count' bytes to back side. (with copying)
        // Returns 0 on success(include count == 0), -1 otherwise.
        int append(void const *data, size_t count);

        // Append multiple data to back side in one call, faster than appending
        // one by one separately.
        // Returns 0 on success, -1 otherwise.
        // Example:
        //   const_iovec vec[] = { { data1, len1 },
        //                         { data2, len2 },
        //                         { data3, len3 } };
        //   foo.appendv(vec, arraysize(vec));
        int appendv(const const_iovec vec[], size_t n);

        int appendv(const iovec *vec, size_t n) { return appendv((const const_iovec *) vec, n); }

        // Append a c-style string to back side. (with copying)
        // Returns 0 on success, -1 otherwise.
        // NOTE: Returns 0 when `s' is empty.
        int append(char const *s);

        // Append a std::string to back side. (with copying)
        // Returns 0 on success, -1 otherwise.
        // NOTE: Returns 0 when `s' is empty.
        int append(const std::string &s);

        // Append the user-data to back side WITHOUT copying.
        // The user-data can be split and shared by smaller IOBufs and will be
        // deleted using the deleter func when no IOBuf references it anymore.
        int append_user_data(void *data, size_t size, void (*deleter)(void *));

        // Append the user-data to back side WITHOUT copying.
        // The meta is associated with this piece of user-data.
        int append_user_data_with_meta(void *data, size_t size, void (*deleter)(void *), uint64_t meta);

        // Get the data meta of the first byte in this IOBuf.
        // The meta is specified with append_user_data_with_meta before.
        // 0 means the meta is invalid.
        uint64_t get_first_data_meta();

        // Resizes the buf to a length of n characters.
        // If n is smaller than the current length, all bytes after n will be
        // truncated.
        // If n is greater than the current length, the buffer would be append with
        // as many |c| as needed to reach a size of n. If c is not specified,
        // null-character would be appended.
        // Returns 0 on success, -1 otherwise.
        int resize(size_t n) { return resize(n, '\0'); }

        int resize(size_t n, char c);

        // Reserve `n' uninitialized bytes at back-side.
        // Returns an object representing the reserved area, INVALID_AREA on failure.
        // NOTE: reserve(0) returns INVALID_AREA.
        Area reserve(size_t n);

        // [EXTREMELY UNSAFE]
        // Copy `data' to the reserved `area'. `data' must be as long as the
        // reserved size.
        // Returns 0 on success, -1 otherwise.
        // [Rules]
        // 1. Make sure the IOBuf to be assigned was NOT cut/pop from front side
        //    after reserving, otherwise behavior of this function is undefined,
        //    even if it returns 0.
        // 2. Make sure the IOBuf to be assigned was NOT copied to/from another
        //    IOBuf after reserving to prevent underlying blocks from being shared,
        //    otherwise the assignment affects all IOBuf sharing the blocks, which
        //    is probably not what we want.
        int unsafe_assign(Area area, const void *data);

        // Append min(n, length()) bytes starting from `pos' at front side to `buf'.
        // The real payload is shared rather than copied.
        // Returns bytes copied.
        size_t append_to(IOBuf *buf, size_t n = (size_t) -1L, size_t pos = 0) const;

        // Copy min(n, length()) bytes starting from `pos' at front side into `buf'.
        // Returns bytes copied.
        size_t copy_to(void *buf, size_t n = (size_t) -1L, size_t pos = 0) const;

        // NOTE: first parameter is not std::string& because user may pass in
        // a pointer of std::string by mistake, in which case, the void* overload
        // would be wrongly called.
        size_t copy_to(std::string *s, size_t n = (size_t) -1L, size_t pos = 0) const;

        size_t append_to(std::string *s, size_t n = (size_t) -1L, size_t pos = 0) const;

        // Copy min(n, length()) bytes staring from `pos' at front side into
        // `cstr' and end it with '\0'.
        // `cstr' must be as long as min(n, length())+1.
        // Returns bytes copied (not including ending '\0')
        size_t copy_to_cstr(char *cstr, size_t n = (size_t) -1L, size_t pos = 0) const;

        // Convert all data in this buffer to a std::string.
        std::string to_string() const;

        // Get `n' front-side bytes with minimum copying. Length of `aux_buffer'
        // must not be less than `n'.
        // Returns:
        //   nullptr            -  n is greater than length()
        //   aux_buffer      -  n bytes are copied into aux_buffer
        //   internal buffer -  the bytes are stored continuously in the internal
        //                      buffer, no copying is needed. This function does not
        //                      add additional reference to the underlying block,
        //                      so user should not change this IOBuf during using
        //                      the internal buffer.
        // If n == 0 and buffer is empty, return value is undefined.
        const void *fetch(void *aux_buffer, size_t n) const;

        // Fetch one character from front side.
        // Returns pointer to the character, nullptr on empty.
        const void *fetch1() const;

        // Remove all data
        void clear();

        // True iff there's no data
        bool empty() const;

        // Number of bytes
        size_t length() const;

        size_t size() const { return length(); }

        // Get number of Blocks in use. block_memory = block_count * BLOCK_SIZE
        static size_t block_count();

        static size_t block_memory();

        static size_t new_bigview_count();

        static size_t block_count_hit_tls_threshold();

        // Equal with a string/IOBuf or not.
        bool equals(const std::string_view &) const;

        bool equals(const IOBuf &other) const;

        // Get the number of backing blocks
        size_t backing_block_num() const { return _ref_num(); }

        // Get #i backing_block, an empty StringPiece is returned if no such block
        std::string_view backing_block(size_t i) const;

        // Make a movable version of self
        Movable movable() { return Movable(*this); }

    protected:
        int _cut_by_char(IOBuf *out, char);

        int _cut_by_delim(IOBuf *out, char const *dbegin, size_t ndelim);

        // Returns: true iff this should be viewed as SmallView
        bool _small() const;

        template<bool MOVE>
        void _push_or_move_back_ref_to_smallview(const BlockRef &);

        template<bool MOVE>
        void _push_or_move_back_ref_to_bigview(const BlockRef &);

        // Push a BlockRef to back side
        // NOTICE: All fields of the ref must be initialized or assigned
        //         properly, or it will ruin this queue
        void _push_back_ref(const BlockRef &);

        // Move a BlockRef to back side. After calling this function, content of
        // the BlockRef will be invalid and should never be used again.
        void _move_back_ref(const BlockRef &);

        // Pop a BlockRef from front side.
        // Returns: 0 on success and -1 on empty.
        int _pop_front_ref() { return _pop_or_moveout_front_ref<false>(); }

        // Move a BlockRef out from front side.
        // Returns: 0 on success and -1 on empty.
        int _moveout_front_ref() { return _pop_or_moveout_front_ref<true>(); }

        template<bool MOVEOUT>
        int _pop_or_moveout_front_ref();

        // Pop a BlockRef from back side.
        // Returns: 0 on success and -1 on empty.
        int _pop_back_ref();

        // Number of refs in the queue
        size_t _ref_num() const;

        // Get reference to front/back BlockRef in the queue
        // should not be called if queue is empty or the behavior is undefined
        BlockRef &_front_ref();

        const BlockRef &_front_ref() const;

        BlockRef &_back_ref();

        const BlockRef &_back_ref() const;

        // Get reference to n-th BlockRef(counting from front) in the queue
        // NOTICE: should not be called if queue is empty and the `n' must
        //         be inside [0, _ref_num()-1] or behavior is undefined
        BlockRef &_ref_at(size_t i);

        const BlockRef &_ref_at(size_t i) const;

        // Get pointer to n-th BlockRef(counting from front)
        // If i is out-of-range, nullptr is returned.
        const BlockRef *_pref_at(size_t i) const;

    private:
        union {
            BigView _bv;
            SmallView _sv;
        };
    };

    typedef void (*UserDataDeleter)(void *);

    struct UserDataExtension {
        UserDataDeleter deleter;
    };

    struct IOBuf::Block {
        std::atomic<int> nshared;
        uint16_t flags;
        uint16_t abi_check;  // original cap, never be zero.
        uint32_t size;
        uint32_t cap;
        // When flag is 0, portal_next is valid.
        // When flag & IOBUF_BLOCK_FLAGS_USER_DATA is non-0, data_meta is valid.
        union {
            Block *portal_next;
            uint64_t data_meta;
        } u;
        // When flag is 0, data points to `size` bytes starting at `(char*)this+sizeof(Block)'
        // When flag & IOBUF_BLOCK_FLAGS_USER_DATA is non-0, data points to the user data and
        // the deleter is put in UserDataExtension at `(char*)this+sizeof(Block)'
        char *data;

        Block(char *data_in, uint32_t data_size);

        Block(char *data_in, uint32_t data_size, UserDataDeleter deleter);

        // Undefined behavior when (flags & IOBUF_BLOCK_FLAGS_USER_DATA) is 0.
        UserDataExtension *get_user_data_extension() {
            char *p = (char *) this;
            return (UserDataExtension *) (p + sizeof(Block));
        }

        inline void check_abi() {
#ifndef NDEBUG
            if (abi_check != 0) {
                TURBO_ASSERT(false&&"Your program seems to wrongly contain two "
                              "ABI-incompatible implementations of IOBuf");
            }
#endif
        }

        void inc_ref() {
            check_abi();
            nshared.fetch_add(1, std::memory_order_relaxed);
        }

        void dec_ref();

        int ref_count() const {
            return nshared.load(std::memory_order_relaxed);
        }

        bool full() const { return size >= cap; }

        size_t left_space() const { return cap - size; }
    };

    std::ostream &operator<<(std::ostream &, const IOBuf &buf);

    inline bool operator==(const turbo::files_internal::IOBuf &b, const std::string_view &s) { return b.equals(s); }

    inline bool operator==(const std::string_view &s, const turbo::files_internal::IOBuf &b) { return b.equals(s); }

    inline bool operator!=(const turbo::files_internal::IOBuf &b, const std::string_view &s) { return !b.equals(s); }

    inline bool operator!=(const std::string_view &s, const turbo::files_internal::IOBuf &b) { return !b.equals(s); }

    inline bool operator==(const turbo::files_internal::IOBuf &b1, const turbo::files_internal::IOBuf &b2) { return b1.equals(b2); }

    inline bool operator!=(const turbo::files_internal::IOBuf &b1, const turbo::files_internal::IOBuf &b2) { return !b1.equals(b2); }

    // IOPortal is a subclass of IOBuf that can read from file descriptors.
    // Typically used as the buffer to store bytes from sockets.
    class IOPortal : public IOBuf {
    public:
        IOPortal() : _block(nullptr) {}

        IOPortal(const IOPortal &rhs) : IOBuf(rhs), _block(nullptr) {}

        ~IOPortal();

        IOPortal &operator=(const IOPortal &rhs);

        // Read at most `max_count' bytes from the reader and append to self.
        turbo::ResultStatus<ssize_t> append_from_reader(IReader *reader, size_t max_count);

        // Read at most `max_count' bytes from file descriptor `fd' and
        // append to self.
        turbo::ResultStatus<ssize_t> append_from_file_descriptor(int fd, size_t max_count);

        // Read at most `max_count' bytes from file descriptor `fd' at a given
        // offset and append to self. The file offset is not changed.
        // If `offset' is negative, does exactly what append_from_file_descriptor does.
        turbo::ResultStatus<ssize_t> pappend_from_file_descriptor(int fd, off_t offset, size_t max_count);

        // Remove all data inside and return cached blocks.
        void clear();

        // Return cached blocks to TLS. This function should be called by users
        // when this IOPortal are cut into intact messages and becomes empty, to
        // let continuing code on IOBuf to reuse the blocks. Calling this function
        // after each call to append_xxx does not make sense and may hurt
        // performance. Read comments on field `_block' below.
        void return_cached_blocks();

    protected:
        static void return_cached_blocks_impl(Block *);

        // Cached blocks for appending. Notice that the blocks are released
        // until return_cached_blocks()/clear()/dtor() are called, rather than
        // released after each append_xxx(), which makes messages read from one
        // file descriptor more likely to share blocks and have less BlockRefs.
        Block *_block;
    };

    // Specialized utility to cut from IOBuf faster than using corresponding
    // methods in IOBuf.
    // Designed for efficiently parsing data from IOBuf.
    // The cut IOBuf can be appended during cutting.
    class IOBufCutter {
    public:
        explicit IOBufCutter(IOBuf *buf);

        ~IOBufCutter();

        // Cut off n bytes and APPEND to `out'
        // Returns bytes cut.
        size_t cutn(IOBuf *out, size_t n);

        size_t cutn(std::string *out, size_t n);

        size_t cutn(void *out, size_t n);

        // Cut off 1 byte from the front side and set to *c
        // Return true on cut, false otherwise.
        bool cut1(void *data);

        // Copy n bytes into `data'
        // Returns bytes copied.
        size_t copy_to(void *data, size_t n);

        // Fetch one character.
        // Returns pointer to the character, nullptr on empty
        const void *fetch1();

        // Pop n bytes from front side
        // Returns bytes popped.
        size_t pop_front(size_t n);

        // Uncut bytes
        size_t remaining_bytes() const;

    private:
        size_t slower_copy_to(void *data, size_t n);

        bool load_next_ref();

    private:
        void *_data;
        void *_data_end;
        IOBuf::Block *_block;
        IOBuf *_buf;
    };

    // Parse protobuf message from IOBuf. Notice that this wrapper does not change
    // source IOBuf, which also should not change during lifetime of the wrapper.
    // Even if a IOBufAsZeroCopyInputStream is created but parsed, the source
    // IOBuf should not be changed as well becuase constructor of the stream
    // saves internal information of the source IOBuf which is assumed to be
    // unchanged.
    // Example:
    //     IOBufAsZeroCopyInputStream wrapper(the_iobuf_with_protobuf_format_data);
    //     some_pb_message.ParseFromZeroCopyStream(&wrapper);
    class IOBufAsZeroCopyInputStream : public ZeroCopyInputStream {
    public:
        explicit IOBufAsZeroCopyInputStream(const IOBuf &);

        bool next(const void **data, int *size) override;

        void back_up(int count) override;

        bool skip(int count) override;

        size_t byte_count() const override;

    private:
        int _ref_index;
        int _add_offset;
        size_t _byte_count;
        const IOBuf *_buf;
    };

    // Serialize protobuf message into IOBuf. This wrapper does not clear source
    // IOBuf before appending. You can change the source IOBuf when stream is
    // not used(append sth. to the IOBuf, serialize a protobuf message, append
    // sth. again, serialize messages again...). This is different from
    // IOBufAsZeroCopyInputStream which needs the source IOBuf to be unchanged.
    // Example:
    //     IOBufAsZeroCopyOutputStream wrapper(&the_iobuf_to_put_data_in);
    //     some_pb_message.SerializeToZeroCopyStream(&wrapper);
    //
    // NOTE: Blocks are by default shared among all the ZeroCopyOutputStream in one
    // thread. If there are many manipulated streams at one time, there may be many
    // fragments. You can create a ZeroCopyOutputStream which has its own block by
    // passing a positive `block_size' argument to avoid this problem.
    class IOBufAsZeroCopyOutputStream : public  ZeroCopyOutputStream {
    public:
        explicit IOBufAsZeroCopyOutputStream(IOBuf *);

        IOBufAsZeroCopyOutputStream(IOBuf *, uint32_t block_size);

        ~IOBufAsZeroCopyOutputStream() override;

        bool next(void **data, int *size) override;

        void back_up(int count) override;
        size_t byte_count() const override;

    private:
        void _release_block();

        IOBuf *_buf;
        uint32_t _block_size;
        IOBuf::Block *_cur_block;
        int64_t _byte_count;
    };


    // A std::ostream to build IOBuf.
    // Example:
    //   IOBufBuilder builder;
    //   builder << "Anything that can be sent to std::ostream";
    //   // You have several methods to fetch the IOBuf.
    //   target_iobuf.append(builder.buf()); // builder.buf() was not changed
    //   OR
    //   builder.move_to(target_iobuf);      // builder.buf() was clear()-ed.
    class IOBufBuilder :
            // Have to use private inheritance to arrange initialization order.
            virtual private IOBuf,
            virtual private IOBufAsZeroCopyOutputStream,
            virtual private ZeroCopyStreamAsStreamBuf,
            public std::ostream {
    public:
        explicit IOBufBuilder()
                : IOBufAsZeroCopyOutputStream(this), ZeroCopyStreamAsStreamBuf(this), std::ostream(this) {}

        IOBuf &buf() {
            this->shrink();
            return *this;
        }

        void buf(const IOBuf &buf) {
            *static_cast<IOBuf *>(this) = buf;
        }

        void move_to(IOBuf &target) {
            target = Movable(buf());
        }
    };

    // Create IOBuf by appending data *faster*
    class IOBufAppender {
    public:
        IOBufAppender();

        // Append `n' bytes starting from `data' to back side of the internal buffer
        // Costs 2/3 time of IOBuf.append for short data/strings on Intel(R) Xeon(R)
        // CPU E5-2620 @ 2.00GHz. Longer data/strings make differences smaller.
        // Returns 0 on success, -1 otherwise.
        int append(const void *data, size_t n);

        int append(const std::string_view &str);

        // Format integer |d| to back side of the internal buffer, which is much faster
        // than snprintf(..., "%lu", d).
        // Returns 0 on success, -1 otherwise.
        int append_decimal(long d);

        // Push the character to back side of the internal buffer.
        // Costs ~3ns while IOBuf.push_back costs ~13ns on Intel(R) Xeon(R) CPU
        // E5-2620 @ 2.00GHz
        // Returns 0 on success, -1 otherwise.
        int push_back(char c);

        IOBuf &buf() {
            shrink();
            return _buf;
        }

        void move_to(IOBuf &target) {
            target = IOBuf::Movable(buf());
        }

    protected:
        void shrink();

        int add_block();

        void *_data;
        // Saving _data_end instead of _size avoid modifying _data and _size
        // in each push_back() which is probably a hotspot.
        void *_data_end;
        IOBuf _buf;
        IOBufAsZeroCopyOutputStream _zc_stream;
    };

    // Iterate bytes of a IOBuf.
    // During iteration, the iobuf should NOT be changed.
    class IOBufBytesIterator {
    public:
        explicit IOBufBytesIterator(const IOBuf &buf);

        // Construct from another iterator.
        IOBufBytesIterator(const IOBufBytesIterator &it);

        IOBufBytesIterator(const IOBufBytesIterator &it, size_t bytes_left);

        // Returning unsigned is safer than char which would be more error prone
        // to bitwise operations. For example: in "uint32_t value = *it", value
        // is (unexpected) 4294967168 when *it returns (char)128.
        unsigned char operator*() const { return (unsigned char) *_block_begin; }

        operator const void *() const { return (const void *) !!_bytes_left; }

        void operator++();

        void operator++(int) { return operator++(); }

        // Copy at most n bytes into buf, forwarding this iterator.
        // Returns bytes copied.
        size_t copy_and_forward(void *buf, size_t n);

        size_t copy_and_forward(std::string *s, size_t n);

        // Just forward this iterator for at most n bytes.
        size_t forward(size_t n);

        // Append at most n bytes into buf, forwarding this iterator. Data are
        // referenced rather than copied.
        size_t append_and_forward(IOBuf *buf, size_t n);

        bool forward_one_block(const void **data, size_t *size);

        size_t bytes_left() const { return _bytes_left; }

    private:
        void try_next_block();

        const char *_block_begin;
        const char *_block_end;
        uint32_t _block_count;
        uint32_t _bytes_left;
        const IOBuf *_buf;
    };

    /// inlined functions


    inline turbo::ResultStatus<ssize_t> IOBuf::cut_into_file_descriptor(int fd, size_t size_hint) {
        return pcut_into_file_descriptor(fd, -1, size_hint);
    }

    inline turbo::ResultStatus<ssize_t> IOBuf::cut_multiple_into_file_descriptor(
            int fd, IOBuf* const* pieces, size_t count) {
        return pcut_multiple_into_file_descriptor(fd, -1, pieces, count);
    }

    inline int IOBuf::append_user_data(void* data, size_t size, void (*deleter)(void*)) {
        return append_user_data_with_meta(data, size, deleter, 0);
    }

    inline turbo::ResultStatus<ssize_t> IOPortal::append_from_file_descriptor(int fd, size_t max_count) {
        return pappend_from_file_descriptor(fd, -1, max_count);
    }

    inline void IOPortal::return_cached_blocks() {
        if (_block) {
            return_cached_blocks_impl(_block);
            _block = nullptr;
        }
    }

    inline void reset_block_ref(IOBuf::BlockRef& ref) {
        ref.offset = 0;
        ref.length = 0;
        ref.block = nullptr;
    }

    inline IOBuf::IOBuf() {
        reset_block_ref(_sv.refs[0]);
        reset_block_ref(_sv.refs[1]);
    }

    inline IOBuf::IOBuf(IOBuf&& rhs) {
        _sv = rhs._sv;
        new (&rhs) IOBuf;
    }

    inline IOBuf::IOBuf(const Movable& rhs) {
        _sv = rhs.value()._sv;
        new (&rhs.value()) IOBuf;
    }

    inline void IOBuf::operator=(const Movable& rhs) {
        clear();
        _sv = rhs.value()._sv;
        new (&rhs.value()) IOBuf;
    }

    inline void IOBuf::operator=(IOBuf&& rhs) {
        clear();
        _sv = rhs._sv;
        new (&rhs) IOBuf;
    }

    inline void IOBuf::operator=(const char* s) {
        clear();
        append(s);
    }

    inline void IOBuf::operator=(const std::string& s) {
        clear();
        append(s);
    }

    inline void IOBuf::swap(IOBuf& other) {
        const SmallView tmp = other._sv;
        other._sv = _sv;
        _sv = tmp;
    }

    inline int IOBuf::cut_until(IOBuf* out, char const* delim) {
        if (*delim) {
            if (!*(delim+1)) {
                return _cut_by_char(out, *delim);
            } else {
                return _cut_by_delim(out, delim, strlen(delim));
            }
        }
        return -1;
    }

    inline int IOBuf::cut_until(IOBuf* out, const std::string& delim) {
        if (delim.length() == 1UL) {
            return _cut_by_char(out, delim[0]);
        } else if (delim.length() > 1UL) {
            return _cut_by_delim(out, delim.data(), delim.length());
        } else {
            return -1;
        }
    }

    inline int IOBuf::append(const std::string& s) {
        return append(s.data(), s.length());
    }

    inline std::string IOBuf::to_string() const {
        std::string s;
        copy_to(&s);
        return s;
    }

    inline bool IOBuf::empty() const {
        return _small() ? !_sv.refs[0].block : !_bv.nbytes;
    }

    inline size_t IOBuf::length() const {
        return _small() ?
               (_sv.refs[0].length + _sv.refs[1].length) : _bv.nbytes;
    }

    inline bool IOBuf::_small() const {
        return _bv.magic >= 0;
    }

    inline size_t IOBuf::_ref_num() const {
        return _small()
               ? (!!_sv.refs[0].block + !!_sv.refs[1].block) : _bv.nref;
    }

    inline IOBuf::BlockRef& IOBuf::_front_ref() {
        return _small() ? _sv.refs[0] : _bv.refs[_bv.start];
    }

    inline const IOBuf::BlockRef& IOBuf::_front_ref() const {
        return _small() ? _sv.refs[0] : _bv.refs[_bv.start];
    }

    inline IOBuf::BlockRef& IOBuf::_back_ref() {
        return _small() ? _sv.refs[!!_sv.refs[1].block] : _bv.ref_at(_bv.nref - 1);
    }

    inline const IOBuf::BlockRef& IOBuf::_back_ref() const {
        return _small() ? _sv.refs[!!_sv.refs[1].block] : _bv.ref_at(_bv.nref - 1);
    }

    inline IOBuf::BlockRef& IOBuf::_ref_at(size_t i) {
        return _small() ? _sv.refs[i] : _bv.ref_at(i);
    }

    inline const IOBuf::BlockRef& IOBuf::_ref_at(size_t i) const {
        return _small() ? _sv.refs[i] : _bv.ref_at(i);
    }

    inline const IOBuf::BlockRef* IOBuf::_pref_at(size_t i) const {
        if (_small()) {
            return i < (size_t)(!!_sv.refs[0].block + !!_sv.refs[1].block) ? &_sv.refs[i] : nullptr;
        } else {
            return i < _bv.nref ? &_bv.ref_at(i) : nullptr;
        }
    }

    inline bool operator==(const IOBuf::BlockRef& r1, const IOBuf::BlockRef& r2) {
        return r1.offset == r2.offset && r1.length == r2.length &&
               r1.block == r2.block;
    }

    inline bool operator!=(const IOBuf::BlockRef& r1, const IOBuf::BlockRef& r2) {
        return !(r1 == r2);
    }

    inline void IOBuf::_push_back_ref(const BlockRef& r) {
        if (_small()) {
            return _push_or_move_back_ref_to_smallview<false>(r);
        } else {
            return _push_or_move_back_ref_to_bigview<false>(r);
        }
    }

    inline void IOBuf::_move_back_ref(const BlockRef& r) {
        if (_small()) {
            return _push_or_move_back_ref_to_smallview<true>(r);
        } else {
            return _push_or_move_back_ref_to_bigview<true>(r);
        }
    }

    ////////////////  IOBufCutter ////////////////
    inline size_t IOBufCutter::remaining_bytes() const {
        if (_block) {
            return (char*)_data_end - (char*)_data + _buf->size() - _buf->_front_ref().length;
        } else {
            return _buf->size();
        }
    }

    inline bool IOBufCutter::cut1(void* c) {
        if (_data == _data_end) {
            if (!load_next_ref()) {
                return false;
            }
        }
        *(char*)c = *(const char*)_data;
        _data = (char*)_data + 1;
        return true;
    }

    inline const void* IOBufCutter::fetch1() {
        if (_data == _data_end) {
            if (!load_next_ref()) {
                return nullptr;
            }
        }
        return _data;
    }

    inline size_t IOBufCutter::copy_to(void* out, size_t n) {
        size_t size = (char*)_data_end - (char*)_data;
        if (n <= size) {
            memcpy(out, _data, n);
            return n;
        }
        return slower_copy_to(out, n);
    }

    inline size_t IOBufCutter::pop_front(size_t n) {
        const size_t saved_n = n;
        do {
            const size_t size = (char*)_data_end - (char*)_data;
            if (n <= size) {
                _data = (char*)_data + n;
                return saved_n;
            }
            n -= size;
            if (!load_next_ref()) {
                return saved_n - n;
            }
        } while (true);
    }

    inline size_t IOBufCutter::cutn(std::string* out, size_t n) {
        if (n == 0) {
            return 0;
        }
        const size_t len = remaining_bytes();
        if (n > len) {
            n = len;
        }
        const size_t old_size = out->size();
        out->resize(out->size() + n);
        return cutn(&(*out)[old_size], n);
    }

    /////////////// IOBufAppender /////////////////
    inline int IOBufAppender::append(const void* src, size_t n) {
        do {
            const size_t size = (char*)_data_end - (char*)_data;
            if (n <= size) {
                memcpy(_data, src, n);
                _data = (char*)_data + n;
                return 0;
            }
            if (size != 0) {
                memcpy(_data, src, size);
                src = (const char*)src + size;
                n -= size;
            }
            if (add_block() != 0) {
                return -1;
            }
        } while (true);
    }

    inline int IOBufAppender::append(const std::string_view& str) {
        return append(str.data(), str.size());
    }

    inline int IOBufAppender::append_decimal(long d) {
        char buf[24];  // enough for decimal 64-bit integers
        size_t n = sizeof(buf);
        bool negative = false;
        if (d < 0) {
            negative = true;
            d = -d;
        }
        do {
            const long q = d / 10;
            buf[--n] = d - q * 10 + '0';
            d = q;
        } while (d);
        if (negative) {
            buf[--n] = '-';
        }
        return append(buf + n, sizeof(buf) - n);
    }

    inline int IOBufAppender::push_back(char c) {
        if (_data == _data_end) {
            if (add_block() != 0) {
                return -1;
            }
        }
        char* const p = (char*)_data;
        *p = c;
        _data = p + 1;
        return 0;
    }

    inline int IOBufAppender::add_block() {
        int size = 0;
        if (_zc_stream.next(&_data, &size)) {
            _data_end = (char*)_data + size;
            return 0;
        }
        _data = nullptr;
        _data_end = nullptr;
        return -1;
    }

    inline void IOBufAppender::shrink() {
        const size_t size = (char*)_data_end - (char*)_data;
        if (size != 0) {
            _zc_stream.back_up(size);
            _data = nullptr;
            _data_end = nullptr;
        }
    }

    inline IOBufBytesIterator::IOBufBytesIterator(const IOBuf& buf)
            : _block_begin(nullptr), _block_end(nullptr), _block_count(0),
              _bytes_left(buf.length()), _buf(&buf) {
        try_next_block();
    }

    inline IOBufBytesIterator::IOBufBytesIterator(const IOBufBytesIterator& it)
            : _block_begin(it._block_begin)
            , _block_end(it._block_end)
            , _block_count(it._block_count)
            , _bytes_left(it._bytes_left)
            , _buf(it._buf) {
    }

    inline IOBufBytesIterator::IOBufBytesIterator(
            const IOBufBytesIterator& it, size_t bytes_left)
            : _block_begin(it._block_begin)
            , _block_end(it._block_end)
            , _block_count(it._block_count)
            , _bytes_left(bytes_left)
            , _buf(it._buf) {
        //CHECK_LE(_bytes_left, it._bytes_left);
        if (_block_end > _block_begin + _bytes_left) {
            _block_end = _block_begin + _bytes_left;
        }
    }

    inline void IOBufBytesIterator::try_next_block() {
        if (_bytes_left == 0) {
            return;
        }
        std::string_view s = _buf->backing_block(_block_count++);
        _block_begin = s.data();
        _block_end = s.data() + std::min(s.size(), (size_t)_bytes_left);
    }

    inline void IOBufBytesIterator::operator++() {
        ++_block_begin;
        --_bytes_left;
        if (_block_begin == _block_end) {
            try_next_block();
        }
    }

    inline size_t IOBufBytesIterator::copy_and_forward(void* buf, size_t n) {
        size_t nc = 0;
        while (nc < n && _bytes_left != 0) {
            const size_t block_size = _block_end - _block_begin;
            const size_t to_copy = std::min(block_size, n - nc);
            memcpy((char*)buf + nc, _block_begin, to_copy);
            _block_begin += to_copy;
            _bytes_left -= to_copy;
            nc += to_copy;
            if (_block_begin == _block_end) {
                try_next_block();
            }
        }
        return nc;
    }

    inline size_t IOBufBytesIterator::copy_and_forward(std::string* s, size_t n) {
        bool resized = false;
        if (s->size() < n) {
            resized = true;
            s->resize(n);
        }
        const size_t nc = copy_and_forward(const_cast<char*>(s->data()), n);
        if (nc < n && resized) {
            s->resize(nc);
        }
        return nc;
    }

    inline size_t IOBufBytesIterator::forward(size_t n) {
        size_t nc = 0;
        while (nc < n && _bytes_left != 0) {
            const size_t block_size = _block_end - _block_begin;
            const size_t to_copy = std::min(block_size, n - nc);
            _block_begin += to_copy;
            _bytes_left -= to_copy;
            nc += to_copy;
            if (_block_begin == _block_end) {
                try_next_block();
            }
        }
        return nc;
    }


}  // namespace turbo::files_internal

#endif  // TURBO_SYSTEM_IO_IOBUF_H_
