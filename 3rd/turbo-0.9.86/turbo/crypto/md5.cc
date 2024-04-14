
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "turbo/crypto/md5.h"
#include "turbo/strings/escaping.h"
#include "turbo/base/bits.h"
#include "turbo/strings/string_view.h"

namespace turbo {

    namespace digest_detail {
        TURBO_DISABLE_CLANG_WARNING(-Wsign-conversion)
        static TURBO_FORCE_INLINE uint32_t min(uint32_t x, uint32_t y) {
            return x < y ? x : y;
        }

        static TURBO_FORCE_INLINE uint32_t load32l(const uint8_t *y) {
            uint32_t res = 0;
            for (size_t i = 0; i != 4; ++i)
                res |= uint32_t(y[i]) << (i * 8);
            return res;
        }

        static TURBO_FORCE_INLINE void store32l(uint32_t x, uint8_t *y) {
            for (size_t i = 0; i != 4; ++i)
                y[i] = (x >> (i * 8)) & 255;
        }

        static TURBO_FORCE_INLINE void store64l(uint64_t x, uint8_t *y) {
            for (size_t i = 0; i != 8; ++i)
                y[i] = (x >> (i * 8)) & 255;
        }

        static TURBO_FORCE_INLINE
        uint32_t F(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return (z ^ (x & (y ^ z)));
        }

        static TURBO_FORCE_INLINE
        uint32_t G(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return (y ^ (z & (y ^ x)));
        }

        static TURBO_FORCE_INLINE
        uint32_t H(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return (x ^ y ^ z);
        }

        static TURBO_FORCE_INLINE
        uint32_t I(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return (y ^ (x | (~z)));
        }

        static TURBO_FORCE_INLINE void FF(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
                                         uint32_t M, uint32_t s, uint32_t t) {
            a = (a + F(b, c, d) + M + t);
            a = rotl(a, s) + b;
        }

        static TURBO_FORCE_INLINE void GG(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
                                         uint32_t M, uint32_t s, uint32_t t) {
            a = (a + G(b, c, d) + M + t);
            a = rotl(a, s) + b;
        }

        static TURBO_FORCE_INLINE void HH(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
                                         uint32_t M, uint32_t s, uint32_t t) {
            a = (a + H(b, c, d) + M + t);
            a = rotl(a, s) + b;
        }

        static TURBO_FORCE_INLINE void II(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
                                         uint32_t M, uint32_t s, uint32_t t) {
            a = (a + I(b, c, d) + M + t);
            a = rotl(a, s) + b;
        }

        static const uint8_t kWorder[64] = {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12,
                5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2,
                0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9
        };

        static const uint8_t kRorder[64] = {
                7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
        };

        static const uint32_t kOrder[64] = {
                0xd76aa478UL, 0xe8c7b756UL, 0x242070dbUL, 0xc1bdceeeUL, 0xf57c0fafUL,
                0x4787c62aUL, 0xa8304613UL, 0xfd469501UL, 0x698098d8UL, 0x8b44f7afUL,
                0xffff5bb1UL, 0x895cd7beUL, 0x6b901122UL, 0xfd987193UL, 0xa679438eUL,
                0x49b40821UL, 0xf61e2562UL, 0xc040b340UL, 0x265e5a51UL, 0xe9b6c7aaUL,
                0xd62f105dUL, 0x02441453UL, 0xd8a1e681UL, 0xe7d3fbc8UL, 0x21e1cde6UL,
                0xc33707d6UL, 0xf4d50d87UL, 0x455a14edUL, 0xa9e3e905UL, 0xfcefa3f8UL,
                0x676f02d9UL, 0x8d2a4c8aUL, 0xfffa3942UL, 0x8771f681UL, 0x6d9d6122UL,
                0xfde5380cUL, 0xa4beea44UL, 0x4bdecfa9UL, 0xf6bb4b60UL, 0xbebfbc70UL,
                0x289b7ec6UL, 0xeaa127faUL, 0xd4ef3085UL, 0x04881d05UL, 0xd9d4d039UL,
                0xe6db99e5UL, 0x1fa27cf8UL, 0xc4ac5665UL, 0xf4292244UL, 0x432aff97UL,
                0xab9423a7UL, 0xfc93a039UL, 0x655b59c3UL, 0x8f0ccc92UL, 0xffeff47dUL,
                0x85845dd1UL, 0x6fa87e4fUL, 0xfe2ce6e0UL, 0xa3014314UL, 0x4e0811a1UL,
                0xf7537e82UL, 0xbd3af235UL, 0x2ad7d2bbUL, 0xeb86d391UL
        };

        static void md5_compress(uint32_t state[4], const uint8_t *buf) {
            uint32_t i, W[16], a, b, c, d, t;

            // copy the state into 512-bits into W[0..15]
            for (i = 0; i < 16; i++) {
                W[i] = load32l(buf + (4 * i));
            }

            // copy state
            a = state[0];
            b = state[1];
            c = state[2];
            d = state[3];

            for (i = 0; i < 16; ++i) {
                FF(a, b, c, d, W[kWorder[i]], kRorder[i], kOrder[i]);
                t = d, d = c, c = b, b = a, a = t;
            }

            for (; i < 32; ++i) {
                GG(a, b, c, d, W[kWorder[i]], kRorder[i], kOrder[i]);
                t = d, d = c, c = b, b = a, a = t;
            }

            for (; i < 48; ++i) {
                HH(a, b, c, d, W[kWorder[i]], kRorder[i], kOrder[i]);
                t = d, d = c, c = b, b = a, a = t;
            }

            for (; i < 64; ++i) {
                II(a, b, c, d, W[kWorder[i]], kRorder[i], kOrder[i]);
                t = d, d = c, c = b, b = a, a = t;
            }

            state[0] = state[0] + a;
            state[1] = state[1] + b;
            state[2] = state[2] + c;
            state[3] = state[3] + d;
        }
        TURBO_RESTORE_CLANG_WARNING();
    }  // namespace digest_detail

    MD5::MD5() {
        _curlen = 0;
        _length = 0;
        _state[0] = 0x67452301UL;
        _state[1] = 0xefcdab89UL;
        _state[2] = 0x98badcfeUL;
        _state[3] = 0x10325476UL;
    }

    MD5::MD5(const void *data, uint32_t size) : MD5() {
        process(data, size);
    }

    MD5::MD5(const std::string &str) : MD5() {
        process(str);
    }

    void MD5::process(const void *data, uint32_t size) {
        const uint32_t block_size = sizeof(MD5::_buf);
        auto in = static_cast<const uint8_t *>(data);

        while (size > 0) {
            if (_curlen == 0 && size >= block_size) {
                digest_detail::md5_compress(_state, in);
                _length += block_size * 8;
                in += block_size;
                size -= block_size;
            } else {
                uint32_t n = digest_detail::min(size, (block_size - _curlen));
                uint8_t *b = _buf + _curlen;
                for (const uint8_t *a = in; a != in + n; ++a, ++b) {
                    *b = *a;
                }
                _curlen += n;
                in += n;
                size -= n;

                if (_curlen == block_size) {
                    digest_detail::md5_compress(_state, _buf);
                    _length += 8 * block_size;
                    _curlen = 0;
                }
            }
        }
    }

    void MD5::process(const std::string &str) {
        TURBO_DISABLE_CLANG_WARNING(-Wshorten-64-to-32)
        return process(str.data(), str.size());
        TURBO_RESTORE_CLANG_WARNING()
    }

    void MD5::finalize(void *digest) {
        // Increase the length of the message
        _length += _curlen * 8;

        // Append the '1' bit
        _buf[_curlen++] = static_cast<uint8_t>(0x80);

        // If the _length is currently above 56 bytes we append zeros then
        // md5_compress().  Then we can fall back to padding zeros and length
        // encoding like normal.
        if (_curlen > 56) {
            while (_curlen < 64)
                _buf[_curlen++] = 0;
            digest_detail::md5_compress(_state, _buf);
            _curlen = 0;
        }

        // Pad up to 56 bytes of zeroes
        while (_curlen < 56)
            _buf[_curlen++] = 0;

        // Store length
        digest_detail::store64l(_length, _buf + 56);
        digest_detail::md5_compress(_state, _buf);

        // Copy output
        for (size_t i = 0; i < 4; i++) {
            digest_detail::store32l(
                    _state[i], static_cast<uint8_t *>(digest) + (4 * i));
        }
    }

    std::string MD5::digest() {
        std::string out(kDigestLength, '0');
        finalize(const_cast<char *>(out.data()));
        return out;
    }

    std::string MD5::digest_hex() {
        uint8_t digest[kDigestLength];
        finalize(digest);
        return turbo::bytes_to_hex(std::string_view(reinterpret_cast<char*>(digest), kDigestLength));
    }


    std::string md5_hex(const void *data, uint32_t size) {
        return MD5(data, size).digest_hex();
    }

    std::string md5_hex(const std::string &str) {
        return MD5(str).digest_hex();
    }

}  // namespace turbo
