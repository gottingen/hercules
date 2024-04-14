
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "turbo/crypto/sha1.h"
#include "turbo/base/bits.h"
#include "turbo/platform/port.h"
#include "turbo/strings/escaping.h"
#include "turbo/strings/string_view.h"

namespace turbo {

    namespace digest_detail {

        static TURBO_FORCE_INLINE uint32_t min(uint32_t x, uint32_t y) {
            return x < y ? x : y;
        }

        static TURBO_FORCE_INLINE void store64h(uint64_t x, unsigned char *y) {
            for (int i = 0; i != 8; ++i)
                y[i] = (x >> ((7 - i) * 8)) & 255;
        }

        static TURBO_FORCE_INLINE uint32_t load32h(const uint8_t *y) {
            return (uint32_t(y[0]) << 24) | (uint32_t(y[1]) << 16) |
                   (uint32_t(y[2]) << 8) | (uint32_t(y[3]) << 0);
        }

        static TURBO_FORCE_INLINE void store32h(uint32_t x, uint8_t *y) {
            for (int i = 0; i != 4; ++i)
                y[i] = (x >> ((3 - i) * 8)) & 255;
        }

        static TURBO_FORCE_INLINE
        uint32_t F0(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return (z ^ (x & (y ^ z)));
        }

        static TURBO_FORCE_INLINE
        uint32_t F1(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return (x ^ y ^ z);
        }

        static TURBO_FORCE_INLINE
        uint32_t F2(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return ((x & y) | (z & (x | y)));
        }

        static TURBO_FORCE_INLINE
        uint32_t F3(const uint32_t &x, const uint32_t &y, const uint32_t &z) {
            return (x ^ y ^ z);
        }

        static void sha1_compress(uint32_t state[4], const uint8_t *buf) {
            uint32_t a, b, c, d, e, W[80], i, t;

            /* copy the state into 512-bits into W[0..15] */
            for (i = 0; i < 16; i++) {
                W[i] = load32h(buf + (4 * i));
            }

            /* copy state */
            a = state[0];
            b = state[1];
            c = state[2];
            d = state[3];
            e = state[4];

            /* expand it */
            for (i = 16; i < 80; i++) {
                W[i] = rotl(W[i - 3] ^ W[i - 8] ^ W[i - 14] ^ W[i - 16], 1);
            }

            /* compress */
            for (i = 0; i < 20; ++i) {
                e = (rotl(a, 5) + F0(b, c, d) + e + W[i] + 0x5a827999UL);
                b = rotl(b, 30);
                t = e, e = d, d = c, c = b, b = a, a = t;
            }
            for (; i < 40; ++i) {
                e = (rotl(a, 5) + F1(b, c, d) + e + W[i] + 0x6ed9eba1UL);
                b = rotl(b, 30);
                t = e, e = d, d = c, c = b, b = a, a = t;
            }
            for (; i < 60; ++i) {
                e = (rotl(a, 5) + F2(b, c, d) + e + W[i] + 0x8f1bbcdcUL);
                b = rotl(b, 30);
                t = e, e = d, d = c, c = b, b = a, a = t;
            }
            for (; i < 80; ++i) {
                e = (rotl(a, 5) + F3(b, c, d) + e + W[i] + 0xca62c1d6UL);
                b = rotl(b, 30);
                t = e, e = d, d = c, c = b, b = a, a = t;
            }

            /* store */
            state[0] = state[0] + a;
            state[1] = state[1] + b;
            state[2] = state[2] + c;
            state[3] = state[3] + d;
            state[4] = state[4] + e;
        }

    } // namespace digest_detail

    SHA1::SHA1() {
        _curlen = 0;
        _length = 0;
        _state[0] = 0x67452301UL;
        _state[1] = 0xefcdab89UL;
        _state[2] = 0x98badcfeUL;
        _state[3] = 0x10325476UL;
        _state[4] = 0xc3d2e1f0UL;
    }

    SHA1::SHA1(const void *data, uint32_t size) : SHA1() {
        process(data, size);
    }

    SHA1::SHA1(const std::string &str) : SHA1() {
        process(str);
    }

    void SHA1::process(const void *data, uint32_t size) {
        const uint32_t block_size = sizeof(SHA1::_buf);
        auto in = static_cast<const uint8_t *>(data);

        while (size > 0) {
            if (_curlen == 0 && size >= block_size) {
                digest_detail::sha1_compress(_state, in);
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
                    digest_detail::sha1_compress(_state, _buf);
                    _length += 8 * block_size;
                    _curlen = 0;
                }
            }
        }
    }

    void SHA1::process(const std::string &str) {
        TURBO_DISABLE_CLANG_WARNING(-Wshorten-64-to-32)
        return process(str.data(), str.size());
        TURBO_RESTORE_CLANG_WARNING()
    }

    void SHA1::finalize(void *digest) {
        // Increase the length of the message
        _length += _curlen * 8;

        // Append the '1' bit
        _buf[_curlen++] = static_cast<uint8_t>(0x80);

        // If the _length is currently above 56 bytes we append zeros then
        // sha1_compress().  Then we can fall back to padding zeros and length
        // encoding like normal.
        if (_curlen > 56) {
            while (_curlen < 64)
                _buf[_curlen++] = 0;
            digest_detail::sha1_compress(_state, _buf);
            _curlen = 0;
        }

        // Pad up to 56 bytes of zeroes
        while (_curlen < 56)
            _buf[_curlen++] = 0;

        // Store length
        digest_detail::store64h(_length, _buf + 56);
        digest_detail::sha1_compress(_state, _buf);

        // Copy output
        for (size_t i = 0; i < 5; i++)
            digest_detail::store32h(_state[i], static_cast<uint8_t *>(digest) + (4 * i));
    }

    std::string SHA1::digest() {
        std::string out(kDigestLength, '0');
        finalize(const_cast<char *>(out.data()));
        return out;
    }

    std::string SHA1::digest_hex() {
        uint8_t digest[kDigestLength];
        finalize(digest);
        return turbo::bytes_to_hex(std::string_view(reinterpret_cast<char*>(digest), kDigestLength));
    }


    std::string sha1_hex(const void *data, uint32_t size) {
        return SHA1(data, size).digest_hex();
    }

    std::string sha1_hex(const std::string &str) {
        return SHA1(str).digest_hex();
    }

}  // namespace turbo
