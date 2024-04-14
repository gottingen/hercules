
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "turbo/crypto/sha256.h"
#include "turbo/base/bits.h"
#include "turbo/platform/port.h"
#include "turbo/strings/escaping.h"
#include "turbo/strings/string_view.h"

namespace turbo {

    typedef uint32_t u32;
    typedef uint64_t u64;

    namespace {

        static const u32 K[64] = {
                0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL,
                0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL, 0xd807aa98UL, 0x12835b01UL,
                0x243185beUL, 0x550c7dc3UL, 0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL,
                0xc19bf174UL, 0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL,
                0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL, 0x983e5152UL,
                0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL,
                0x06ca6351UL, 0x14292967UL, 0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL,
                0x53380d13UL, 0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
                0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL,
                0xd6990624UL, 0xf40e3585UL, 0x106aa070UL, 0x19a4c116UL, 0x1e376c08UL,
                0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL,
                0x682e6ff3UL, 0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
                0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL
        };

        static TURBO_FORCE_INLINE u32 min(u32 x, u32 y) {
            return x < y ? x : y;
        }

        static TURBO_FORCE_INLINE u32 load32(const uint8_t *y) {
            return (u32(y[0]) << 24) | (u32(y[1]) << 16) |
                   (u32(y[2]) << 8) | (u32(y[3]) << 0);
        }

        static TURBO_FORCE_INLINE void store64(u64 x, uint8_t *y) {
            for (int i = 0; i != 8; ++i)
                y[i] = (x >> ((7 - i) * 8)) & 255;
        }

        static TURBO_FORCE_INLINE void store32(u32 x, uint8_t *y) {
            for (int i = 0; i != 4; ++i)
                y[i] = (x >> ((3 - i) * 8)) & 255;
        }

        static TURBO_FORCE_INLINE u32 Ch(u32 x, u32 y, u32 z) {
            return z ^ (x & (y ^ z));
        }

        static TURBO_FORCE_INLINE u32 Maj(u32 x, u32 y, u32 z) {
            return ((x | y) & z) | (x & y);
        }

        static TURBO_FORCE_INLINE u32 Sh(u32 x, u32 n) {
            return x >> n;
        }

        static TURBO_FORCE_INLINE u32 Sigma0(u32 x) {
            return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
        }

        static TURBO_FORCE_INLINE u32 Sigma1(u32 x) {
            return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
        }

        static TURBO_FORCE_INLINE u32 Gamma0(u32 x) {
            return rotr(x, 7) ^ rotr(x, 18) ^ Sh(x, 3);
        }

        static TURBO_FORCE_INLINE u32 Gamma1(u32 x) {
            return rotr(x, 17) ^ rotr(x, 19) ^ Sh(x, 10);
        }

        static void sha256_compress(uint32_t state[8], const uint8_t *buf) {
            u32 S[8], W[64], t0, t1, t;

            // Copy state into S
            for (size_t i = 0; i < 8; i++)
                S[i] = state[i];

            // Copy the state into 512-bits into W[0..15]
            for (size_t i = 0; i < 16; i++)
                W[i] = load32(buf + (4 * i));

            // Fill W[16..63]
            for (size_t i = 16; i < 64; i++)
                W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];

            // Compress
            auto RND =
                    [&](u32 a, u32 b, u32 c, u32 &d, u32 e, u32 f, u32 g, u32 &h, u32 i) {
                        t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
                        t1 = Sigma0(a) + Maj(a, b, c);
                        d += t0;
                        h = t0 + t1;
                    };

            for (uint32_t i = 0; i < 64; ++i) {
                RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], i);
                t = S[7], S[7] = S[6], S[6] = S[5], S[5] = S[4],
                S[4] = S[3], S[3] = S[2], S[2] = S[1], S[1] = S[0], S[0] = t;
            }

            // Feedback
            for (size_t i = 0; i < 8; i++)
                state[i] = state[i] + S[i];
        }

    } // namespace

    SHA256::SHA256() {
        _curlen = 0;
        _length = 0;
        _state[0] = 0x6A09E667UL;
        _state[1] = 0xBB67AE85UL;
        _state[2] = 0x3C6EF372UL;
        _state[3] = 0xA54FF53AUL;
        _state[4] = 0x510E527FUL;
        _state[5] = 0x9B05688CUL;
        _state[6] = 0x1F83D9ABUL;
        _state[7] = 0x5BE0CD19UL;
    }

    SHA256::SHA256(const void *data, uint32_t size)
            : SHA256() {
        process(data, size);
    }

    SHA256::SHA256(const std::string &str)
            : SHA256() {
        process(str);
    }

    void SHA256::process(const void *data, u32 size) {
        const u32 block_size = sizeof(SHA256::_buf);
        auto in = static_cast<const uint8_t *>(data);

        while (size > 0) {
            if (_curlen == 0 && size >= block_size) {
                sha256_compress(_state, in);
                _length += block_size * 8;
                in += block_size;
                size -= block_size;
            } else {
                u32 n = min(size, (block_size - _curlen));
                std::copy(in, in + n, _buf + _curlen);
                _curlen += n;
                in += n;
                size -= n;

                if (_curlen == block_size) {
                    sha256_compress(_state, _buf);
                    _length += 8 * block_size;
                    _curlen = 0;
                }
            }
        }
    }

    void SHA256::process(const std::string &str) {
        TURBO_DISABLE_CLANG_WARNING(-Wshorten-64-to-32);
        return process(str.data(), str.size());
        TURBO_RESTORE_CLANG_WARNING();
    }

    void SHA256::finalize(void *digest) {
        // Increase the length of the message
        _length += _curlen * 8;

        // Append the '1' bit
        _buf[_curlen++] = static_cast<uint8_t>(0x80);

        // If the _length is currently above 56 bytes we append zeros then
        // sha256_compress().  Then we can fall back to padding zeros and length
        // encoding like normal.
        if (_curlen > 56) {
            while (_curlen < 64)
                _buf[_curlen++] = 0;
            sha256_compress(_state, _buf);
            _curlen = 0;
        }

        // Pad up to 56 bytes of zeroes
        while (_curlen < 56)
            _buf[_curlen++] = 0;

        // Store length
        store64(_length, _buf + 56);
        sha256_compress(_state, _buf);

        // Copy output
        for (size_t i = 0; i < 8; i++)
            store32(_state[i], static_cast<uint8_t *>(digest) + (4 * i));
    }

    std::string SHA256::digest() {
        std::string out(kDigestLength, '0');
        finalize(const_cast<char *>(out.data()));
        return out;
    }

    std::string SHA256::digest_hex() {
        uint8_t digest[kDigestLength];
        finalize(digest);
        return turbo::bytes_to_hex(std::string_view(reinterpret_cast<char*>(digest), kDigestLength));
    }

    std::string sha256_hex(const void *data, uint32_t size) {
        return SHA256(data, size).digest_hex();
    }

    std::string sha256_hex(const std::string &str) {
        return SHA256(str).digest_hex();
    }


}  // namespace turbo
