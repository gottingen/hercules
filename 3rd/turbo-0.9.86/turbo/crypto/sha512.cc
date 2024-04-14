
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "turbo/crypto/sha512.h"
#include "turbo/base/bits.h"
#include "turbo/platform/port.h"
#include "turbo/strings/escaping.h"
#include "turbo/strings/string_view.h"

namespace turbo {


    namespace digest_detail {

        static const uint64_t K[80] = {
                0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
                0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
                0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
                0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
                0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
                0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
                0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
                0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
                0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
                0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
                0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
                0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
                0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
                0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
                0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
                0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
                0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
                0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
                0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
                0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
                0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
                0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
                0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
                0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
                0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
                0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
                0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
        };

        static TURBO_FORCE_INLINE uint32_t min(uint32_t x, uint32_t y) {
            return x < y ? x : y;
        }

        static TURBO_FORCE_INLINE void store64(uint64_t x, unsigned char *y) {
            for (int i = 0; i != 8; ++i)
                y[i] = (x >> ((7 - i) * 8)) & 255;
        }

        static TURBO_FORCE_INLINE uint64_t load64(const unsigned char *y) {
            uint64_t res = 0;
            for (int i = 0; i != 8; ++i)
                res |= uint64_t(y[i]) << ((7 - i) * 8);
            return res;
        }

        static TURBO_FORCE_INLINE
        uint64_t Ch(const uint64_t &x, const uint64_t &y, const uint64_t &z) {
            return z ^ (x & (y ^ z));
        }

        static TURBO_FORCE_INLINE
        uint64_t Maj(const uint64_t &x, const uint64_t &y, const uint64_t &z) {
            return ((x | y) & z) | (x & y);
        }

        static TURBO_FORCE_INLINE uint64_t Sh(uint64_t x, uint64_t n) {
            return x >> n;
        }

        static TURBO_FORCE_INLINE uint64_t Sigma0(uint64_t x) {
            return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39);
        }

        static TURBO_FORCE_INLINE uint64_t Sigma1(uint64_t x) {
            return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41);
        }

        static TURBO_FORCE_INLINE uint64_t Gamma0(uint64_t x) {
            return rotr(x, 1) ^ rotr(x, 8) ^ Sh(x, 7);
        }

        static TURBO_FORCE_INLINE uint64_t Gamma1(uint64_t x) {
            return rotr(x, 19) ^ rotr(x, 61) ^ Sh(x, 6);
        }

        static void sha512_compress(uint64_t state[8], const uint8_t *buf) {
            uint64_t S[8], W[80], t0, t1;

            // Copy _state into S
            for (int i = 0; i < 8; i++)
                S[i] = state[i];

            // Copy the state into 1024-bits into W[0..15]
            for (int i = 0; i < 16; i++)
                W[i] = load64(buf + (8 * i));

            // Fill W[16..79]
            for (int i = 16; i < 80; i++)
                W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];

            // Compress
            auto RND =
                    [&](uint64_t a, uint64_t b, uint64_t c, uint64_t &d, uint64_t e,
                        uint64_t f, uint64_t g, uint64_t &h, uint64_t i) {
                        t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
                        t1 = Sigma0(a) + Maj(a, b, c);
                        d += t0;
                        h = t0 + t1;
                    };

            for (size_t i = 0; i < 80; i += 8) {
                RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], i + 0);
                RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], i + 1);
                RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], i + 2);
                RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], i + 3);
                RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], i + 4);
                RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], i + 5);
                RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], i + 6);
                RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], i + 7);
            }

            // Feedback
            for (int i = 0; i < 8; i++)
                state[i] = state[i] + S[i];
        }

    } // namespace digest_detail

    SHA512::SHA512() {
        _curlen = 0;
        _length = 0;
        _state[0] = 0x6a09e667f3bcc908ULL;
        _state[1] = 0xbb67ae8584caa73bULL;
        _state[2] = 0x3c6ef372fe94f82bULL;
        _state[3] = 0xa54ff53a5f1d36f1ULL;
        _state[4] = 0x510e527fade682d1ULL;
        _state[5] = 0x9b05688c2b3e6c1fULL;
        _state[6] = 0x1f83d9abfb41bd6bULL;
        _state[7] = 0x5be0cd19137e2179ULL;
    }

    SHA512::SHA512(const void *data, uint32_t size) : SHA512() {
        process(data, size);
    }

    SHA512::SHA512(const std::string &str) : SHA512() {
        process(str);
    }

    void SHA512::process(const void *data, uint32_t size) {
        const uint32_t block_size = sizeof(SHA512::_buf);
        auto in = static_cast<const uint8_t *>(data);

        while (size > 0) {
            if (_curlen == 0 && size >= block_size) {
                digest_detail::sha512_compress(_state, in);
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
                    digest_detail::sha512_compress(_state, _buf);
                    _length += 8 * block_size;
                    _curlen = 0;
                }
            }
        }
    }

    void SHA512::process(const std::string &str) {
        TURBO_DISABLE_CLANG_WARNING(-Wshorten-64-to-32)
        return process(str.data(), str.size());
        TURBO_RESTORE_CLANG_WARNING()
    }

    void SHA512::finalize(void *digest) {
        // Increase the length of the message
        _length += _curlen * 8ULL;

        // Append the '1' bit
        _buf[_curlen++] = static_cast<uint8_t>(0x80);

        // If the length is currently above 112 bytes we append zeros then compress.
        // Then we can fall back to padding zeros and length encoding like normal.
        if (_curlen > 112) {
            while (_curlen < 128)
                _buf[_curlen++] = 0;
            digest_detail::sha512_compress(_state, _buf);
            _curlen = 0;
        }

        // Pad up to 120 bytes of zeroes
        // note: that from 112 to 120 is the 64 MSB of the length.  We assume that
        // you won't hash 2^64 bits of data... :-)
        while (_curlen < 120)
            _buf[_curlen++] = 0;

        // Store length
        digest_detail::store64(_length, _buf + 120);
        digest_detail::sha512_compress(_state, _buf);

        // Copy output
        for (int i = 0; i < 8; i++) {
            digest_detail::store64(
                    _state[i], static_cast<uint8_t *>(digest) + (8 * i));
        }
    }

    std::string SHA512::digest() {
        std::string out(kDigestLength, '0');
        finalize(const_cast<char *>(out.data()));
        return out;
    }

    std::string SHA512::digest_hex() {
        uint8_t digest[kDigestLength];
        finalize(digest);
        return turbo::bytes_to_hex(std::string_view(reinterpret_cast<char*>(digest), kDigestLength));
    }


    std::string sha512_hex(const void *data, uint32_t size) {
        return SHA512(data, size).digest_hex();
    }

    std::string sha512_hex(const std::string &str) {
        return SHA512(str).digest_hex();
    }



}  // namespace turbo
