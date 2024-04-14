
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef TURBO_CRYPTO_SHA1_H_
#define TURBO_CRYPTO_SHA1_H_

#include <cstdint>
#include <string>

namespace turbo {

    /*!
     * SHA-1 processor without external dependencies.
     */
    class SHA1 {
      public:
        //! construct empty object.
        SHA1();

        //! construct context and process data range
        SHA1(const void *data, uint32_t size);

        //! construct context and process string
        explicit SHA1(const std::string &str);

        //! process more data
        void process(const void *data, uint32_t size);

        //! process more data
        void process(const std::string &str);

        //! digest length in bytes
        static constexpr size_t kDigestLength = 20;

        //! finalize computation and output 20 byte (160 bit) digest
        void finalize(void *digest);

        //! finalize computation and return 20 byte (160 bit) digest
        std::string digest();

        //! finalize computation and return 20 byte (160 bit) digest hex encoded
        std::string digest_hex();

      private:
        uint64_t _length;
        uint32_t _state[5];
        uint32_t _curlen;
        uint8_t _buf[64];
    };

    //! process data and return 20 byte (160 bit) digest hex encoded
    std::string sha1_hex(const void *data, uint32_t size);

    //! process data and return 20 byte (160 bit) digest hex encoded
    std::string sha1_hex(const std::string &str);

}  // namespace turbo

#endif  // TURBO_CRYPTO_SHA1_H_
