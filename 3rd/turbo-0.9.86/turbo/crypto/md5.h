
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef TURBO_CRYPTO_MD5_H_
#define TURBO_CRYPTO_MD5_H_

#include <cstdint>
#include <string>

namespace turbo {

    class MD5 {
      public:
        //! construct empty object.
        MD5();

        //! construct context and process data range
        MD5(const void *data, uint32_t size);

        //! construct context and process string
        explicit MD5(const std::string &str);

        //! process more data
        void process(const void *data, uint32_t size);

        //! process more data
        void process(const std::string &str);

        //! digest length in bytes
        static constexpr size_t kDigestLength = 16;

        //! finalize computation and output 16 byte (128 bit) digest
        void finalize(void *digest);

        //! finalize computation and return 16 byte (128 bit) digest
        std::string digest();

        //! finalize computation and return 16 byte (128 bit) digest hex encoded
        std::string digest_hex();

      private:
        uint64_t _length;
        uint32_t _state[4];
        uint32_t _curlen;
        uint8_t _buf[64];
    };

    //! process data and return 16 byte (128 bit) digest hex encoded
    std::string md5_hex(const void *data, uint32_t size);

    //! process data and return 16 byte (128 bit) digest hex encoded
    std::string md5_hex(const std::string &str);


}  // namespace turbo

#endif  // TURBO_CRYPTO_MD5_H_
