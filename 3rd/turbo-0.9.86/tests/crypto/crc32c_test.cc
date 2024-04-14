// Copyright 2022 The Turbo Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo/crypto/crc32c.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"
#include "turbo/crypto/internal/crc32c.h"
#include "turbo/format/format.h"
#include "turbo/strings/string_view.h"

namespace {

    TEST_CASE("CRC32C, RFC3720") {
        // Test the results of the vectors from
        // https://www.rfc-editor.org/rfc/rfc3720#appendix-B.4
        char data[32];

        // 32 bytes of ones.
        memset(data, 0, sizeof(data));
        CHECK_EQ(turbo::ComputeCrc32c(std::string_view(data, sizeof(data))),
                  turbo::crc32c_t{0x8a9136aa});

        // 32 bytes of ones.
        memset(data, 0xff, sizeof(data));
        CHECK_EQ(turbo::ComputeCrc32c(std::string_view(data, sizeof(data))),
                  turbo::crc32c_t{0x62a8ab43});

        // 32 incrementing bytes.
        for (int i = 0; i < 32; ++i) data[i] = static_cast<char>(i);
        CHECK_EQ(turbo::ComputeCrc32c(std::string_view(data, sizeof(data))),
                  turbo::crc32c_t{0x46dd794e});

        // 32 decrementing bytes.
        for (int i = 0; i < 32; ++i) data[i] = static_cast<char>(31 - i);
        CHECK_EQ(turbo::ComputeCrc32c(std::string_view(data, sizeof(data))),
                  turbo::crc32c_t{0x113fdb5c});

        // An iSCSI - SCSI Read (10) Command PDU.
        constexpr uint8_t cmd[48] = {
                0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
                0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        };
        CHECK_EQ(turbo::ComputeCrc32c(std::string_view(
                reinterpret_cast<const char *>(cmd), sizeof(cmd))),
                  turbo::crc32c_t{0xd9963a56});
    }

    std::string TestString(size_t len) {
        std::string result;
        result.reserve(len);
        for (size_t i = 0; i < len; ++i) {
            result.push_back(static_cast<char>(i % 256));
        }
        return result;
    }

    TEST_CASE("CRC32C, Compute") {
        CHECK_EQ(turbo::ComputeCrc32c(""), turbo::crc32c_t{0});
        CHECK_EQ(turbo::ComputeCrc32c("hello world"), turbo::crc32c_t{0xc99465aa});
    }

    TEST_CASE("CRC32C, Extend") {
        uint32_t base = 0xC99465AA;  // CRC32C of "Hello World"
        std::string extension = "Extension String";

        CHECK_EQ(
                turbo::ExtendCrc32c(turbo::crc32c_t{base}, extension),
                turbo::crc32c_t{0xD2F65090});  // CRC32C of "Hello WorldExtension String"
    }

    TEST_CASE("CRC32C, ExtendByZeroes") {
        std::string base = "hello world";
        turbo::crc32c_t base_crc = turbo::crc32c_t{0xc99465aa};

        constexpr size_t kExtendByValues[] = {100, 10000, 100000};
        for (const size_t extend_by: kExtendByValues) {
            CAPTURE(extend_by);
            turbo::crc32c_t crc2 = turbo::ExtendCrc32cByZeroes(base_crc, extend_by);
            CHECK_EQ(crc2, turbo::ComputeCrc32c(base + std::string(extend_by, '\0')));
        }
    }

    TEST_CASE("CRC32C, UnextendByZeroes") {
        constexpr size_t kExtendByValues[] = {2, 200, 20000, 200000, 20000000};
        constexpr size_t kUnextendByValues[] = {0, 100, 10000, 100000, 10000000};

        for (auto seed_crc: {turbo::crc32c_t{0}, turbo::crc32c_t{0xc99465aa}}) {
            CAPTURE(seed_crc);
            for (const size_t size_1: kExtendByValues) {
                for (const size_t size_2: kUnextendByValues) {
                    size_t extend_size = std::max(size_1, size_2);
                    size_t unextend_size = std::min(size_1, size_2);
                    CAPTURE(extend_size);
                    CAPTURE(unextend_size);

                    // Extending by A zeroes an unextending by B<A zeros should be identical
                    // to extending by A-B zeroes.
                    turbo::crc32c_t crc1 = seed_crc;
                    crc1 = turbo::ExtendCrc32cByZeroes(crc1, extend_size);
                    crc1 = turbo::crc_internal::UnextendCrc32cByZeroes(crc1, unextend_size);

                    turbo::crc32c_t crc2 = seed_crc;
                    crc2 = turbo::ExtendCrc32cByZeroes(crc2, extend_size - unextend_size);

                    CHECK_EQ(crc1, crc2);
                }
            }
        }

        constexpr size_t kSizes[] = {0, 1, 100, 10000};
        for (const size_t size: kSizes) {
            CAPTURE(size);
            std::string string_before = TestString(size);
            std::string string_after = string_before + std::string(size, '\0');

            turbo::crc32c_t crc_before = turbo::ComputeCrc32c(string_before);
            turbo::crc32c_t crc_after = turbo::ComputeCrc32c(string_after);

            CHECK_EQ(crc_before,
                      turbo::crc_internal::UnextendCrc32cByZeroes(crc_after, size));
        }
    }

    TEST_CASE("CRC32C, Concat") {
        std::string hello = "Hello, ";
        std::string world = "world!";
        std::string hello_world = turbo::format("{}{}", hello, world);

        turbo::crc32c_t crc_a = turbo::ComputeCrc32c(hello);
        turbo::crc32c_t crc_b = turbo::ComputeCrc32c(world);
        turbo::crc32c_t crc_ab = turbo::ComputeCrc32c(hello_world);

        CHECK_EQ(turbo::ConcatCrc32c(crc_a, crc_b, world.size()), crc_ab);
    }

    TEST_CASE("CRC32C, Memcpy") {
        constexpr size_t kBytesSize[] = {0, 1, 20, 500, 100000};
        for (size_t bytes: kBytesSize) {
            CAPTURE(bytes);
            std::string sample_string = TestString(bytes);
            std::string target_buffer = std::string(bytes, '\0');

            turbo::crc32c_t memcpy_crc =
                    turbo::MemcpyCrc32c(&(target_buffer[0]), sample_string.data(), bytes);
            turbo::crc32c_t compute_crc = turbo::ComputeCrc32c(sample_string);

            CHECK_EQ(memcpy_crc, compute_crc);
            CHECK_EQ(sample_string, target_buffer);
        }
    }

    TEST_CASE("CRC32C, RemovePrefix") {
        std::string hello = "Hello, ";
        std::string world = "world!";
        std::string hello_world = turbo::format("{}{}", hello, world);

        turbo::crc32c_t crc_a = turbo::ComputeCrc32c(hello);
        turbo::crc32c_t crc_b = turbo::ComputeCrc32c(world);
        turbo::crc32c_t crc_ab = turbo::ComputeCrc32c(hello_world);

        CHECK_EQ(turbo::RemoveCrc32cPrefix(crc_a, crc_ab, world.size()), crc_b);
    }

    TEST_CASE("CRC32C, RemoveSuffix") {
        std::string hello = "Hello, ";
        std::string world = "world!";
        std::string hello_world = turbo::format("{}{}", hello, world);

        turbo::crc32c_t crc_a = turbo::ComputeCrc32c(hello);
        turbo::crc32c_t crc_b = turbo::ComputeCrc32c(world);
        turbo::crc32c_t crc_ab = turbo::ComputeCrc32c(hello_world);

        CHECK_EQ(turbo::RemoveCrc32cSuffix(crc_ab, crc_b, world.size()), crc_a);
    }
}  // namespace
