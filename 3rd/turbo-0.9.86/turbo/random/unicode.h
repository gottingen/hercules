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

#ifndef TURBO_RANDOM_UNICODE_H_
#define TURBO_RANDOM_UNICODE_H_

#include <cstdint>
#include <vector>
#include "turbo/random/engine.h"
#include "turbo/random/discrete_distribution.h"

namespace turbo {

    /**
     * @ingroup turbo_random_unicode
     * @brief Generates random UTF-8 sequences.
     *        The generator is based on the following probabilities:
     *        - 1 byte: 0x00 - 0x7f (127)
     *        - 2 bytes: 0xc2 - 0xdf (192 - 223)
     *        - 3 bytes: 0xe0 - 0xef (224 - 239)
     *        - 4 bytes: 0xf0 - 0xf4 (240 - 244)
     *        The probabilities are configurable.
     *        Example:
     *        @code{.cpp}
     *        Utf8Generator gen(50, 30, 15, 5);
     *        std::vector<uint8_t> bytes = gen.generate(100);
     *        @endcode
     *        The example above will generate 100 random bytes with the following
     *        probabilities:
     *        - 1 byte: 50%
     *        - 2 bytes: 30%
     *        - 3 bytes: 15%
     *        - 4 bytes: 5%
     * @note The generator is not thread-safe.
     */
    class Utf8Generator {
    public:
        Utf8Generator(int prob_1byte, int prob_2bytes,
                      int prob_3bytes, int prob_4bytes);

        std::vector<uint8_t> generate(size_t output_bytes);

        std::pair<std::vector<uint8_t>, size_t> generate_counted(size_t output_bytes);

    private:

        BitGen _urbg;
        discrete_distribution<int> _bytes_count;
        uniform_int_distribution<int> val_7bit{0x00, 0x7f}; // 0b0xxxxxxx
        uniform_int_distribution<int> val_6bit{0x00, 0x3f}; // 0b10xxxxxx
        uniform_int_distribution<int> val_5bit{0x00, 0x1f}; // 0b110xxxxx
        uniform_int_distribution<int> val_4bit{0x00, 0x0f}; // 0b1110xxxx
        uniform_int_distribution<int> val_3bit{0x00, 0x07}; // 0b11110xxx
    };

    /**
     * @ingroup turbo_random_unicode
     * @brief Generates random UTF-16 sequences.
     *        The generator is based on the following probabilities:
     *        - 1 word: 0x0000 - 0xd7ff (0 - 55295)
     *        - 2 words: 0xe000 - 0xffff (57344 - 65535)
     *        - 2 words: 0x10000 - 0x10ffff (65536 - 1114111)
     *        The probabilities are configurable.
     *        Example:
     *        @code{.cpp}
     *        Utf16Generator gen(50, 50);
     *        std::vector<uint16_t> words = gen.generate(100);
     *        @endcode
     *        The example above will generate 100 random words with the following
     *        probabilities:
     *        - 1 word: 50%
     *        - 2 words: 50%
     * @note The generator is not thread-safe.
     */
    class Utf16Generator {
    public:
        Utf16Generator(int single_word_prob, int two_words_probability)
                : _utf16_length({double(single_word_prob),
                                 double(single_word_prob),
                                 double(2 * two_words_probability)}) {}

        std::vector<char16_t> generate(size_t size);

        std::pair<std::vector<char16_t>, size_t> generate_counted(size_t size);

    private:
        BitGen _urbg;
        discrete_distribution<> _utf16_length;
        uniform_int_distribution<uint32_t> _single_word0{0x00000000, 0x0000d7ff};
        uniform_int_distribution<uint32_t> _single_word1{0x0000e000, 0x0000ffff};
        uniform_int_distribution<uint32_t> _two_words{0x00010000, 0x0010ffff};

        uint32_t generate();
    };

    /**
     * @ingroup turbo_random_unicode
     * @brief Generates random UTF-32 sequences.
     *        by the following formula:
     *        Example:
     *        @code{.cpp}
     *        Utf32Generator gen;
     *        std::vector<uint32_t> words = gen.generate(100);
     *        @endcode
     */
    class Utf32Generator {
    public:
        static constexpr int32_t number_code_points = 0x0010ffff - (0xdfff - 0xd800);
        static constexpr int32_t length_first_range = 0x0000d7ff;
        static constexpr int32_t length_second_range = 0x0010ffff - 0x0000e000;
    public:
        Utf32Generator()
                : _range({double(length_first_range) / double(number_code_points),
                          double(length_second_range) / double(number_code_points)}) {}


        std::vector<char32_t> generate(size_t size);

    private:
        BitGen _urbg;
        discrete_distribution<> _range;
        uniform_int_distribution<uint32_t> _first_range{0x00000000, 0x0000d7ff};
        uniform_int_distribution<uint32_t> _second_range{0x0000e000, 0x0010ffff};

        uint32_t generate();
    };

}  // namespace turbo
#endif  // TURBO_RANDOM_UNICODE_H_
