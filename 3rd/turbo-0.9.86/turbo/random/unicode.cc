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


#include "turbo/random/unicode.h"
#include "turbo/base/endian.h"
#include <turbo/unicode/utf.h>

namespace turbo {

    Utf8Generator::Utf8Generator(int prob_1byte, int prob_2bytes,
                  int prob_3bytes, int prob_4bytes) :_bytes_count({double(prob_1byte), double(prob_2bytes),
                                                                   double(prob_3bytes), double(prob_4bytes)}){
    }

    std::vector<uint8_t> Utf8Generator::generate(size_t output_bytes) {
        return generate_counted(output_bytes).first;
    }

    std::pair<std::vector<uint8_t>, size_t> Utf8Generator::generate_counted(size_t output_bytes) {
        std::vector<uint8_t> result;
        result.reserve(output_bytes);
        uint8_t candidate, head;
        size_t count{0};
        while (result.size() < output_bytes) {
            count++;
            switch (_bytes_count(_urbg)) {
                case 0: // 1 byte
                    candidate = uint8_t(val_7bit(_urbg));
                    while (candidate == 0) { // though strictly speaking, a stream of nulls is
                        // UTF8, it tends to break some code
                        candidate = uint8_t(val_7bit(_urbg));
                    }
                    result.push_back(candidate);
                    break;
                case 1: // 2 bytes
                    candidate = 0xc0 | uint8_t(val_5bit(_urbg));
                    while (candidate < 0xC2) {
                        candidate = 0xc0 | uint8_t(val_5bit(_urbg));
                    }
                    result.push_back(candidate);
                    result.push_back(0x80 | uint8_t(val_6bit(_urbg)));
                    break;
                case 2: // 3 bytes
                    head = 0xe0 | uint8_t(val_4bit(_urbg));
                    result.push_back(head);
                    candidate = 0x80 | uint8_t(val_6bit(_urbg));
                    if (head == 0xE0) {
                        while (candidate < 0xA0) {
                            candidate = 0x80 | uint8_t(val_6bit(_urbg));
                        }
                    } else if (head == 0xED) {
                        while (candidate > 0x9F) {
                            candidate = 0x80 | uint8_t(val_6bit(_urbg));
                        }
                    }
                    result.push_back(candidate);
                    result.push_back(0x80 | uint8_t(val_6bit(_urbg)));
                    break;
                case 3: // 4 bytes
                    head = 0xf0 | uint8_t(val_3bit(_urbg));
                    while (head > 0xF4) {
                        head = 0xf0 | uint8_t(val_3bit(_urbg));
                    }
                    result.push_back(head);
                    candidate = 0x80 | uint8_t(val_6bit(_urbg));
                    if (head == 0xF0) {
                        while (candidate < 0x90) {
                            candidate = 0x80 | uint8_t(val_6bit(_urbg));
                        }
                    } else if (head == 0xF4) {
                        while (candidate > 0x8F) {
                            candidate = 0x80 | uint8_t(val_6bit(_urbg));
                        }
                    }
                    result.push_back(candidate);
                    result.push_back(0x80 | uint8_t(val_6bit(_urbg)));
                    result.push_back(0x80 | uint8_t(val_6bit(_urbg)));
                    break;
            }
        }
        result.push_back(0); // EOS for scalar code

        return make_pair(result,count);
    }

    std::vector<char16_t> Utf16Generator::generate(size_t size) {
        return generate_counted(size).first;
    }

    std::pair<std::vector<char16_t>,size_t> Utf16Generator::generate_counted(size_t size) {
        std::vector<char16_t> result;
        result.reserve(size);

        size_t count{0};
        char16_t output_buf[2];
            count++;
            const uint32_t value = generate();
            auto r = turbo::convert_utf32_to_utf16(reinterpret_cast<const char32_t *>(&value), 1, output_buf);
            switch (r) {
                case 0:
                    throw std::runtime_error("Random UTF-16 generator is broken");
                case 1:
                    result.push_back(output_buf[0]);
                    break;
                case 2:
                    result.push_back(output_buf[0]);
                    result.push_back(output_buf[1]);
                    break;
            }
        if constexpr (!kIsLittleEndian) {
            turbo::change_endianness_utf16(result.data(),result.size(), result.data());
        }
        return make_pair(result,count);
    }


    uint32_t Utf16Generator::generate() {
        switch (_utf16_length(_urbg)) {
            case 0:
                return _single_word0(_urbg);
            case 1:
                return _single_word1(_urbg);
            case 2:
                return _two_words(_urbg);
            default:
                abort();
        }
    }


    std::vector<char32_t> Utf32Generator::generate(size_t size) {

        std::vector<char32_t> result;
        result.reserve(size);

        size_t count{0};
        for(;count < size; count++) {
            const uint32_t value = generate();
            result.push_back(value);
        }

        return result;
    }

    uint32_t Utf32Generator::generate() {
        switch (_range(_urbg)) {
            case 0:
                return _first_range(_urbg);
            case 1:
                return _second_range(_urbg);
            default:
                abort();
        }
    }
}  // namespace turbo