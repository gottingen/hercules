// Copyright 2024 The Elastic AI Search Authors.
//
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

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <collie/table/format.h>
#include <collie/table/utf8.h>
#include <optional>
#include <vector>

namespace collie::table {

    class Cell {
    public:
        explicit Cell(std::shared_ptr<class Row> parent) : parent_(parent) {}

        void set_text(const std::string &text) { data_ = text; }

        const std::string &get_text() { return data_; }

        size_t size() {
            return get_sequence_length(data_, locale(), is_multi_byte_character_support_enabled());
        }

        std::string locale() { return *format().locale_; }

        Format &format();

        bool is_multi_byte_character_support_enabled();

    private:
        std::string data_;
        std::weak_ptr<class Row> parent_;
        std::optional<Format> format_;
    };

} // namespace collie::table
