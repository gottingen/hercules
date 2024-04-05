// Copyright 2024 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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


#include <hercules/ast/cc/cpp_language_linkage.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    cpp_entity_kind cpp_language_linkage::kind() noexcept {
        return cpp_entity_kind::language_linkage_t;
    }

    bool cpp_language_linkage::is_block() const noexcept {
        if (begin() == end()) {
            // An empty container must be a "block" of the form: extern "C" {}
            return true;
        }
        return std::next(begin()) != end(); // more than one entity, so block
    }

    cpp_entity_kind cpp_language_linkage::do_get_entity_kind() const noexcept {
        return kind();
    }
}