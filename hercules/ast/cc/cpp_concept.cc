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


#include <hercules/ast/cc/cpp_concept.h>
#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    cpp_entity_kind hercules::ccast::cpp_concept::kind() noexcept {
        return cpp_entity_kind::concept_t;
    }

    cpp_entity_kind cpp_concept::do_get_entity_kind() const noexcept {
        return cpp_entity_kind::concept_t;
    }

    std::unique_ptr<cpp_concept> cpp_concept::builder::finish(const cpp_entity_index &idx,
                                                              cpp_entity_id id) {
        idx.register_definition(id, collie::ts::ref(*concept_));
        return std::move(concept_);
    }
}
