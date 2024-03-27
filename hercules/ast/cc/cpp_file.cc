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

#include <hercules/ast/cc/cpp_file.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

using namespace hercules::ccast;

cpp_entity_kind cpp_file::kind() noexcept
{
    return cpp_entity_kind::file_t;
}

cpp_entity_kind cpp_file::do_get_entity_kind() const noexcept
{
    return kind();
}

bool detail::cpp_file_ref_predicate::operator()(const cpp_entity& e)
{
    return e.kind() == cpp_entity_kind::file_t;
}
