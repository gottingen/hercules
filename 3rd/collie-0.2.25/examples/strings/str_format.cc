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
//

#include <collie/strings/format.h>
#include <atomic>
#include <memory>

int main() {

    collie::println("hello");
    collie::println("{} {}","hello", "Jeff");
    collie::println(collie::format_range("[{}]", {1,2,3}, ", "));
    std::string h = "hello";
    collie::format_append(&h, " {}", "Jeff");
    collie::println(h);

    std::string h1 = "hello";
    collie::format_append(&h1, " Jeff");
    collie::println(h1);

    std::vector<int> abc = {1,4,7};
    collie::println("v: {}", abc);

    std::atomic<int> a{1};
    collie::println("a: {}",a);

    std::unique_ptr<int> iptr = std::make_unique<int>(10);
    collie::println("ptr: {}", collie::ptr(iptr.get()));

    return 0;
}