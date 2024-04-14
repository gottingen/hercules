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

#include <collie/strings/cat.h>
#include <iostream>

void str_cat_example() {
    std::cout << "number: "<<collie::str_cat(1, 2, 3, 4, 5) << std::endl;
    std::cout << "string: "<<collie::str_cat("a", "b", "c", "d", "e") << std::endl;
    std::cout << "vector string: "<<collie::str_cat(std::vector<std::string>{"a", "b", "c", "d", "e"}) << std::endl;
    std::cout << "vector string_view: "<<collie::str_cat(std::vector<std::string_view>{"a", "b", "c", "d", "e"}) << std::endl;
}

void str_cat_append_example() {
    std::string result("number: ");
    collie::str_cat_append(result, 1);
    collie::str_cat_append(result, 2);
    collie::str_cat_append(result, 3);
    collie::str_cat_append(result, 4);
    collie::str_cat_append(result, 5);
    std::cout << result << std::endl;
}
int main() {
    str_cat_example();
    str_cat_append_example();
    return 0;
}