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

#include <collie/table/table.h>
using namespace collie::table;
using Row_t = Table::Row_t;

#if __cplusplus >= 201703L
#include <string_view>
using std::string_view;
#else
#include <collie/table/string_view_lite.h>
using collie::string_view;
#endif

int main() {
  Table table;

  string_view c0 = "string_view";
  const char *c1 = "const char *";
  std::string c2 = "std::string";

  Table c3;
  c3.add_row({"Table", "", ""});
  c3.add_row({c0, c1, c2});
  c3[0].format().border("").corner("");

  table.add_row({c0, c1, c2, c3});

  std::cout << table << std::endl;
}