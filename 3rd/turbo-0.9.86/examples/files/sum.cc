// Copyright 2023 The Turbo Authors.
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
#include "turbo/flags/flags.h"
#include "turbo/files/filesystem.h"
#include "turbo/format/print.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {

    turbo::App app("sum");
    // add version output
    app.set_version_flag("--version", std::string("0.1.0"));
    std::string file;
    std::string sum_type;
    app.add_option("-f,--file,file", file, "File name")->required();
    app.add_option("-s,--sum", sum_type, "sum type [md5|sha1]")->default_val("md5");

    TURBO_FLAGS_PARSE(app, argc, argv);

    turbo::ResultStatus<std::string> rs;
    int64_t file_size;
    if(sum_type  == "md5") {
        rs = turbo::md5_sum_file(file, &file_size);
    } else if (sum_type  == "sha1"){
        rs = turbo::sha1_sum_file(file, &file_size);
    }
    if(!rs.ok()) {
        turbo::Println("{} sum file error: {}", sum_type, rs.status().message());
    } else {
        turbo::Println("{} sum file bytes: {} result: {}", sum_type, file_size, rs.value());
    }
    return 0;
}
