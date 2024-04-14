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
#include <iostream>
#include <string>

int main(int argc, char **argv) {

    turbo::App app("Validator checker");

    std::string file;
    app.add_option("-f,--file,file", file, "File name")->check(turbo::ExistingFile);

    int count{0};
    app.add_option("-v,--value", count, "Value in range")->check(turbo::Range(3, 6));
    TURBO_FLAGS_PARSE(app, argc, argv);

    std::cout << "Try printing help or failing the validator" << std::endl;

    return 0;
}
