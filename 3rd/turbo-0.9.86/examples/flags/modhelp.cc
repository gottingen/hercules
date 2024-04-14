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
    turbo::App test{R"raw(Modify the help print so that argument values are accessible.
Note that this will not shortcut `->required` and other similar options.)raw"};

    // Remove help flag because it shortcuts all processing
    test.set_help_flag();

    // Add custom flag that activates help
    auto *help = test.add_flag("-h,--help", "Request help");

    std::string some_option;
    test.add_option("-a", some_option, "Some description");

    try {
        test.parse(argc, argv);
        if(*help)
            throw turbo::CallForHelp();
    } catch(const turbo::Error &e) {
        std::cout << "Option -a string in help: " << some_option << std::endl;
        return test.exit(e);
    }

    std::cout << "Option -a string: " << some_option << std::endl;
    return 0;
}
