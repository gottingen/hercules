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
#include "turbo/times/stop_watcher.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    turbo::StopWatcher watcher("This is a timer");
    turbo::AutoWatcher give_me_a_name(&watcher, true);

    turbo::App app("K3Pi goofit fitter");

    std::string file;
    turbo::Option *opt = app.add_option("-f,--file,file", file, "File name")->required()->group("Important");

    int count{0};
    turbo::Option *copt = app.add_flag("-c,--count", count, "Counter")->required()->group("Important");

    double value{0.0};  // = 3.14;
    app.add_option("-d,--double", value, "Some Value")->group("Other");

    try {
        app.parse(argc, argv);
    } catch(const turbo::ParseError &e) {
        return app.exit(e);
    }

    std::cout << "Working on file: " << file << ", direct count: " << app.count("--file")
              << ", opt count: " << opt->count() << std::endl;
    std::cout << "Working on count: " << count << ", direct count: " << app.count("--count")
              << ", opt count: " << copt->count() << std::endl;
    std::cout << "Some value: " << value << std::endl;

    return 0;
}
