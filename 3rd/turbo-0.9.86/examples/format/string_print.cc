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

#include "turbo/format/print.h"

int main() {

    turbo::Print("{} {} {}", "this is", "example number is ", 42);
    turbo::Println("{} {} {}", "this is", "example number is ", 42);
    turbo::Println(turbo::RedFG, "{} {} {}", "this is", "example number is ", 42);
    turbo::Println(turbo::GreenFG, "{} {} {}", "this is", "example number is ", 42);
    turbo::Print(turbo::RedFG, "{}", "red");
    turbo::Print(turbo::GreenFG, " {}", "green");
    turbo::Println(turbo::YellowFG, " {}", "yellow");
    turbo::Println(turbo::color::medium_violet_red, "{} {} {}", "this is", "example number is ", 42);
    turbo::Println(turbo::color::blue, "{} {} {}", "this is", "example number is ", 42);
}