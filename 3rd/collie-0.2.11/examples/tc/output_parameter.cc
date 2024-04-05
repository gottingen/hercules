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

#include <iostream>
#include <sstream>
#include <string>

#include <collie/type_safe/output_parameter.h>

namespace ts = collie::ts;

// task: read strings from stream until EOF
// concatenate all strings
// return whether any strings read

// using a reference as output parameter - BAD:
// * not obvious from the caller that the string will be modified  (in the general case)
// * requires default constructor
// * implicit precondition that the output is empty
bool read_str_naive(std::istream& in, std::string& out)
{
    for (std::string tmp; in >> tmp;)
        out += tmp;
    return !out.empty();
}

// does not have these problems
bool read_str_better(std::istream& in, ts::output_parameter<std::string> out)
{
    std::string result; // no way to access the string directly
    // so need to create new one

    for (std::string tmp; in >> tmp;)
        result += tmp;

    // granted, that's verbose
    auto empty = result.empty();    // we need to query here, because after move it might be empty
    out        = std::move(result); // set output parameter
    return !empty;

    // so use this one:
    // assignment op returns the value that was assigned
    return (out = std::move(result)).empty();
}

int main()
{
    {
        std::istringstream in("hello world");
        std::string        str;
        auto               res = read_str_naive(in, str);
        std::cout << res << ' ' << str << '\n';
    }
    {
        std::istringstream in("hello world");
        std::string        str;
        // use ts::out() to create an output_parameter easily
        auto res = read_str_better(in, ts::out(str));
        std::cout << res << ' ' << str << '\n';
    }
    // what if std::string had no default constructor?
    {
        // use this one:
        ts::deferred_construction<std::string> str;
        // str is not initialized yet,
        // so it does not require a constructor
        // once you give it a value, it will never be empty again

        std::istringstream in("hello world");
        auto               res = read_str_better(in, ts::out(str));
        // if the function assigned a value to the output parameter,
        // it will call the constructor and directly initializes it with the correct value
        std::cout << res << ' ' << str.has_value() << ' ' << str.value() << '\n';
    }
}
