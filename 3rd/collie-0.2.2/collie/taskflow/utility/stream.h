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
#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace collie::tf {

    // Procedure: ostreamize
    template<typename T>
    void ostreamize(std::ostream &os, T &&token) {
        os << std::forward<T>(token);
    }

    // Procedure: ostreamize
    template<typename T, typename... Rest>
    void ostreamize(std::ostream &os, T &&token, Rest &&... rest) {
        os << std::forward<T>(token);
        ostreamize(os, std::forward<Rest>(rest)...);
    }

    // Function: stringify
    template<typename... ArgsT>
    std::string stringify(ArgsT &&... args) {
        std::ostringstream oss;
        ostreamize(oss, std::forward<ArgsT>(args)...);
        return oss.str();
    }


}  // end of namespace collie::tf -----------------------------------------------------

