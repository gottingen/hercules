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
// Created by jeff on 24-3-27.
//

#ifndef COLLIE_BASE_IDENTITY_H_
#define COLLIE_BASE_IDENTITY_H_

namespace collie {

    // Similar to `std::identity` from C++20.
    template <class Ty> struct identity {
        using is_transparent = void;
        using argument_type = Ty;

        Ty &operator()(Ty &self) const {
            return self;
        }
        const Ty &operator()(const Ty &self) const {
            return self;
        }
    };

}  // namespace collie

#endif  // COLLIE_BASE_IDENTITY_H_
