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


#ifndef TURBO_HASH_INTERNAL_UTILITY_H_
#define TURBO_HASH_INTERNAL_UTILITY_H_

namespace turbo::hash_internal {

    struct AggregateBarrier {
    };

    // HashImpl

    // Add a private base class to make sure this type is not an aggregate.
    // Aggregates can be aggregate initialized even if the default constructor is
    // deleted.
    struct PoisonedHash : private AggregateBarrier {
        PoisonedHash() = delete;

        PoisonedHash(const PoisonedHash &) = delete;

        PoisonedHash &operator=(const PoisonedHash &) = delete;
    };
}  // namespace turbo::hash_internal
#endif  // TURBO_HASH_INTERNAL_UTILITY_H_
