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
#include "turbo/random/fwd.h"
#include "turbo/concurrent/thread_local.h"

namespace turbo {
    static thread_local BitGen bitgen;

    static thread_local InsecureBitGen fast_bitgen;


    BitGen &get_tls_bit_gen() {
        return bitgen;
    }

    void set_tls_bit_gen(BitGen &&bit_gen) {
        bitgen = std::move(bit_gen);
    }

    InsecureBitGen &get_tls_fast_bit_gen() {
        return fast_bitgen;
    }

    void set_tls_fast_bit_gen(InsecureBitGen &&bit_gen) {
        fast_bitgen = std::move(bit_gen);
    }
}
