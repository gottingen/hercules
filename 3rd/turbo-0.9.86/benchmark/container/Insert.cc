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

#include "bench.h"
#include "sfc64.h"
//#include "benchmark/benchmark.h"

/**
 * "turbo::flat_hash_map"; "turbo::bytes_hash"; "InsertHugeInt"; "01"; "insert 100M int"; 98841586; 4.96218; 1728.14
"turbo::flat_hash_map"; "turbo::bytes_hash"; "InsertHugeInt"; "02"; "clear 100M int"; 0; 0.0287225; 1728.14
"turbo::flat_hash_map"; "turbo::bytes_hash"; "InsertHugeInt"; "03"; "reinsert 100M int"; 98843646; 5.0678; 1728.32
"turbo::flat_hash_map"; "turbo::bytes_hash"; "InsertHugeInt"; "04"; "remove 100M int"; 0; 5.50145; 1728.32
"turbo::flat_hash_map"; "turbo::bytes_hash"; "InsertHugeInt"; "05"; "destructor empty map"; 0; 0.0290062; 1728.32
 "robin_hood::unordered_flat_map"; "turbo::bytes_hash"; "InsertHugeInt"; "01"; "insert 100M int"; 98841586; 7.80891; 1728.13
"robin_hood::unordered_flat_map"; "turbo::bytes_hash"; "InsertHugeInt"; "02"; "clear 100M int"; 0; 0.00824967; 1728.13
"robin_hood::unordered_flat_map"; "turbo::bytes_hash"; "InsertHugeInt"; "03"; "reinsert 100M int"; 98843646; 4.37516; 1728.13
"robin_hood::unordered_flat_map"; "turbo::bytes_hash"; "InsertHugeInt"; "04"; "remove 100M int"; 0; 4.03407; 1728.13
"robin_hood::unordered_flat_map"; "turbo::bytes_hash"; "InsertHugeInt"; "05"; "destructor empty map"; 0; 0.0301318; 1728.13
 "std::unordered_map"; "std::hash"; "InsertHugeInt"; "01"; "insert 100M int"; 98841586; 30.0006; 3790.8
"std::unordered_map"; "std::hash"; "InsertHugeInt"; "02"; "clear 100M int"; 0; 8.4348; 3790.8
"std::unordered_map"; "std::hash"; "InsertHugeInt"; "03"; "reinsert 100M int"; 98843646; 18.3158; 3790.8
"std::unordered_map"; "std::hash"; "InsertHugeInt"; "04"; "remove 100M int"; 0; 16.9767; 3790.8
"std::unordered_map"; "std::hash"; "InsertHugeInt"; "05"; "destructor empty map"; 0; 0.0648546; 3790.8
 "robin_hood::node_hash_map"; "std::hash"; "InsertHugeInt"; "01"; "insert 100M int"; 98841586; 10.9678; 2304.09
"robin_hood::node_hash_map"; "std::hash"; "InsertHugeInt"; "02"; "clear 100M int"; 0; 0.955022; 2304.09
"robin_hood::node_hash_map"; "std::hash"; "InsertHugeInt"; "03"; "reinsert 100M int"; 98843646; 12.5819; 2304.09
"robin_hood::node_hash_map"; "std::hash"; "InsertHugeInt"; "04"; "remove 100M int"; 0; 6.21677; 2304.09
"robin_hood::node_hash_map"; "std::hash"; "InsertHugeInt"; "05"; "destructor empty map"; 0; 0.0562332; 2304.09
 "turbo::node_hash_map"; "turbo::Hash"; "InsertHugeInt"; "01"; "insert 100M int"; 98841586; 13.6424; 4168.46
"turbo::node_hash_map"; "turbo::Hash"; "InsertHugeInt"; "02"; "clear 100M int"; 0; 3.50438; 4168.46
"turbo::node_hash_map"; "turbo::Hash"; "InsertHugeInt"; "03"; "reinsert 100M int"; 98843646; 27.7797; 4744.56
"turbo::node_hash_map"; "turbo::Hash"; "InsertHugeInt"; "04"; "remove 100M int"; 0; 9.6975; 4744.56
"turbo::node_hash_map"; "turbo::Hash"; "InsertHugeInt"; "05"; "destructor empty map"; 0; 0.0567687; 4744.56

 * @param bench
 */
BENCHMARK(InsertHugeInt) {
    sfc64 rng(213);

    {
        bench.beginMeasure("insert 100M int");
        using M = Map<int, int>;
#ifdef USE_POOL_ALLOCATOR
        Resource<int, int> resource;
        M map{0, M::hasher{}, M::key_equal{}, &resource};
#else
        M map;
#endif
        for (size_t n = 0; n < 100'000'000; ++n) {
            map[static_cast<int>(rng())];
        }
        bench.endMeasure(98841586, map.size());

        bench.beginMeasure("clear 100M int");
        map.clear();
        bench.endMeasure(0, map.size());

        // remember the rng's state so we can remove like we've added
        auto const state = rng.state();
        bench.beginMeasure("reinsert 100M int");
        for (size_t n = 0; n < 100'000'000; ++n) {
            map[static_cast<int>(rng())];
        }
        bench.endMeasure(98843646, map.size());

        rng.state(state);
        bench.beginMeasure("remove 100M int");
        for (size_t n = 0; n < 100'000'000; ++n) {
            map.erase(static_cast<int>(rng()));
        }
        bench.endMeasure(0, map.size());

        bench.beginMeasure("destructor empty map");
    }
    bench.endMeasure(0, 0);
}
