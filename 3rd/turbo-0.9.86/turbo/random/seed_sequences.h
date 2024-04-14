// Copyright 2020 The Turbo Authors.
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
// -----------------------------------------------------------------------------
// File: seed_sequences.h
// -----------------------------------------------------------------------------
//
// This header contains utilities for creating and working with seed sequences
// conforming to [rand.req.seedseq]. In general, direct construction of seed
// sequences is discouraged, but use-cases for construction of identical bit
// generators (using the same seed sequence) may be helpful (e.g. replaying a
// simulation whose state is derived from variates of a bit generator).

#ifndef TURBO_RANDOM_SEED_SEQUENCES_H_
#define TURBO_RANDOM_SEED_SEQUENCES_H_

#include <iterator>
#include <random>

#include "turbo/meta/span.h"
#include "turbo/platform/port.h"
#include "turbo/random/internal/salted_seed_seq.h"
#include "turbo/random/internal/seed_material.h"
#include "turbo/random/seed_gen_exception.h"

namespace turbo {

    // -----------------------------------------------------------------------------
    // turbo::SeedSeq
    // -----------------------------------------------------------------------------
    //
    // `turbo::SeedSeq` constructs a seed sequence according to [rand.req.seedseq]
    // for use within bit generators. `turbo::SeedSeq`, unlike `std::seed_seq`
    // additionally salts the generated seeds with extra implementation-defined
    // entropy. For that reason, you can use `turbo::SeedSeq` in combination with
    // standard library bit generators (e.g. `std::mt19937`) to introduce
    // non-determinism in your seeds.
    //
    // Example:
    //
    //   turbo::SeedSeq my_seed_seq({a, b, c});
    //   std::mt19937 my_bitgen(my_seed_seq);
    //
    using SeedSeq = random_internal::SaltedSeedSeq<std::seed_seq>;

    // -----------------------------------------------------------------------------
    // turbo::CreateSeedSeqFrom(bitgen*)
    // -----------------------------------------------------------------------------
    //
    // Constructs a seed sequence conforming to [rand.req.seedseq] using variates
    // produced by a provided bit generator.
    //
    // You should generally avoid direct construction of seed sequences, but
    // use-cases for reuse of a seed sequence to construct identical bit generators
    // may be helpful (eg. replaying a simulation whose state is derived from bit
    // generator values).
    //
    // If bitgen == nullptr, then behavior is undefined.
    //
    // Example:
    //
    //   turbo::BitGen my_bitgen;
    //   auto seed_seq = turbo::CreateSeedSeqFrom(&my_bitgen);
    //   turbo::BitGen new_engine(seed_seq); // derived from my_bitgen, but not
    //                                      // correlated.
    //
    template<typename URBG>
    SeedSeq CreateSeedSeqFrom(URBG *urbg) {
        SeedSeq::result_type
                seed_material[random_internal::kEntropyBlocksNeeded];

        if (!random_internal::ReadSeedMaterialFromURBG(
                urbg, turbo::MakeSpan(seed_material))) {
            random_internal::ThrowSeedGenException();
        }
        return SeedSeq(std::begin(seed_material), std::end(seed_material));
    }

    // -----------------------------------------------------------------------------
    // turbo::make_seed_seq()
    // -----------------------------------------------------------------------------
    //
    // Constructs an `turbo::SeedSeq` salting the generated values using
    // implementation-defined entropy. The returned sequence can be used to create
    // equivalent bit generators correlated using this sequence.
    //
    // Example:
    //
    //   auto my_seed_seq = turbo::make_seed_seq();
    //   std::mt19937 rng1(my_seed_seq);
    //   std::mt19937 rng2(my_seed_seq);
    //   EXPECT_EQ(rng1(), rng2());
    //
    SeedSeq make_seed_seq();

}  // namespace turbo

#endif  // TURBO_RANDOM_SEED_SEQUENCES_H_
