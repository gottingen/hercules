// Copyright 2022 The Turbo Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo/crypto/internal/crc_cord_state.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "turbo/crypto/crc32c.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

namespace {

    TEST_CASE("CrcCordState, Default") {
        turbo::crc_internal::CrcCordState state;
        CHECK(state.IsNormalized());
        CHECK_EQ(state.Checksum(), turbo::crc32c_t{0});
        state.Normalize();
        CHECK_EQ(state.Checksum(), turbo::crc32c_t{0});
    }

    TEST_CASE("CrcCordState, Normalize") {
        turbo::crc_internal::CrcCordState state;
        auto *rep = state.mutable_rep();
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(1000, turbo::crc32c_t{1000}));
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(2000, turbo::crc32c_t{2000}));
        rep->removed_prefix =
                turbo::crc_internal::CrcCordState::PrefixCrc(500, turbo::crc32c_t{500});

        // The removed_prefix means state is not normalized.
        CHECK_FALSE(state.IsNormalized());

        turbo::crc32c_t crc = state.Checksum();
        state.Normalize();
        CHECK(state.IsNormalized());

        // The checksum should not change as a result of calling Normalize().
        CHECK_EQ(state.Checksum(), crc);
        CHECK_EQ(rep->removed_prefix.length, 0);
    }

    TEST_CASE("CrcCordState, Copy") {
        turbo::crc_internal::CrcCordState state;
        auto *rep = state.mutable_rep();
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(1000, turbo::crc32c_t{1000}));

        turbo::crc_internal::CrcCordState copy = state;

        CHECK_EQ(state.Checksum(), turbo::crc32c_t{1000});
        CHECK_EQ(copy.Checksum(), turbo::crc32c_t{1000});
    }

    TEST_CASE("CrcCordState, UnsharedSelfCopy") {
        turbo::crc_internal::CrcCordState state;
        auto *rep = state.mutable_rep();
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(1000, turbo::crc32c_t{1000}));

        const turbo::crc_internal::CrcCordState &ref = state;
        state = ref;

        CHECK_EQ(state.Checksum(), turbo::crc32c_t{1000});
    }

    TEST_CASE("CrcCordState, Move") {
        turbo::crc_internal::CrcCordState state;
        auto *rep = state.mutable_rep();
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(1000, turbo::crc32c_t{1000}));

        turbo::crc_internal::CrcCordState moved = std::move(state);
        CHECK_EQ(moved.Checksum(), turbo::crc32c_t{1000});
    }

    TEST_CASE("CrcCordState, UnsharedSelfMove") {
        turbo::crc_internal::CrcCordState state;
        auto *rep = state.mutable_rep();
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(1000, turbo::crc32c_t{1000}));

        turbo::crc_internal::CrcCordState &ref = state;
        state = std::move(ref);

        CHECK_EQ(state.Checksum(), turbo::crc32c_t{1000});
    }

    TEST_CASE("CrcCordState, PoisonDefault") {
        turbo::crc_internal::CrcCordState state;
        state.Poison();
        CHECK_NE(state.Checksum(), turbo::crc32c_t{0});
    }

    TEST_CASE("CrcCordState, PoisonData") {
        turbo::crc_internal::CrcCordState state;
        auto *rep = state.mutable_rep();
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(1000, turbo::crc32c_t{1000}));
        rep->prefix_crc.push_back(
                turbo::crc_internal::CrcCordState::PrefixCrc(2000, turbo::crc32c_t{2000}));
        rep->removed_prefix =
                turbo::crc_internal::CrcCordState::PrefixCrc(500, turbo::crc32c_t{500});

        turbo::crc32c_t crc = state.Checksum();
        state.Poison();
        CHECK_NE(state.Checksum(), crc);
    }

}  // namespace
