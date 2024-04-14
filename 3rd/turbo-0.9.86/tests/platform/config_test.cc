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

#include "turbo/platform/port.h"

#include <cstdint>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/concurrent/internal/thread_pool.h"

namespace {

TEST_CASE("ConfigTest, Endianness") {
  union {
    uint32_t value;
    uint8_t data[sizeof(uint32_t)];
  } number;
  number.data[0] = 0x00;
  number.data[1] = 0x01;
  number.data[2] = 0x02;
  number.data[3] = 0x03;
#if TURBO_IS_LITTLE_ENDIAN && TURBO_IS_BIG_ENDIAN
#error Both TURBO_IS_LITTLE_ENDIAN and TURBO_IS_BIG_ENDIAN are defined
#elif TURBO_IS_LITTLE_ENDIAN
  CHECK_EQ(UINT32_C(0x03020100), number.value);
#elif TURBO_IS_BIG_ENDIAN
  EXPECT_EQ(UINT32_C(0x00010203), number.value);
#else
#error Unknown endianness
#endif
}
/*
#if defined(TURBO_HAVE_THREAD_LOCAL)
TEST(ConfigTest, ThreadLocal) {
  static thread_local int mine_mine_mine = 16;
  EXPECT_EQ(16, mine_mine_mine);
  {
    turbo::concurrent_internal::ThreadPool pool(1);
    pool.Schedule([&] {
      EXPECT_EQ(16, mine_mine_mine);
      mine_mine_mine = 32;
      EXPECT_EQ(32, mine_mine_mine);
    });
  }
  EXPECT_EQ(16, mine_mine_mine);
}
#endif
*/
}  // namespace
