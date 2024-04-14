// Copyright 2023 The titan-search Authors.
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

#include "turbo/flags/reflection.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "turbo/flags/declare.h"
#include "turbo/flags/flag.h"
#include "turbo/flags/internal/commandlineflag.h"
#include "turbo/flags/marshalling.h"
#include "turbo/memory/memory.h"
#include "turbo/strings/str_split.h"
#include "turbo/strings/numbers.h"
#include "turbo/format/format.h"

TURBO_FLAG(int, int_flag, 1, "int_flag help");
TURBO_FLAG(std::string, string_flag, "dflt", "string_flag help");
TURBO_RETIRED_FLAG(bool, bool_retired_flag, false, "bool_retired_flag help");

namespace {

class ReflectionTest : public testing::Test {
 protected:
  void SetUp() override { flag_saver_ = turbo::make_unique<turbo::FlagSaver>(); }
  void TearDown() override { flag_saver_.reset(); }

 private:
  std::unique_ptr<turbo::FlagSaver> flag_saver_;
};

// --------------------------------------------------------------------

TEST_F(ReflectionTest, TestFindCommandLineFlag) {
  auto* handle = turbo::find_command_line_flag("some_flag");
  EXPECT_EQ(handle, nullptr);

  handle = turbo::find_command_line_flag("int_flag");
  EXPECT_NE(handle, nullptr);

  handle = turbo::find_command_line_flag("string_flag");
  EXPECT_NE(handle, nullptr);

  handle = turbo::find_command_line_flag("bool_retired_flag");
  EXPECT_NE(handle, nullptr);
}

// --------------------------------------------------------------------

TEST_F(ReflectionTest, TestGetAllFlags) {
  auto all_flags = turbo::get_all_flags();
  EXPECT_NE(all_flags.find("int_flag"), all_flags.end());
  EXPECT_EQ(all_flags.find("bool_retired_flag"), all_flags.end());
  EXPECT_EQ(all_flags.find("some_undefined_flag"), all_flags.end());

  std::vector<std::string_view > flag_names_first_attempt;
  auto all_flags_1 = turbo::get_all_flags();
  for (auto f : all_flags_1) {
    flag_names_first_attempt.push_back(f.first);
  }

  std::vector<std::string_view > flag_names_second_attempt;
  auto all_flags_2 = turbo::get_all_flags();
  for (auto f : all_flags_2) {
    flag_names_second_attempt.push_back(f.first);
  }

  EXPECT_THAT(flag_names_first_attempt,
              ::testing::UnorderedElementsAreArray(flag_names_second_attempt));
}

// --------------------------------------------------------------------

struct CustomUDT {
  CustomUDT() : a(1), b(1) {}
  CustomUDT(int a_, int b_) : a(a_), b(b_) {}

  friend bool operator==(const CustomUDT& f1, const CustomUDT& f2) {
    return f1.a == f2.a && f1.b == f2.b;
  }

  int a;
  int b;
};
bool turbo_parse_flag(std::string_view in, CustomUDT* f, std::string*) {
  std::vector<std::string_view > parts =
      turbo::str_split(in, ':', turbo::skip_whitespace());

  if (parts.size() != 2) return false;

  if (!turbo::simple_atoi(parts[0], &f->a)) return false;

  if (!turbo::simple_atoi(parts[1], &f->b)) return false;

  return true;
}
std::string turbo_unparse_flag(const CustomUDT& f) {
  return turbo::format("{}{}{}", f.a, ":", f.b);
}

}  // namespace

// --------------------------------------------------------------------

TURBO_FLAG(bool, test_flag_01, true, "");
TURBO_FLAG(int, test_flag_02, 1234, "");
TURBO_FLAG(int16_t, test_flag_03, -34, "");
TURBO_FLAG(uint16_t, test_flag_04, 189, "");
TURBO_FLAG(int32_t, test_flag_05, 10765, "");
TURBO_FLAG(uint32_t, test_flag_06, 40000, "");
TURBO_FLAG(int64_t, test_flag_07, -1234567, "");
TURBO_FLAG(uint64_t, test_flag_08, 9876543, "");
TURBO_FLAG(double, test_flag_09, -9.876e-50, "");
TURBO_FLAG(float, test_flag_10, 1.234e12f, "");
TURBO_FLAG(std::string, test_flag_11, "", "");
TURBO_FLAG(turbo::Duration, test_flag_12, turbo::Duration::minutes(10), "");
static int counter = 0;
TURBO_FLAG(int, test_flag_13, 200, "").on_update([]() { counter++; });
TURBO_FLAG(CustomUDT, test_flag_14, {}, "");

namespace {

TEST_F(ReflectionTest, TestFlagSaverInScope) {
  {
    turbo::FlagSaver s;
    counter = 0;
    turbo::set_flag(&FLAGS_test_flag_01, false);
    turbo::set_flag(&FLAGS_test_flag_02, -1021);
    turbo::set_flag(&FLAGS_test_flag_03, 6009);
    turbo::set_flag(&FLAGS_test_flag_04, 44);
    turbo::set_flag(&FLAGS_test_flag_05, +800);
    turbo::set_flag(&FLAGS_test_flag_06, -40978756);
    turbo::set_flag(&FLAGS_test_flag_07, 23405);
    turbo::set_flag(&FLAGS_test_flag_08, 975310);
    turbo::set_flag(&FLAGS_test_flag_09, 1.00001);
    turbo::set_flag(&FLAGS_test_flag_10, -3.54f);
    turbo::set_flag(&FLAGS_test_flag_11, "asdf");
    turbo::set_flag(&FLAGS_test_flag_12, turbo::Duration::hours(20));
    turbo::set_flag(&FLAGS_test_flag_13, 4);
    turbo::set_flag(&FLAGS_test_flag_14, CustomUDT{-1, -2});
  }

  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_01), true);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_02), 1234);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_03), -34);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_04), 189);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_05), 10765);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_06), 40000);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_07), -1234567);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), 9876543);
  EXPECT_NEAR(turbo::get_flag(FLAGS_test_flag_09), -9.876e-50, 1e-55);
  EXPECT_NEAR(turbo::get_flag(FLAGS_test_flag_10), 1.234e12f, 1e5f);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_11), "");
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_12), turbo::Duration::minutes(10));
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_13), 200);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_14), CustomUDT{});
  EXPECT_EQ(counter, 2);
}

// --------------------------------------------------------------------

TEST_F(ReflectionTest, TestFlagSaverVsUpdateViaReflection) {
  {
    turbo::FlagSaver s;
    counter = 0;
    std::string error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_01")->parse_from("false", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_02")->parse_from("-4536", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_03")->parse_from("111", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_04")->parse_from("909", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_05")->parse_from("-2004", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_06")->parse_from("1000023", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_07")->parse_from("69305", &error))
        << error;
    EXPECT_TRUE(turbo::find_command_line_flag("test_flag_08")
                    ->parse_from("1000000001", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_09")->parse_from("2.09021", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_10")->parse_from("-33.1", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_11")->parse_from("ADD_FOO", &error))
        << error;
    EXPECT_TRUE(turbo::find_command_line_flag("test_flag_12")
                    ->parse_from("3h11m16s", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_13")->parse_from("0", &error))
        << error;
    EXPECT_TRUE(
        turbo::find_command_line_flag("test_flag_14")->parse_from("10:1", &error))
        << error;
  }

  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_01), true);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_02), 1234);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_03), -34);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_04), 189);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_05), 10765);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_06), 40000);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_07), -1234567);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), 9876543);
  EXPECT_NEAR(turbo::get_flag(FLAGS_test_flag_09), -9.876e-50, 1e-55);
  EXPECT_NEAR(turbo::get_flag(FLAGS_test_flag_10), 1.234e12f, 1e5f);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_11), "");
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_12), turbo::Duration::minutes(10));
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_13), 200);
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_14), CustomUDT{});
  EXPECT_EQ(counter, 2);
}

// --------------------------------------------------------------------

TEST_F(ReflectionTest, TestMultipleFlagSaversInEnclosedScopes) {
  {
    turbo::FlagSaver s;
    turbo::set_flag(&FLAGS_test_flag_08, 10);
    EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), 10);
    {
      turbo::FlagSaver s;
      turbo::set_flag(&FLAGS_test_flag_08, 20);
      EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), 20);
      {
        turbo::FlagSaver s;
        turbo::set_flag(&FLAGS_test_flag_08, -200);
        EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), -200);
      }
      EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), 20);
    }
    EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), 10);
  }
  EXPECT_EQ(turbo::get_flag(FLAGS_test_flag_08), 9876543);
}

}  // namespace
