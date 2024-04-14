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
#include "turbo/flags/commandlineflag.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "turbo/flags/flag.h"
#include "turbo/flags/internal/commandlineflag.h"
#include "turbo/flags/internal/private_handle_accessor.h"
#include "turbo/flags/reflection.h"
#include "turbo/flags/usage_config.h"
#include "turbo/memory/memory.h"
#include "turbo/strings/match.h"
#include "turbo/strings/string_view.h"

TURBO_FLAG(int, int_flag, 201, "int_flag help");
TURBO_FLAG(std::string, string_flag, "dflt",
          turbo::format("string_flag {}", "help"));
TURBO_RETIRED_FLAG(bool, bool_retired_flag, false, "bool_retired_flag help");

// These are only used to test default values.
TURBO_FLAG(int, int_flag2, 201, "");
TURBO_FLAG(std::string, string_flag2, "dflt", "");

namespace {

namespace flags = turbo::flags_internal;

class CommandLineFlagTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Install a function to normalize filenames before this test is run.
    turbo::FlagsUsageConfig default_config;
    default_config.normalize_filename = &CommandLineFlagTest::NormalizeFileName;
    turbo::set_flags_usage_config(default_config);
  }

  void SetUp() override { flag_saver_ = turbo::make_unique<turbo::FlagSaver>(); }
  void TearDown() override { flag_saver_.reset(); }

 private:
  static std::string NormalizeFileName(std::string_view fname) {
#ifdef _WIN32
    std::string normalized(fname);
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    fname = normalized;
#endif
    return std::string(fname);
  }

  std::unique_ptr<turbo::FlagSaver> flag_saver_;
};

TEST_F(CommandLineFlagTest, TestAttributesAccessMethods) {
  auto* flag_01 = turbo::find_command_line_flag("int_flag");

  ASSERT_TRUE(flag_01);
  EXPECT_EQ(flag_01->name(), "int_flag");
  EXPECT_EQ(flag_01->help(), "int_flag help");
  EXPECT_TRUE(!flag_01->is_retired());
  EXPECT_TRUE(flag_01->is_of_type<int>());
  EXPECT_TRUE(!flag_01->is_of_type<bool>());
  EXPECT_TRUE(!flag_01->is_of_type<std::string>());
  EXPECT_TRUE(turbo::ends_with(flag_01->filename(),
                             "tests/flags/commandlineflag_test.cc"))
      << flag_01->filename();

  auto* flag_02 = turbo::find_command_line_flag("string_flag");

  ASSERT_TRUE(flag_02);
  EXPECT_EQ(flag_02->name(), "string_flag");
  EXPECT_EQ(flag_02->help(), "string_flag help");
  EXPECT_TRUE(!flag_02->is_retired());
  EXPECT_TRUE(flag_02->is_of_type<std::string>());
  EXPECT_TRUE(!flag_02->is_of_type<bool>());
  EXPECT_TRUE(!flag_02->is_of_type<int>());
  EXPECT_TRUE(turbo::ends_with(flag_02->filename(),
                             "tests/flags/commandlineflag_test.cc"))
      << flag_02->filename();
}

// --------------------------------------------------------------------

TEST_F(CommandLineFlagTest, TestValueAccessMethods) {
  turbo::set_flag(&FLAGS_int_flag2, 301);
  auto* flag_01 = turbo::find_command_line_flag("int_flag2");

  ASSERT_TRUE(flag_01);
  EXPECT_EQ(flag_01->current_value(), "301");
  EXPECT_EQ(flag_01->default_value(), "201");

  turbo::set_flag(&FLAGS_string_flag2, "new_str_value");
  auto* flag_02 = turbo::find_command_line_flag("string_flag2");

  ASSERT_TRUE(flag_02);
  EXPECT_EQ(flag_02->current_value(), "new_str_value");
  EXPECT_EQ(flag_02->default_value(), "dflt");
}

// --------------------------------------------------------------------

TEST_F(CommandLineFlagTest, TestParseFromCurrentValue) {
  std::string err;

  auto* flag_01 = turbo::find_command_line_flag("int_flag");
  EXPECT_FALSE(
      flags::PrivateHandleAccessor::is_specified_on_command_line(*flag_01));

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "11", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange, err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), 11);
  EXPECT_FALSE(
      flags::PrivateHandleAccessor::is_specified_on_command_line(*flag_01));

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "-123", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), -123);
  EXPECT_FALSE(
      flags::PrivateHandleAccessor::is_specified_on_command_line(*flag_01));

  EXPECT_TRUE(!flags::PrivateHandleAccessor::parse_from(
      *flag_01, "xyz", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), -123);
  EXPECT_EQ(err, "Illegal value 'xyz' specified for flag 'int_flag'");
  EXPECT_FALSE(
      flags::PrivateHandleAccessor::is_specified_on_command_line(*flag_01));

  EXPECT_TRUE(!flags::PrivateHandleAccessor::parse_from(
      *flag_01, "A1", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange, err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), -123);
  EXPECT_EQ(err, "Illegal value 'A1' specified for flag 'int_flag'");
  EXPECT_FALSE(
      flags::PrivateHandleAccessor::is_specified_on_command_line(*flag_01));

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "0x10", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), 16);
  EXPECT_FALSE(
      flags::PrivateHandleAccessor::is_specified_on_command_line(*flag_01));

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "011", flags::SET_FLAGS_VALUE, flags::kCommandLine, err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), 11);
  EXPECT_TRUE(flags::PrivateHandleAccessor::is_specified_on_command_line(*flag_01));

  EXPECT_TRUE(!flags::PrivateHandleAccessor::parse_from(
      *flag_01, "", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange, err));
  EXPECT_EQ(err, "Illegal value '' specified for flag 'int_flag'");

  auto* flag_02 = turbo::find_command_line_flag("string_flag");
  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_02, "xyz", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(turbo::get_flag(FLAGS_string_flag), "xyz");

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_02, "", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange, err));
  EXPECT_EQ(turbo::get_flag(FLAGS_string_flag), "");
}

// --------------------------------------------------------------------

TEST_F(CommandLineFlagTest, TestParseFromDefaultValue) {
  std::string err;

  auto* flag_01 = turbo::find_command_line_flag("int_flag");

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "111", flags::SET_FLAGS_DEFAULT, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(flag_01->default_value(), "111");

  auto* flag_02 = turbo::find_command_line_flag("string_flag");

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_02, "abc", flags::SET_FLAGS_DEFAULT, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(flag_02->default_value(), "abc");
}

// --------------------------------------------------------------------

TEST_F(CommandLineFlagTest, TestParseFromIfDefault) {
  std::string err;

  auto* flag_01 = turbo::find_command_line_flag("int_flag");

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "22", flags::SET_FLAG_IF_DEFAULT, flags::kProgrammaticChange,
      err))
      << err;
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), 22);

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "33", flags::SET_FLAG_IF_DEFAULT, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), 22);
  // EXPECT_EQ(err, "ERROR: int_flag is already set to 22");

  // Reset back to default value
  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "201", flags::SET_FLAGS_VALUE, flags::kProgrammaticChange,
      err));

  EXPECT_TRUE(flags::PrivateHandleAccessor::parse_from(
      *flag_01, "33", flags::SET_FLAG_IF_DEFAULT, flags::kProgrammaticChange,
      err));
  EXPECT_EQ(turbo::get_flag(FLAGS_int_flag), 201);
  // EXPECT_EQ(err, "ERROR: int_flag is already set to 201");
}

}  // namespace
