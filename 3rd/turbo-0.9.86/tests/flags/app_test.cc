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
//
#include "turbo/flags/flags.h"
#include <cmath>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include "flags_test_helper.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

TEST_CASE_FIXTURE(TApp, "OneFlagShort [app]") {
    app.add_flag("-c,--count");
    args = {"-c"};
    run();
    CHECK_EQ(app.count("-c"), 1u);
    CHECK_EQ(app.count("--count"), 1u);
}

TEST_CASE_FIXTURE(TApp, "OneFlagShortValues [app]") {
    app.add_flag("-c{v1},--count{v2}");
    args = {"-c"};
    run();
    CHECK_EQ(app.count("-c"), 1u);
    CHECK_EQ(app.count("--count"), 1u);
    auto v = app["-c"]->results();
    CHECK_EQ("v1",  v[0]);

    CHECK_THROWS_AS(app["--invalid"], turbo::OptionNotFound);
}

TEST_CASE_FIXTURE(TApp, "OneFlagShortValuesAs [app]") {
    auto *flg = app.add_flag("-c{1},--count{2}");
    args = {"-c"};
    run();
    const auto *opt = app["-c"];
    CHECK_EQ(1 , opt->as<int>());
    args = {"--count"};
    run();
    CHECK_EQ(2 , opt->as<int>());
    flg->take_first();
    args = {"-c", "--count"};
    run();
    CHECK_EQ(1 , opt->as<int>());
    flg->take_last();
    CHECK_EQ(2 , opt->as<int>());
    flg->multi_option_policy(turbo::MultiOptionPolicy::Throw);
    CHECK_THROWS_AS((void)opt->as<int>(), turbo::ArgumentMismatch);
    flg->multi_option_policy(turbo::MultiOptionPolicy::TakeAll);
    auto vec = opt->as<std::vector<int>>();
    CHECK_EQ(1 , vec[0]);
    CHECK_EQ(2 , vec[1]);
    flg->multi_option_policy(turbo::MultiOptionPolicy::Join);
    CHECK_EQ("1\n2" , opt->as<std::string>());
    flg->delimiter(',');
    CHECK_EQ("1,2" , opt->as<std::string>());
}

TEST_CASE_FIXTURE(TApp, "OneFlagShortWindows [app]") {
    app.add_flag("-c,--count");
    args = {"/c"};
    app.allow_windows_style_options();
    run();
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--count") , 1u);
}

TEST_CASE_FIXTURE(TApp, "WindowsLongShortMix1 [app]") {
    app.allow_windows_style_options();

    auto *a = app.add_flag("-c");
    auto *b = app.add_flag("--c");
    args = {"/c"};
    run();
    CHECK_EQ(a->count() , 1u);
    CHECK_EQ(b->count() , 0u);
}

TEST_CASE_FIXTURE(TApp, "WindowsLongShortMix2 [app]") {
    app.allow_windows_style_options();

    auto *a = app.add_flag("--c");
    auto *b = app.add_flag("-c");
    args = {"/c"};
    run();
    CHECK_EQ(a->count() , 1u);
    CHECK_EQ(b->count() , 0u);
}

TEST_CASE_FIXTURE(TApp, "CountNonExist [app]") {
    app.add_flag("-c,--count");
    args = {"-c"};
    run();
    CHECK_THROWS_AS((void)app.count("--nonexist"), turbo::OptionNotFound);
}

TEST_CASE_FIXTURE(TApp, "OneFlagLong [app]") {
    app.add_flag("-c,--count");
    args = {"--count"};
    run();
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--count") , 1u);
}

TEST_CASE_FIXTURE(TApp, "DashedOptions [app]") {
    app.add_flag("-c");
    app.add_flag("--q");
    app.add_flag("--this,--that");

    args = {"-c", "--q", "--this", "--that"};
    run();
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--q") , 1u);
    CHECK_EQ(app.count("--this") , 2u);
    CHECK_EQ(app.count("--that") , 2u);
}

TEST_CASE_FIXTURE(TApp, "DashedOptionsSingleString [app]") {
    app.add_flag("-c");
    app.add_flag("--q");
    app.add_flag("--this,--that");

    app.parse("-c --q --this --that");
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--q") , 1u);
    CHECK_EQ(app.count("--this") , 2u);
    CHECK_EQ(app.count("--that") , 2u);
}

TEST_CASE_FIXTURE(TApp, "StrangeFlagNames [app]") {
    app.add_flag("-=");
    app.add_flag("--t\tt");
    app.add_flag("-{");
    CHECK_THROWS_AS(app.add_flag("--t t"), turbo::ConstructionError);
    args = {"-=", "--t\tt"};
    run();
    CHECK_EQ(app.count("-=") , 1u);
    CHECK_EQ(app.count("--t\tt") , 1u);
}

TEST_CASE_FIXTURE(TApp, "BoolFlagOverride [app]") {
    bool val{false};
    auto *flg = app.add_flag("--this,--that", val);

    app.parse("--this");
    CHECK(val);
    app.parse("--this=false");
    CHECK(!val);
    flg->disable_flag_override(true);
    app.parse("--this");
    CHECK(val);
    // this is allowed since the matching string is the default
    app.parse("--this=true");
    CHECK(val);

    CHECK_THROWS_AS(app.parse("--this=false"), turbo::ArgumentMismatch);
    // try a string that specifies 'use default val'
    CHECK_NOTHROW(app.parse("--this={}"));
}

TEST_CASE_FIXTURE(TApp, "OneFlagRef [app]") {
    int ref{0};
    app.add_flag("-c,--count", ref);
    args = {"--count"};
    run();
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--count") , 1u);
    CHECK_EQ(ref , 1);
}

TEST_CASE_FIXTURE(TApp, "OneFlagRefValue [app]") {
    int ref{0};
    app.add_flag("-c,--count", ref);
    args = {"--count=7"};
    run();
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--count") , 1u);
    CHECK_EQ(ref , 7);
}

TEST_CASE_FIXTURE(TApp, "OneFlagRefValueFalse [app]") {
    int ref{0};
    auto *flg = app.add_flag("-c,--count", ref);
    args = {"--count=false"};
    run();
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--count") , 1u);
    CHECK_EQ(ref , -1);

    CHECK(!flg->check_fname("c"));
    args = {"--count=0"};
    run();
    CHECK_EQ(app.count("-c") , 1u);
    CHECK_EQ(app.count("--count") , 1u);
    CHECK_EQ(ref , 0);

    args = {"--count=happy"};
    CHECK_THROWS_AS(run(), turbo::ConversionError);
}

TEST_CASE_FIXTURE(TApp, "FlagNegation [app]") {
    int ref{0};
    auto *flg = app.add_flag("-c,--count,--ncount{false}", ref);
    args = {"--count", "-c", "--ncount"};
    CHECK(!flg->check_fname("count"));
    CHECK(flg->check_fname("ncount"));
    run();
    CHECK_EQ(app.count("-c") , 3u);
    CHECK_EQ(app.count("--count") , 3u);
    CHECK_EQ(app.count("--ncount") , 3u);
    CHECK_EQ(ref , 1);
}

TEST_CASE_FIXTURE(TApp, "FlagNegationShortcutNotation [app]") {
    int ref{0};
    app.add_flag("-c,--count{true},!--ncount", ref);
    args = {"--count=TRUE", "-c", "--ncount"};
    run();
    CHECK_EQ(app.count("-c") , 3u);
    CHECK_EQ(app.count("--count") , 3u);
    CHECK_EQ(app.count("--ncount") , 3u);
    CHECK_EQ(ref , 1);
}

TEST_CASE_FIXTURE(TApp, "FlagNegationShortcutNotationInvalid [app]") {
    int ref{0};
    app.add_flag("-c,--count,!--ncount", ref);
    args = {"--ncount=happy"};
    CHECK_THROWS_AS(run(), turbo::ConversionError);
}

TEST_CASE_FIXTURE(TApp, "OneString [app]") {
    std::string str;
    app.add_option("-s,--string", str);
    args = {"--string", "mystring"};
    run();
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK_EQ("mystring" , str);
}

TEST_CASE_FIXTURE(TApp, "OneStringWindowsStyle [app]") {
    std::string str;
    app.add_option("-s,--string", str);
    args = {"/string", "mystring"};
    app.allow_windows_style_options();
    run();
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK_EQ("mystring" , str);
}

TEST_CASE_FIXTURE(TApp, "OneStringSingleStringInput [app]") {
    std::string str;
    app.add_option("-s,--string", str);

    app.parse("--string mystring");
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK_EQ("mystring" , str);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersion [app]") {
    std::string str;
    app.add_option("-s,--string", str);
    args = {"--string=mystring"};
    run();
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK_EQ("mystring" , str);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionWindowsStyle [app]") {
    std::string str;
    app.add_option("-s,--string", str);
    args = {"/string:mystring"};
    app.allow_windows_style_options();
    run();
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK_EQ("mystring" , str);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleString [app]") {
    std::string str;
    app.add_option("-s,--string", str);
    app.parse("--string=mystring");
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK_EQ("mystring" , str);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringQuoted [app]") {
    std::string str;
    app.add_option("-s,--string", str);
    app.parse(R"raw(--string="this is my quoted string")raw");
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK_EQ("this is my quoted string" , str);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringQuotedMultiple [app]") {
    std::string str, str2, str3;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("-m,--mstr", str3);
    app.parse(R"raw(--string="this is my quoted string" -t 'qstring 2' -m=`"quoted string"`)raw");
    CHECK_EQ("this is my quoted string" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringEmbeddedEqual [app]") {
    std::string str, str2, str3;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("-m,--mstr", str3);
    app.parse(R"raw(--string="app=\"test1 b\" test2=\"frogs\"" -t 'qstring 2' -m=`"quoted string"`)raw");
    CHECK_EQ("app=\"test1 b\" test2=\"frogs\"" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);

    app.parse(R"raw(--string="app='test1 b' test2='frogs'" -t 'qstring 2' -m=`"quoted string"`)raw");
    CHECK_EQ("app='test1 b' test2='frogs'" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringEmbeddedEqualWindowsStyle [app]") {
    std::string str, str2, str3;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("--mstr", str3);
    app.allow_windows_style_options();
    app.parse(R"raw(/string:"app:\"test1 b\" test2:\"frogs\"" /t 'qstring 2' /mstr:`"quoted string"`)raw");
    CHECK_EQ("app:\"test1 b\" test2:\"frogs\"" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);

    app.parse(R"raw(/string:"app:'test1 b' test2:'frogs'" /t 'qstring 2' /mstr:`"quoted string"`)raw");
    CHECK_EQ("app:'test1 b' test2:'frogs'" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringQuotedMultipleMixedStyle [app]") {
    std::string str, str2, str3;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("-m,--mstr", str3);
    app.allow_windows_style_options();
    app.parse(R"raw(/string:"this is my quoted string" /t 'qstring 2' -m=`"quoted string"`)raw");
    CHECK_EQ("this is my quoted string" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringQuotedMultipleInMiddle [app]") {
    std::string str, str2, str3;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("-m,--mstr", str3);
    app.parse(R"raw(--string="this is my quoted string" -t "qst\"ring 2" -m=`"quoted string"`)raw");
    CHECK_EQ("this is my quoted string" , str);
    CHECK_EQ("qst\"ring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringQuotedEscapedCharacters [app]") {
    std::string str, str2, str3;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("-m,--mstr", str3);
    app.parse(R"raw(--string="this is my \"quoted\" string" -t 'qst\'ring 2' -m=`"quoted\` string"`")raw");
    CHECK_EQ("this is my \"quoted\" string" , str);
    CHECK_EQ("qst\'ring 2" , str2);
    CHECK_EQ("\"quoted` string\"" , str3);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringQuotedMultipleWithEqual [app]") {
    std::string str, str2, str3, str4;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("-m,--mstr", str3);
    app.add_option("-j,--jstr", str4);
    app.parse(R"raw(--string="this is my quoted string" -t 'qstring 2' -m=`"quoted string"` --jstr=Unquoted)raw");
    CHECK_EQ("this is my quoted string" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);
    CHECK_EQ("Unquoted" , str4);
}

TEST_CASE_FIXTURE(TApp, "OneStringEqualVersionSingleStringQuotedMultipleWithEqualAndProgram [app]") {
    std::string str, str2, str3, str4;
    app.add_option("-s,--string", str);
    app.add_option("-t,--tstr", str2);
    app.add_option("-m,--mstr", str3);
    app.add_option("-j,--jstr", str4);
    app.parse(
            R"raw(program --string="this is my quoted string" -t 'qstring 2' -m=`"quoted string"` --jstr=Unquoted)raw",
            true);
    CHECK_EQ("this is my quoted string" , str);
    CHECK_EQ("qstring 2" , str2);
    CHECK_EQ("\"quoted string\"" , str3);
    CHECK_EQ("Unquoted" , str4);
}

TEST_CASE_FIXTURE(TApp, "OneStringFlagLike [app]") {
    std::string str{"something"};
    app.add_option("-s,--string", str)->expected(0, 1);
    args = {"--string"};
    run();
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(app.count("--string") , 1u);
    CHECK(str.empty());
}

TEST_CASE_FIXTURE(TApp, "OneIntFlagLike [app]") {
    int val{0};
    auto *opt = app.add_option("-i", val)->expected(0, 1);
    args = {"-i"};
    run();
    CHECK_EQ(app.count("-i") , 1u);
    opt->default_str("7");
    run();
    CHECK_EQ(7 , val);

    opt->default_val(9);
    run();
    CHECK_EQ(9 , val);
}

TEST_CASE_FIXTURE(TApp, "TogetherInt [app]") {
    int i{0};
    app.add_option("-i,--int", i);
    args = {"-i4"};
    run();
    CHECK_EQ(app.count("--int") , 1u);
    CHECK_EQ(app.count("-i") , 1u);
    CHECK_EQ(4 , i);
    CHECK_EQ("4" , app["-i"]->as<std::string>());
    CHECK_EQ(4.0 , app["--int"]->as<double>());
}

TEST_CASE_FIXTURE(TApp, "SepInt [app]") {
    int i{0};
    app.add_option("-i,--int", i);
    args = {"-i", "4"};
    run();
    CHECK_EQ(app.count("--int") , 1u);
    CHECK_EQ(app.count("-i") , 1u);
    CHECK_EQ(4 , i);
}

TEST_CASE_FIXTURE(TApp, "DefaultStringAgain [app]") {
    std::string str = "previous";
    app.add_option("-s,--string", str);
    run();
    CHECK_EQ(app.count("-s") , 0u);
    CHECK_EQ(app.count("--string") , 0u);
    CHECK_EQ("previous" , str);
}

TEST_CASE_FIXTURE(TApp, "DefaultStringAgainEmpty [app]") {
    std::string str = "previous";
    app.add_option("-s,--string", str);
    app.parse("   ");
    CHECK_EQ(app.count("-s") , 0u);
    CHECK_EQ(app.count("--string") , 0u);
    CHECK_EQ("previous" , str);
}

TEST_CASE_FIXTURE(TApp, "DualOptions [app]") {

    std::string str = "previous";
    std::vector<std::string> vstr = {"previous"};
    std::vector<std::string> ans = {"one", "two"};
    app.add_option("-s,--string", str);
    app.add_option("-v,--vector", vstr);

    args = {"--vector=one", "--vector=two"};
    run();
    CHECK_EQ(vstr , ans);

    args = {"--string=one", "--string=two"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "LotsOfFlags [app]") {

    app.add_flag("-a");
    app.add_flag("-A");
    app.add_flag("-b");

    args = {"-a", "-b", "-aA"};
    run();
    CHECK_EQ(app.count("-a") , 2u);
    CHECK_EQ(app.count("-b") , 1u);
    CHECK_EQ(app.count("-A") , 1u);
    CHECK_EQ(4u , app.count_all());
}

TEST_CASE_FIXTURE(TApp, "NumberFlags [app]") {

    int val{0};
    app.add_flag("-1{1},-2{2},-3{3},-4{4},-5{5},-6{6}, -7{7}, -8{8}, -9{9}", val);

    args = {"-7"};
    run();
    CHECK_EQ(app.count("-1") , 1u);
    CHECK_EQ(7 , val);
}

TEST_CASE_FIXTURE(TApp, "DisableFlagOverrideTest [app]") {

    int val{0};
    auto *opt = app.add_flag("--1{1},--2{2},--3{3},--4{4},--5{5},--6{6}, --7{7}, --8{8}, --9{9}", val);
    CHECK(!opt->get_disable_flag_override());
    opt->disable_flag_override();
    args = {"--7=5"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
    CHECK(opt->get_disable_flag_override());
    opt->disable_flag_override(false);
    CHECK(!opt->get_disable_flag_override());
    CHECK_NOTHROW(run());
    CHECK_EQ(5 , val);
    opt->disable_flag_override();
    args = {"--7=7"};
    CHECK_NOTHROW(run());
}

TEST_CASE_FIXTURE(TApp, "LotsOfFlagsSingleString [app]") {

    app.add_flag("-a");
    app.add_flag("-A");
    app.add_flag("-b");

    app.parse("-a -b -aA");
    CHECK_EQ(app.count("-a") , 2u);
    CHECK_EQ(app.count("-b") , 1u);
    CHECK_EQ(app.count("-A") , 1u);
}

TEST_CASE_FIXTURE(TApp, "LotsOfFlagsSingleStringExtraSpace [app]") {

    app.add_flag("-a");
    app.add_flag("-A");
    app.add_flag("-b");

    app.parse("  -a    -b    -aA   ");
    CHECK_EQ(app.count("-a") , 2u);
    CHECK_EQ(app.count("-b") , 1u);
    CHECK_EQ(app.count("-A") , 1u);
}

TEST_CASE_FIXTURE(TApp, "SingleArgVector [app]") {

    std::vector<std::string> channels;
    std::vector<std::string> iargs;
    std::string path;
    app.add_option("-c", channels)->type_size(1)->allow_extra_args(false);
    app.add_option("args", iargs);
    app.add_option("-p", path);

    app.parse("-c t1 -c t2 -c t3 a1 a2 a3 a4 -p happy");
    CHECK_EQ(channels.size() , 3u);
    CHECK_EQ(iargs.size() , 4u);
    CHECK_EQ("happy" , path);

    app.parse("-c t1 a1 -c t2 -c t3 a2 a3 a4 -p happy");
    CHECK_EQ(channels.size() , 3u);
    CHECK_EQ(iargs.size() , 4u);
    CHECK_EQ("happy" , path);
}

TEST_CASE_FIXTURE(TApp, "StrangeOptionNames [app]") {
    app.add_option("-:");
    app.add_option("--t\tt");
    app.add_option("--{}");
    app.add_option("--:)");
    CHECK_THROWS_AS(app.add_option("--t t"), turbo::ConstructionError);
    args = {"-:)", "--{}", "5"};
    run();
    CHECK_EQ(app.count("-:") , 1u);
    CHECK_EQ(app.count("--{}") , 1u);
    CHECK_EQ(app["-:"]->as<char>() , ')');
    CHECK_EQ(app["--{}"]->as<int>() , 5);
}

TEST_CASE_FIXTURE(TApp, "FlagLikeOption [app]") {
    bool val{false};
    auto *opt = app.add_option("--flag", val)->type_size(0)->default_str("true");
    args = {"--flag"};
    run();
    CHECK_EQ(app.count("--flag") , 1u);
    CHECK(val);
    val = false;
    opt->type_size(0, 0);  // should be the same as above
    CHECK_EQ(0 , opt->get_type_size_min());
    CHECK_EQ(0 , opt->get_type_size_max());
    run();
    CHECK_EQ(app.count("--flag") , 1u);
    CHECK(val);
}

TEST_CASE_FIXTURE(TApp, "FlagLikeIntOption [app]") {
    int val{-47};
    auto *opt = app.add_option("--flag", val)->expected(0, 1);
    // normally some default value should be set, but this test is for some paths in the validators checks to skip
    // validation on empty string if nothing is expected
    opt->check(turbo::PositiveNumber);
    args = {"--flag"};
    CHECK(opt->as<std::string>().empty());
    run();
    CHECK_EQ(app.count("--flag") , 1u);
    CHECK(-47 != val);
    args = {"--flag", "12"};
    run();

    CHECK_EQ(12 , val);
    args.clear();
    run();
    CHECK(opt->as<std::string>().empty());
}

TEST_CASE_FIXTURE(TApp, "BoolOnlyFlag [app]") {
    bool bflag{false};
    app.add_flag("-b", bflag)->multi_option_policy(turbo::MultiOptionPolicy::Throw);

    args = {"-b"};
    REQUIRE_NOTHROW(run());
    CHECK(bflag);

    args = {"-b", "-b"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "ShortOpts [app]") {

    std::uint64_t funnyint{0};
    std::string someopt;
    app.add_flag("-z", funnyint);
    app.add_option("-y", someopt);

    args = {
            "-zzyzyz",
    };

    run();

    CHECK_EQ(app.count("-z") , 2u);
    CHECK_EQ(app.count("-y") , 1u);
    CHECK_EQ(funnyint , std::uint64_t{2});
    CHECK_EQ(someopt , "zyz");
    CHECK_EQ(3u , app.count_all());
}

TEST_CASE_FIXTURE(TApp, "TwoParamTemplateOpts [app]") {

    double funnyint{0.0};
    auto *opt = app.add_option<double, unsigned int>("-y", funnyint);

    args = {"-y", "32"};

    run();

    CHECK_EQ(funnyint , 32.0);

    args = {"-y", "32.3"};
    CHECK_THROWS_AS(run(), turbo::ConversionError);

    args = {"-y", "-19"};
    CHECK_THROWS_AS(run(), turbo::ConversionError);

    opt->capture_default_str();
    CHECK(opt->get_default_str().empty());
}

TEST_CASE_FIXTURE(TApp, "DefaultOpts [app]") {

    int i{3};
    std::string s = "HI";

    app.add_option("-i,i", i);
    app.add_option("-s,s", s)->capture_default_str();  //  Used to be different

    args = {"-i2", "9"};

    run();

    CHECK_EQ(app.count("i") , 1u);
    CHECK_EQ(app.count("-s") , 1u);
    CHECK_EQ(i , 2);
    CHECK_EQ(s , "9");
}

TEST_CASE_FIXTURE(TApp, "TakeLastOpt [app]") {

    std::string str;
    app.add_option("--str", str)->multi_option_policy(turbo::MultiOptionPolicy::TakeLast);

    args = {"--str=one", "--str=two"};

    run();

    CHECK_EQ("two" , str);
}

TEST_CASE_FIXTURE(TApp, "TakeLastOpt2 [app]") {

    std::string str;
    app.add_option("--str", str)->take_last();

    args = {"--str=one", "--str=two"};

    run();

    CHECK_EQ("two" , str);
}

TEST_CASE_FIXTURE(TApp, "TakeFirstOpt [app]") {

    std::string str;
    app.add_option("--str", str)->multi_option_policy(turbo::MultiOptionPolicy::TakeFirst);

    args = {"--str=one", "--str=two"};

    run();

    CHECK_EQ("one" , str);
}

TEST_CASE_FIXTURE(TApp, "TakeFirstOpt2 [app]") {

    std::string str;
    app.add_option("--str", str)->take_first();

    args = {"--str=one", "--str=two"};

    run();

    CHECK_EQ("one" , str);
}

TEST_CASE_FIXTURE(TApp, "JoinOpt [app]") {

    std::string str;
    app.add_option("--str", str)->multi_option_policy(turbo::MultiOptionPolicy::Join);

    args = {"--str=one", "--str=two"};

    run();

    CHECK_EQ("one\ntwo" , str);
}

TEST_CASE_FIXTURE(TApp, "SumOpt [app]") {

    int val = 0;
    app.add_option("--val", val)->multi_option_policy(turbo::MultiOptionPolicy::Sum);

    args = {"--val=1", "--val=4"};

    run();

    CHECK_EQ(5 , val);
}

TEST_CASE_FIXTURE(TApp, "SumOptFloat [app]") {

    double val = NAN;
    app.add_option("--val", val)->multi_option_policy(turbo::MultiOptionPolicy::Sum);

    args = {"--val=1.3", "--val=-0.7"};

    run();

    CHECK_EQ(0.6 , val);
}

TEST_CASE_FIXTURE(TApp, "SumOptString [app]") {

    std::string val;
    app.add_option("--val", val)->multi_option_policy(turbo::MultiOptionPolicy::Sum);

    args = {"--val=i", "--val=2"};

    run();

    CHECK_EQ("i2" , val);
}

TEST_CASE_FIXTURE(TApp, "JoinOpt2 [app]") {

    std::string str;
    app.add_option("--str", str)->join();

    args = {"--str=one", "--str=two"};

    run();

    CHECK_EQ("one\ntwo" , str);
}

TEST_CASE_FIXTURE(TApp, "TakeLastOptMulti [app]") {
    std::vector<int> vals;
    app.add_option("--long", vals)->expected(2)->take_last();

    args = {"--long", "1", "2", "3"};

    run();

    CHECK_EQ(std::vector<int>({2, 3}) , vals);
}

TEST_CASE_FIXTURE(TApp, "TakeLastOptMulti_alternative_path [app]") {
    std::vector<int> vals;
    app.add_option("--long", vals)->expected(2, -1)->take_last();

    args = {"--long", "1", "2", "3"};

    run();

    CHECK_EQ(std::vector<int>({2, 3}) , vals);
}

TEST_CASE_FIXTURE(TApp, "TakeLastOptMultiCheck [app]") {
    std::vector<int> vals;
    auto *opt = app.add_option("--long", vals)->expected(-2)->take_last();

    opt->check(turbo::Validator(turbo::PositiveNumber).application_index(0));
    opt->check((!turbo::PositiveNumber).application_index(1));
    args = {"--long", "-1", "2", "-3"};

    CHECK_NOTHROW(run());

    CHECK_EQ(std::vector<int>({2, -3}) , vals);
}

TEST_CASE_FIXTURE(TApp, "TakeFirstOptMulti [app]") {
    std::vector<int> vals;
    app.add_option("--long", vals)->expected(2)->take_first();

    args = {"--long", "1", "2", "3"};

    run();

    CHECK_EQ(std::vector<int>({1, 2}) , vals);
}

TEST_CASE_FIXTURE(TApp, "MissingValueNonRequiredOpt [app]") {
    int count{0};
    app.add_option("-c,--count", count);

    args = {"-c"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);

    args = {"--count"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "MissingValueMoreThan [app]") {
    std::vector<int> vals1;
    std::vector<int> vals2;
    app.add_option("-v", vals1)->expected(-2);
    app.add_option("--vals", vals2)->expected(-2);

    args = {"-v", "2"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);

    args = {"--vals", "4"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "NoMissingValueMoreThan [app]") {
    std::vector<int> vals1;
    std::vector<int> vals2;
    app.add_option("-v", vals1)->expected(-2);
    app.add_option("--vals", vals2)->expected(-2);

    args = {"-v", "2", "3", "4"};
    run();
    CHECK_EQ(std::vector<int>({2, 3, 4}) , vals1);

    args = {"--vals", "2", "3", "4"};
    run();
    CHECK_EQ(std::vector<int>({2, 3, 4}) , vals2);
}

TEST_CASE_FIXTURE(TApp, "NotRequiredOptsSingle [app]") {

    std::string str;
    app.add_option("--str", str);

    args = {"--str"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "NotRequiredOptsSingleShort [app]") {

    std::string str;
    app.add_option("-s", str);

    args = {"-s"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "RequiredOptsSingle [app]") {

    std::string str;
    app.add_option("--str", str)->required();

    args = {"--str"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "RequiredOptsSingleShort [app]") {

    std::string str;
    app.add_option("-s", str)->required();

    args = {"-s"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "RequiredOptsDouble [app]") {

    std::vector<std::string> strs;
    app.add_option("--str", strs)->required()->expected(2);

    args = {"--str", "one"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);

    args = {"--str", "one", "two"};

    run();

    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);
}

TEST_CASE_FIXTURE(TApp, "emptyVectorReturn [app]") {

    std::vector<std::string> strs;
    std::vector<std::string> strs2;
    std::vector<std::string> strs3;
    auto *opt1 = app.add_option("--str", strs)->required()->expected(0, 2);
    app.add_option("--str3", strs3)->expected(1, 3);
    app.add_option("--str2", strs2);
    args = {"--str"};

    CHECK_NOTHROW(run());
    CHECK_EQ(std::vector<std::string>({""}) , strs);
    args = {"--str", "one", "two"};

    run();

    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);

    args = {"--str", "{}", "--str2", "{}"};

    run();

    CHECK(strs.empty());
    CHECK_EQ(std::vector<std::string>{"{}"} , strs2);
    opt1->default_str("{}");
    args = {"--str"};

    CHECK_NOTHROW(run());
    CHECK(strs.empty());
    opt1->required(false);
    args = {"--str3", "{}"};

    CHECK_NOTHROW(run());
    CHECK_FALSE(strs3.empty());
}

TEST_CASE_FIXTURE(TApp, "RequiredOptsDoubleShort [app]") {

    std::vector<std::string> strs;
    app.add_option("-s", strs)->required()->expected(2);

    args = {"-s", "one"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);

    args = {"-s", "one", "-s", "one", "-s", "one"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "RequiredOptsDoubleNeg [app]") {
    std::vector<std::string> strs;
    app.add_option("-s", strs)->required()->expected(-2);

    args = {"-s", "one"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);

    args = {"-s", "one", "two", "-s", "three"};

    REQUIRE_NOTHROW(run());
    CHECK_EQ(std::vector<std::string>({"one", "two", "three"}) , strs);

    args = {"-s", "one", "two"};
    REQUIRE_NOTHROW(run());
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);
}

// This makes sure unlimited option priority is
// correct for space vs. no space #90
TEST_CASE_FIXTURE(TApp, "PositionalNoSpace [app]") {
    std::vector<std::string> options;
    std::string foo, bar;

    app.add_option("-O", options);
    app.add_option("foo", foo)->required();
    app.add_option("bar", bar)->required();

    args = {"-O", "Test", "param1", "param2"};
    run();

    CHECK_EQ(1u , options.size());
    CHECK_EQ("Test" , options.at(0));

    args = {"-OTest", "param1", "param2"};
    run();

    CHECK_EQ(1u , options.size());
    CHECK_EQ("Test" , options.at(0));
}

// Tests positionals at end
TEST_CASE_FIXTURE(TApp, "PositionalAtEnd [app]") {
    std::string options;
    std::string foo;

    app.add_option("-O", options);
    app.add_option("foo", foo);
    app.positionals_at_end();
    CHECK(app.get_positionals_at_end());
    args = {"-O", "Test", "param1"};
    run();

    CHECK_EQ("Test" , options);
    CHECK_EQ("param1" , foo);

    args = {"param2", "-O", "Test"};
    CHECK_THROWS_AS(run(), turbo::ExtrasError);
}

// Tests positionals at end
TEST_CASE_FIXTURE(TApp, "RequiredPositionals [app]") {
    std::vector<std::string> sources;
    std::string dest;
    app.add_option("src", sources);
    app.add_option("dest", dest)->required();
    app.positionals_at_end();

    args = {"1", "2", "3"};
    run();

    CHECK_EQ(2u , sources.size());
    CHECK_EQ("3" , dest);

    args = {"a"};
    sources.clear();
    run();

    CHECK(sources.empty());
    CHECK_EQ("a" , dest);
}

TEST_CASE_FIXTURE(TApp, "RequiredPositionalVector [app]") {
    std::string d1;
    std::string d2;
    std::string d3;
    std::vector<std::string> sources;

    app.add_option("dest1", d1);
    app.add_option("dest2", d2);
    app.add_option("dest3", d3);
    app.add_option("src", sources)->required();

    app.positionals_at_end();

    args = {"1", "2", "3"};
    run();

    CHECK_EQ(1u , sources.size());
    CHECK_EQ("1" , d1);
    CHECK_EQ("2" , d2);
    CHECK(d3.empty());
    args = {"a"};
    sources.clear();
    run();

    CHECK_EQ(1u , sources.size());
}

// Tests positionals at end
TEST_CASE_FIXTURE(TApp, "RequiredPositionalValidation [app]") {
    std::vector<std::string> sources;
    int dest = 0;  // required
    std::string d2;
    app.add_option("src", sources);
    app.add_option("dest", dest)->required()->check(turbo::PositiveNumber);
    app.add_option("dest2", d2)->required();
    app.positionals_at_end()->validate_positionals();

    args = {"1", "2", "string", "3"};
    run();

    CHECK_EQ(2u , sources.size());
    CHECK_EQ(3 , dest);
    CHECK_EQ("string" , d2);
}

// Tests positionals at end
TEST_CASE_FIXTURE(TApp, "PositionalValidation [app]") {
    std::string options;
    std::string foo;

    app.add_option("bar", options)->check(turbo::Number.name("valbar"));
    // disable the check on foo
    app.add_option("foo", foo)->check(turbo::Number.active(false));
    app.validate_positionals();
    args = {"1", "param1"};
    run();

    CHECK_EQ("1" , options);
    CHECK_EQ("param1" , foo);

    args = {"param1", "1"};
    CHECK_NOTHROW(run());

    CHECK_EQ("1" , options);
    CHECK_EQ("param1" , foo);

    CHECK(nullptr != app.get_option("bar")->get_validator("valbar"));
}

TEST_CASE_FIXTURE(TApp, "PositionalNoSpaceLong [app]") {
    std::vector<std::string> options;
    std::string foo, bar;

    app.add_option("--option", options);
    app.add_option("foo", foo)->required();
    app.add_option("bar", bar)->required();

    args = {"--option", "Test", "param1", "param2"};
    run();

    CHECK_EQ(1u , options.size());
    CHECK_EQ("Test" , options.at(0));

    args = {"--option=Test", "param1", "param2"};
    run();

    CHECK_EQ(1u , options.size());
    CHECK_EQ("Test" , options.at(0));
}

TEST_CASE_FIXTURE(TApp, "RequiredOptsUnlimited [app]") {

    std::vector<std::string> strs;
    app.add_option("--str", strs)->required();

    args = {"--str"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);

    args = {"--str", "one", "--str", "two"};
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);

    args = {"--str", "one", "two"};
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);

    // It's better to feed a hungry option than to feed allow_extras
    app.allow_extras();
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);
    CHECK_EQ(std::vector<std::string>({}) , app.remaining());

    app.allow_extras(false);
    std::vector<std::string> remain;
    auto *popt = app.add_option("positional", remain);
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);
    CHECK(remain.empty());

    args = {"--str", "one", "--", "two"};

    run();
    CHECK_EQ(std::vector<std::string>({"one"}) , strs);
    CHECK_EQ(std::vector<std::string>({"two"}) , remain);

    args = {"one", "--str", "two"};

    run();
    CHECK_EQ(std::vector<std::string>({"two"}) , strs);
    CHECK_EQ(std::vector<std::string>({"one"}) , remain);

    args = {"--str", "one", "two"};
    popt->required();
    run();
    CHECK_EQ(std::vector<std::string>({"one"}) , strs);
    CHECK_EQ(std::vector<std::string>({"two"}) , remain);
}

TEST_CASE_FIXTURE(TApp, "RequiredOptsUnlimitedShort [app]") {

    std::vector<std::string> strs;
    app.add_option("-s", strs)->required();

    args = {"-s"};
    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);

    args = {"-s", "one", "-s", "two"};
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);

    args = {"-s", "one", "two"};
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);

    // It's better to feed a hungry option than to feed allow_extras
    app.allow_extras();
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);
    CHECK_EQ(std::vector<std::string>({}) , app.remaining());

    app.allow_extras(false);
    std::vector<std::string> remain;
    app.add_option("positional", remain);
    run();
    CHECK_EQ(std::vector<std::string>({"one", "two"}) , strs);
    CHECK(remain.empty());

    args = {"-s", "one", "--", "two"};

    run();
    CHECK_EQ(std::vector<std::string>({"one"}) , strs);
    CHECK_EQ(std::vector<std::string>({"two"}) , remain);

    args = {"one", "-s", "two"};

    run();
    CHECK_EQ(std::vector<std::string>({"two"}) , strs);
    CHECK_EQ(std::vector<std::string>({"one"}) , remain);
}

TEST_CASE_FIXTURE(TApp, "OptsUnlimitedEnd [app]") {
    std::vector<std::string> strs;
    app.add_option("-s,--str", strs);
    app.allow_extras();

    args = {"one", "-s", "two", "three", "--", "four"};

    run();

    CHECK_EQ(std::vector<std::string>({"two", "three"}) , strs);
    CHECK_EQ(std::vector<std::string>({"one", "four"}) , app.remaining());
}

TEST_CASE_FIXTURE(TApp, "RequireOptPriority [app]") {

    std::vector<std::string> strs;
    app.add_option("--str", strs);
    std::vector<std::string> remain;
    app.add_option("positional", remain)->expected(2)->required();

    args = {"--str", "one", "two", "three"};
    run();

    CHECK_EQ(std::vector<std::string>({"one"}) , strs);
    CHECK_EQ(std::vector<std::string>({"two", "three"}) , remain);

    args = {"two", "three", "--str", "one", "four"};
    run();

    CHECK_EQ(std::vector<std::string>({"one", "four"}) , strs);
    CHECK_EQ(std::vector<std::string>({"two", "three"}) , remain);
}

TEST_CASE_FIXTURE(TApp, "RequireOptPriorityShort [app]") {

    std::vector<std::string> strs;
    app.add_option("-s", strs)->required();
    std::vector<std::string> remain;
    app.add_option("positional", remain)->expected(2)->required();

    args = {"-s", "one", "two", "three"};
    run();

    CHECK_EQ(std::vector<std::string>({"one"}) , strs);
    CHECK_EQ(std::vector<std::string>({"two", "three"}) , remain);

    args = {"two", "three", "-s", "one", "four"};
    run();

    CHECK_EQ(std::vector<std::string>({"one", "four"}) , strs);
    CHECK_EQ(std::vector<std::string>({"two", "three"}) , remain);
}

TEST_CASE_FIXTURE(TApp, "NotRequiredExpectedDouble [app]") {

    std::vector<std::string> strs;
    app.add_option("--str", strs)->expected(2);

    args = {"--str", "one"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "NotRequiredExpectedDoubleShort [app]") {

    std::vector<std::string> strs;
    app.add_option("-s", strs)->expected(2);

    args = {"-s", "one"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

TEST_CASE_FIXTURE(TApp, "RequiredFlags [app]") {
    app.add_flag("-a")->required();
    app.add_flag("-b")->mandatory();  // Alternate term

    CHECK_THROWS_AS(run(), turbo::RequiredError);

    args = {"-a"};
    CHECK_THROWS_AS(run(), turbo::RequiredError);

    args = {"-b"};
    CHECK_THROWS_AS(run(), turbo::RequiredError);

    args = {"-a", "-b"};
    run();
}

TEST_CASE_FIXTURE(TApp, "CallbackFlags [app]") {

    std::int64_t value{0};

    auto func = [&value](std::int64_t x) { value = x; };

    app.add_flag_function("-v", func);

    run();
    CHECK_EQ(0u , value);

    args = {"-v"};
    run();
    CHECK_EQ(1u , value);

    args = {"-vv"};
    run();
    CHECK_EQ(2u , value);

    CHECK_THROWS_AS(app.add_flag_function("hi", func), turbo::IncorrectConstruction);
}

TEST_CASE_FIXTURE(TApp, "CallbackFlagsFalse [app]") {
    std::int64_t value = 0;

    auto func = [&value](std::int64_t x) { value = x; };

    app.add_flag_function("-v,-f{false},--val,--fval{false}", func);

    run();
    CHECK_EQ(0 , value);

    args = {"-f"};
    run();
    CHECK_EQ(-1 , value);

    args = {"-vfv"};
    run();
    CHECK_EQ(1 , value);

    args = {"--fval"};
    run();
    CHECK_EQ(-1 , value);

    args = {"--fval=2"};
    run();
    CHECK_EQ(-2 , value);

    CHECK_THROWS_AS(app.add_flag_function("hi", func), turbo::IncorrectConstruction);
}

TEST_CASE_FIXTURE(TApp, "CallbackFlagsFalseShortcut [app]") {
    std::int64_t value = 0;

    auto func = [&value](std::int64_t x) { value = x; };

    app.add_flag_function("-v,!-f,--val,!--fval", func);

    run();
    CHECK_EQ(0 , value);

    args = {"-f"};
    run();
    CHECK_EQ(-1 , value);

    args = {"-vfv"};
    run();
    CHECK_EQ(1 , value);

    args = {"--fval"};
    run();
    CHECK_EQ(-1 , value);

    args = {"--fval=2"};
    run();
    CHECK_EQ(-2 , value);

    CHECK_THROWS_AS(app.add_flag_function("hi", func), turbo::IncorrectConstruction);
}

#if __cplusplus >= 201402L || _MSC_VER >= 1900
TEST_CASE_FIXTURE(TApp, "CallbackFlagsAuto [app]") {

    std::int64_t value{0};

    auto func = [&value](std::int64_t x) { value = x; };

    app.add_flag("-v", func);

    run();
    CHECK_EQ(0u , value);

    args = {"-v"};
    run();
    CHECK_EQ(1u , value);

    args = {"-vv"};
    run();
    CHECK_EQ(2u , value);

    CHECK_THROWS_AS(app.add_flag("hi", func), turbo::IncorrectConstruction);
}

#endif

TEST_CASE_FIXTURE(TApp, "Positionals [app]") {

    std::string posit1;
    std::string posit2;
    app.add_option("posit1", posit1);
    app.add_option("posit2", posit2);

    args = {"thing1", "thing2"};

    run();

    CHECK_EQ(app.count("posit1") , 1u);
    CHECK_EQ(app.count("posit2") , 1u);
    CHECK_EQ(posit1 , "thing1");
    CHECK_EQ(posit2 , "thing2");
}

TEST_CASE_FIXTURE(TApp, "ForcedPositional [app]") {
    std::vector<std::string> posit;
    auto *one = app.add_flag("--one");
    app.add_option("posit", posit);

    args = {"--one", "two", "three"};
    run();
    std::vector<std::string> answers1 = {"two", "three"};
    CHECK(one->count());
    CHECK_EQ(posit , answers1);

    args = {"--", "--one", "two", "three"};
    std::vector<std::string> answers2 = {"--one", "two", "three"};
    run();

    CHECK(!one->count());
    CHECK_EQ(posit , answers2);
}

TEST_CASE_FIXTURE(TApp, "MixedPositionals [app]") {

    int positional_int{0};
    std::string positional_string;
    app.add_option("posit1,--posit1", positional_int, "");
    app.add_option("posit2,--posit2", positional_string, "");

    args = {"--posit2", "thing2", "7"};

    run();

    CHECK_EQ(app.count("posit2") , 1u);
    CHECK_EQ(app.count("--posit1") , 1u);
    CHECK_EQ(positional_int , 7);
    CHECK_EQ(positional_string , "thing2");
}

TEST_CASE_FIXTURE(TApp, "BigPositional [app]") {
    std::vector<std::string> vec;
    app.add_option("pos", vec);

    args = {"one"};

    run();
    CHECK_EQ(vec , args);

    args = {"one", "two"};
    run();

    CHECK_EQ(vec , args);
}

TEST_CASE_FIXTURE(TApp, "VectorArgAndPositional [app]") {
    std::vector<std::string> vec;
    std::vector<int> ivec;
    app.add_option("pos", vec);
    app.add_option("--args", ivec)->check(turbo::Number);
    app.validate_optional_arguments();
    args = {"one"};

    run();
    CHECK_EQ(vec , args);

    args = {"--args", "1", "2"};

    run();
    CHECK_EQ(ivec.size() , 2);
    vec.clear();
    ivec.clear();

    args = {"--args", "1", "2", "one", "two"};
    run();

    CHECK_EQ(vec.size() , 2);
    CHECK_EQ(ivec.size() , 2);

    app.validate_optional_arguments(false);
    CHECK_THROWS(run());
}

TEST_CASE_FIXTURE(TApp, "Reset [app]") {

    app.add_flag("--simple");
    double doub{0.0};
    app.add_option("-d,--double", doub);

    args = {"--simple", "--double", "1.2"};

    run();

    CHECK_EQ(app.count("--simple") , 1u);
    CHECK_EQ(app.count("-d") , 1u);
    CHECK_EQ(doub , doctest::Approx(1.2));

    app.clear();

    CHECK_EQ(app.count("--simple") , 0u);
    CHECK_EQ(app.count("-d") , 0u);

    run();

    CHECK_EQ(app.count("--simple") , 1u);
    CHECK_EQ(app.count("-d") , 1u);
    CHECK_EQ(doub , doctest::Approx(1.2));
}

TEST_CASE_FIXTURE(TApp, "RemoveOption [app]") {
    app.add_flag("--one");
    auto *opt = app.add_flag("--two");

    CHECK(app.remove_option(opt));
    CHECK(!app.remove_option(opt));

    args = {"--two"};

    CHECK_THROWS_AS(run(), turbo::ExtrasError);
}

TEST_CASE_FIXTURE(TApp, "RemoveNeedsLinks [app]") {
    auto *one = app.add_flag("--one");
    auto *two = app.add_flag("--two");

    two->needs(one);
    one->needs(two);

    CHECK(app.remove_option(one));

    args = {"--two"};

    run();
}

TEST_CASE_FIXTURE(TApp, "RemoveExcludesLinks [app]") {
    auto *one = app.add_flag("--one");
    auto *two = app.add_flag("--two");

    two->excludes(one);
    one->excludes(two);

    CHECK(app.remove_option(one));

    args = {"--two"};

    run();  // Mostly hoping it does not crash
}

TEST_CASE_FIXTURE(TApp, "FileNotExists [app]") {
    std::string myfile{"TestNonFileNotUsed.txt"};
    REQUIRE_NOTHROW(turbo::NonexistentPath(myfile));

    std::string filename;
    auto *opt = app.add_option("--file", filename)->check(turbo::NonexistentPath, "path_check");
    args = {"--file", myfile};

    run();
    CHECK_EQ(filename , myfile);

    bool ok = static_cast<bool>(std::ofstream(myfile.c_str()).put('a'));  // create file
    CHECK(ok);
    CHECK_THROWS_AS(run(), turbo::ValidationError);
    // deactivate the check, so it should run now
    opt->get_validator("path_check")->active(false);
    CHECK_NOTHROW(run());
    std::remove(myfile.c_str());
    CHECK(!turbo::ExistingFile(myfile).empty());
}

TEST_CASE_FIXTURE(TApp, "FileExists [app]") {
    std::string myfile{"TestNonFileNotUsed.txt"};
    CHECK(!turbo::ExistingFile(myfile).empty());

    std::string filename = "Failed";
    app.add_option("--file", filename)->check(turbo::ExistingFile);
    args = {"--file", myfile};

    CHECK_THROWS_AS(run(), turbo::ValidationError);

    bool ok = static_cast<bool>(std::ofstream(myfile.c_str()).put('a'));  // create file
    CHECK(ok);
    run();
    CHECK_EQ(filename , myfile);

    std::remove(myfile.c_str());
    CHECK(!turbo::ExistingFile(myfile).empty());
}

TEST_CASE_FIXTURE(TApp, "NotFileExists [app]") {
    std::string myfile{"TestNonFileNotUsed.txt"};
    CHECK(!turbo::ExistingFile(myfile).empty());

    std::string filename = "Failed";
    app.add_option("--file", filename)->check(!turbo::ExistingFile);
    args = {"--file", myfile};

    CHECK_NOTHROW(run());

    bool ok = static_cast<bool>(std::ofstream(myfile.c_str()).put('a'));  // create file
    CHECK(ok);
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    std::remove(myfile.c_str());
    CHECK(!turbo::ExistingFile(myfile).empty());
}

TEST_CASE_FIXTURE(TApp, "DefaultedResult [app]") {
    std::string sval = "NA";
    int ival{0};
    auto *opts = app.add_option("--string", sval)->capture_default_str();
    auto *optv = app.add_option("--val", ival);
    args = {};
    run();
    CHECK_EQ("NA" , sval);
    std::string nString;
    opts->results(nString);
    CHECK_EQ("NA" , nString);
    int newIval = 0;
    // CHECK_THROWS_AS (optv->results(newIval), turbo::ConversionError);
    optv->default_str("442");
    optv->results(newIval);
    CHECK_EQ(442 , newIval);
}

TEST_CASE_FIXTURE(TApp, "OriginalOrder [app]") {
    std::vector<int> st1;
    turbo::Option *op1 = app.add_option("-a", st1);
    std::vector<int> st2;
    turbo::Option *op2 = app.add_option("-b", st2);

    args = {"-a", "1", "-b", "2", "-a3", "-a", "4"};

    run();

    CHECK_EQ(std::vector<int>({1, 3, 4}) , st1);
    CHECK_EQ(std::vector<int>({2}) , st2);

    CHECK_EQ(std::vector<turbo::Option *>({op1, op2, op1, op1}) , app.parse_order());
}

TEST_CASE_FIXTURE(TApp, "NeedsFlags [app]") {
    turbo::Option *opt = app.add_flag("-s,--string");
    app.add_flag("--both")->needs(opt);

    run();

    args = {"-s"};
    run();

    args = {"-s", "--both"};
    run();

    args = {"--both"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    CHECK_NOTHROW(opt->needs(opt));
}

TEST_CASE_FIXTURE(TApp, "ExcludesFlags [app]") {
    turbo::Option *opt = app.add_flag("-s,--string");
    app.add_flag("--nostr")->excludes(opt);

    run();

    args = {"-s"};
    run();

    args = {"--nostr"};
    run();

    args = {"--nostr", "-s"};
    CHECK_THROWS_AS(run(), turbo::ExcludesError);

    args = {"--string", "--nostr"};
    CHECK_THROWS_AS(run(), turbo::ExcludesError);

    CHECK_THROWS_AS(opt->excludes(opt), turbo::IncorrectConstruction);
}

TEST_CASE_FIXTURE(TApp, "ExcludesMixedFlags [app]") {
    turbo::Option *opt1 = app.add_flag("--opt1");
    app.add_flag("--opt2");
    turbo::Option *opt3 = app.add_flag("--opt3");
    app.add_flag("--no")->excludes(opt1, "--opt2", opt3);

    run();

    args = {"--no"};
    run();

    args = {"--opt2"};
    run();

    args = {"--no", "--opt1"};
    CHECK_THROWS_AS(run(), turbo::ExcludesError);

    args = {"--no", "--opt2"};
    CHECK_THROWS_AS(run(), turbo::ExcludesError);
}

TEST_CASE_FIXTURE(TApp, "NeedsMultiFlags [app]") {
    turbo::Option *opt1 = app.add_flag("--opt1");
    turbo::Option *opt2 = app.add_flag("--opt2");
    turbo::Option *opt3 = app.add_flag("--opt3");
    app.add_flag("--optall")->needs(opt1, opt2, opt3);  // NOLINT(readability-suspicious-call-argument)

    run();

    args = {"--opt1"};
    run();

    args = {"--opt2"};
    run();

    args = {"--optall"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--optall", "--opt1"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--optall", "--opt2", "--opt1"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--optall", "--opt1", "--opt2", "--opt3"};
    run();
}

TEST_CASE_FIXTURE(TApp, "NeedsMixedFlags [app]") {
    turbo::Option *opt1 = app.add_flag("--opt1");
    app.add_flag("--opt2");
    app.add_flag("--opt3");
    app.add_flag("--optall")->needs(opt1, "--opt2", "--opt3");

    run();

    args = {"--opt1"};
    run();

    args = {"--opt2"};
    run();

    args = {"--optall"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--optall", "--opt1"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--optall", "--opt2", "--opt1"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--optall", "--opt1", "--opt2", "--opt3"};
    run();
}

TEST_CASE_FIXTURE(TApp, "NeedsChainedFlags [app]") {
    turbo::Option *opt1 = app.add_flag("--opt1");
    turbo::Option *opt2 = app.add_flag("--opt2")->needs(opt1);
    app.add_flag("--opt3")->needs(opt2);

    run();

    args = {"--opt1"};
    run();

    args = {"--opt2"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--opt3"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--opt3", "--opt2"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--opt3", "--opt1"};
    CHECK_THROWS_AS(run(), turbo::RequiresError);

    args = {"--opt2", "--opt1"};
    run();

    args = {"--opt1", "--opt2", "--opt3"};
    run();
}

TEST_CASE_FIXTURE(TApp, "Env [app]") {

    put_env("FLAGS_TEST_ENV_TMP", "2");

    int val{1};
    turbo::Option *vopt = app.add_option("--tmp", val)->envname("FLAGS_TEST_ENV_TMP");

    run();

    CHECK_EQ(val , 2);
    CHECK_EQ(vopt->count() , 1u);

    vopt->required();
    run();

    unset_env("FLAGS_TEST_ENV_TMP");
    CHECK_THROWS_AS(run(), turbo::RequiredError);
}

// curiously check if an environmental only option works
TEST_CASE_FIXTURE(TApp, "EnvOnly [app]") {

    put_env("FLAGS_TEST_ENV_TMP", "2");

    int val{1};
    turbo::Option *vopt = app.add_option("", val)->envname("FLAGS_TEST_ENV_TMP");

    run();

    CHECK_EQ(val , 2);
    CHECK_EQ(vopt->count() , 1u);

    vopt->required();
    run();

    unset_env("FLAGS_TEST_ENV_TMP");
    CHECK_THROWS_AS(run(), turbo::RequiredError);
}

TEST_CASE_FIXTURE(TApp, "RangeInt [app]") {
    int x{0};
    app.add_option("--one", x)->check(turbo::Range(3, 6));

    args = {"--one=1"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--one=7"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--one=3"};
    run();

    args = {"--one=5"};
    run();

    args = {"--one=6"};
    run();
}

TEST_CASE_FIXTURE(TApp, "RangeDouble [app]") {

    double x{0.0};
    /// Note that this must be a double in Range, too
    app.add_option("--one", x)->check(turbo::Range(3.0, 6.0));

    args = {"--one=1"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--one=7"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--one=3"};
    run();

    args = {"--one=5"};
    run();

    args = {"--one=6"};
    run();
}

TEST_CASE_FIXTURE(TApp, "NonNegative [app]") {

    std::string res;
    /// Note that this must be a double in Range, too
    app.add_option("--one", res)->check(turbo::NonNegativeNumber);

    args = {"--one=crazy"};
    try {
        // this should throw
        run();
        CHECK(false);
    } catch (const turbo::ValidationError &e) {
        std::string emess = e.what();
        CHECK(emess.size() < 70U);
    }
}

TEST_CASE_FIXTURE(TApp, "typeCheck [app]") {

    /// Note that this must be a double in Range, too
    app.add_option("--one")->check(turbo::TypeValidator<unsigned int>());

    args = {"--one=1"};
    CHECK_NOTHROW(run());

    args = {"--one=-7"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--one=error"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--one=4.568"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);
}

TEST_CASE_FIXTURE(TApp, "NeedsTrue [app]") {
    std::string str;
    app.add_option("-s,--string", str);
    app.add_flag("--opt1")->check([&](const std::string &) {
        return (str != "val_with_opt1") ? std::string("--opt1 requires --string val_with_opt1") : std::string{};
    });

    run();

    args = {"--opt1"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--string", "val"};
    run();

    args = {"--string", "val", "--opt1"};
    CHECK_THROWS_AS(run(), turbo::ValidationError);

    args = {"--string", "val_with_opt1", "--opt1"};
    run();

    args = {"--opt1", "--string", "val_with_opt1"};  // order is not revelant
    run();
}

// Check to make sure programmatic access to left over is available
TEST_CASE_FIXTURE(TApp, "AllowExtras [app]") {

    app.allow_extras();

    bool val{true};
    app.add_flag("-f", val);

    args = {"-x", "-f"};

    REQUIRE_NOTHROW(run());
    CHECK(val);
    CHECK_EQ(std::vector<std::string>({"-x"}) , app.remaining());
}

TEST_CASE_FIXTURE(TApp, "AllowExtrasOrder [app]") {

    app.allow_extras();

    args = {"-x", "-f"};
    REQUIRE_NOTHROW(run());
    CHECK_EQ(std::vector<std::string>({"-x", "-f"}) , app.remaining());

    std::vector<std::string> left_over = app.remaining();
    app.parse(left_over);
    CHECK_EQ(std::vector<std::string>({"-f", "-x"}) , app.remaining());
    CHECK_EQ(left_over , app.remaining_for_passthrough());
}

TEST_CASE_FIXTURE(TApp, "AllowExtrasCascade [app]") {

    app.allow_extras();

    args = {"-x", "45", "-f", "27"};
    REQUIRE_NOTHROW(run());
    CHECK_EQ(std::vector<std::string>({"-x", "45", "-f", "27"}) , app.remaining());

    std::vector<std::string> left_over = app.remaining_for_passthrough();

    turbo::App capp{"cascade_program"};
    int v1 = 0;
    int v2 = 0;
    capp.add_option("-x", v1);
    capp.add_option("-f", v2);

    capp.parse(left_over);
    CHECK_EQ(45 , v1);
    CHECK_EQ(27 , v2);
}
// makes sure the error throws on the rValue version of the parse
TEST_CASE_FIXTURE(TApp, "ExtrasErrorRvalueParse [app]") {

    args = {"-x", "45", "-f", "27"};
    CHECK_THROWS_AS(app.parse(std::vector<std::string>({"-x", "45", "-f", "27"})), turbo::ExtrasError);
}

TEST_CASE_FIXTURE(TApp, "AllowExtrasCascadeDirect [app]") {

    app.allow_extras();

    args = {"-x", "45", "-f", "27"};
    REQUIRE_NOTHROW(run());
    CHECK_EQ(std::vector<std::string>({"-x", "45", "-f", "27"}) , app.remaining());

    turbo::App capp{"cascade_program"};
    int v1{0};
    int v2{0};
    capp.add_option("-x", v1);
    capp.add_option("-f", v2);

    capp.parse(app.remaining_for_passthrough());
    CHECK_EQ(45 , v1);
    CHECK_EQ(27 , v2);
}

TEST_CASE_FIXTURE(TApp, "AllowExtrasArgModify [app]") {

    int v1{0};
    int v2{0};
    app.allow_extras();
    app.add_option("-f", v2);
    args = {"27", "-f", "45", "-x"};
    app.parse(args);
    CHECK_EQ(std::vector<std::string>({"45", "-x"}) , args);

    turbo::App capp{"cascade_program"};

    capp.add_option("-x", v1);

    capp.parse(args);
    CHECK_EQ(45 , v1);
    CHECK_EQ(27 , v2);
}

// Test horrible error
TEST_CASE_FIXTURE(TApp, "CheckShortFail [app]") {
    args = {"--two"};

    CHECK_THROWS_AS(turbo::detail::AppFriend::parse_arg(&app, args, turbo::detail::Classifier::SHORT),
                    turbo::HorribleError);
}

// Test horrible error
TEST_CASE_FIXTURE(TApp, "CheckLongFail [app]") {
    args = {"-t"};

    CHECK_THROWS_AS(turbo::detail::AppFriend::parse_arg(&app, args, turbo::detail::Classifier::LONG),
                    turbo::HorribleError);
}

// Test horrible error
TEST_CASE_FIXTURE(TApp, "CheckWindowsFail [app]") {
    args = {"-t"};

    CHECK_THROWS_AS(turbo::detail::AppFriend::parse_arg(&app, args, turbo::detail::Classifier::WINDOWS_STYLE),
                    turbo::HorribleError);
}

// Test horrible error
TEST_CASE_FIXTURE(TApp, "CheckOtherFail [app]") {
    args = {"-t"};

    CHECK_THROWS_AS(turbo::detail::AppFriend::parse_arg(&app, args, turbo::detail::Classifier::NONE),
                    turbo::HorribleError);
}

// Test horrible error
TEST_CASE_FIXTURE(TApp, "CheckSubcomFail [app]") {
    args = {"subcom"};

    CHECK_THROWS_AS(turbo::detail::AppFriend::parse_subcommand(&app, args), turbo::HorribleError);
}

TEST_CASE_FIXTURE(TApp, "FallthroughParentFail [app]") {
    CHECK_THROWS_AS(turbo::detail::AppFriend::get_fallthrough_parent(&app), turbo::HorribleError);
}

TEST_CASE_FIXTURE(TApp, "FallthroughParents [app]") {
    auto *sub = app.add_subcommand("test");
    CHECK_EQ(&app , turbo::detail::AppFriend::get_fallthrough_parent(sub));

    auto *ssub = sub->add_subcommand("sub2");
    CHECK_EQ(sub , turbo::detail::AppFriend::get_fallthrough_parent(ssub));

    auto *og1 = app.add_option_group("g1");
    auto *og2 = og1->add_option_group("g2");
    auto *og3 = og2->add_option_group("g3");
    CHECK_EQ(&app , turbo::detail::AppFriend::get_fallthrough_parent(og3));

    auto *ogb1 = sub->add_option_group("g1");
    auto *ogb2 = ogb1->add_option_group("g2");
    auto *ogb3 = ogb2->add_option_group("g3");
    CHECK_EQ(sub , turbo::detail::AppFriend::get_fallthrough_parent(ogb3));

    ogb2->name("groupb");
    CHECK_EQ(ogb2 , turbo::detail::AppFriend::get_fallthrough_parent(ogb3));
}

TEST_CASE_FIXTURE(TApp, "OptionWithDefaults [app]") {
    int someint{2};
    app.add_option("-a", someint)->capture_default_str();

    args = {"-a1", "-a2"};

    CHECK_THROWS_AS(run(), turbo::ArgumentMismatch);
}

// Added to test ->transform
TEST_CASE_FIXTURE(TApp, "OrderedModifyingTransforms [app]") {
    std::vector<std::string> val;
    auto *m = app.add_option("-m", val);
    m->transform([](std::string x) { return x + "1"; });
    m->transform([](std::string x) { return x + "2"; });

    args = {"-mone", "-mtwo"};

    run();

    CHECK_EQ(std::vector<std::string>({"one21", "two21"}) , val);
}

TEST_CASE_FIXTURE(TApp, "ThrowingTransform [app]") {
    std::string val;
    auto *m = app.add_option("-m,--mess", val);
    m->transform([](std::string) -> std::string { throw turbo::ValidationError("My Message"); });

    REQUIRE_NOTHROW(run());

    args = {"-mone"};

    REQUIRE_THROWS_AS(run(), turbo::ValidationError);

    try {
        run();
    } catch (const turbo::ValidationError &e) {
        CHECK_EQ(std::string("--mess: My Message") , e.what());
    }
}

// This was added to make running a simple function on each item easier
TEST_CASE_FIXTURE(TApp, "EachItem [app]") {

    std::vector<std::string> results;
    std::vector<std::string> dummy;

    auto *opt = app.add_option("--vec", dummy);

    opt->each([&results](std::string item) { results.push_back(item); });

    args = {"--vec", "one", "two", "three"};

    run();

    CHECK_EQ(dummy , results);
}

// #128
TEST_CASE_FIXTURE(TApp, "RepeatingMultiArgumentOptions [app]") {
    std::vector<std::string> entries;
    app.add_option("--entry", entries, "set a key and value")->type_name("KEY VALUE")->type_size(-2);

    args = {"--entry", "key1", "value1", "--entry", "key2", "value2"};
    REQUIRE_NOTHROW(run());
    CHECK_EQ(std::vector<std::string>({"key1", "value1", "key2", "value2"}) , entries);

    args.pop_back();
    REQUIRE_THROWS_AS(run(), turbo::ArgumentMismatch);
}

// #122
TEST_CASE_FIXTURE(TApp, "EmptyOptionEach [app]") {
    std::string q;
    app.add_option("--each")->each([&q](std::string s) { q = s; });

    args = {"--each", "that"};
    run();

    CHECK_EQ("that" , q);
}

// #122
TEST_CASE_FIXTURE(TApp, "EmptyOptionFail [app]") {
    std::string q;
    app.add_option("--each");

    args = {"--each", "that"};
    run();
}

TEST_CASE_FIXTURE(TApp, "BeforeRequirements [app]") {
    app.add_flag_function("-a", [](std::int64_t) { throw turbo::Success(); });
    app.add_flag_function("-b", [](std::int64_t) { throw turbo::CallForHelp(); });

    args = {"extra"};
    CHECK_THROWS_AS(run(), turbo::ExtrasError);

    args = {"-a", "extra"};
    CHECK_THROWS_AS(run(), turbo::Success);

    args = {"-b", "extra"};
    CHECK_THROWS_AS(run(), turbo::CallForHelp);

    // These run in definition order.
    args = {"-a", "-b", "extra"};
    CHECK_THROWS_AS(run(), turbo::Success);

    // Currently, the original order is not preserved when calling callbacks
    // args = {"-b", "-a", "extra"};
    // CHECK_THROWS_AS (run(), turbo::CallForHelp);
}

// #209
TEST_CASE_FIXTURE(TApp, "CustomUserSepParse [app]") {

    std::vector<int> vals{1, 2, 3};
    args = {"--idx", "1,2,3"};
    auto *opt = app.add_option("--idx", vals)->delimiter(',');
    run();
    CHECK_EQ(std::vector<int>({1, 2, 3}) , vals);
    std::vector<int> vals2;
    // check that the results vector gets the results in the same way
    opt->results(vals2);
    CHECK_EQ(vals , vals2);

    app.remove_option(opt);

    app.add_option("--idx", vals)->delimiter(',')->capture_default_str();
    run();
    CHECK_EQ(std::vector<int>({1, 2, 3}) , vals);
}

// #209
TEST_CASE_FIXTURE(TApp, "DefaultUserSepParse [app]") {

    std::vector<std::string> vals;
    args = {"--idx", "1 2 3", "4 5 6"};
    auto *opt = app.add_option("--idx", vals, "");
    run();
    CHECK_EQ(std::vector<std::string>({"1 2 3", "4 5 6"}) , vals);
    opt->delimiter(',');
    run();
    CHECK_EQ(std::vector<std::string>({"1 2 3", "4 5 6"}) , vals);
}

// #209
TEST_CASE_FIXTURE(TApp, "BadUserSepParse [app]") {

    std::vector<int> vals;
    app.add_option("--idx", vals);

    args = {"--idx", "1,2,3"};

    CHECK_THROWS_AS(run(), turbo::ConversionError);
}

// #209
TEST_CASE_FIXTURE(TApp, "CustomUserSepParse2 [app]") {

    std::vector<int> vals{1, 2, 3};
    args = {"--idx", "1,2,"};
    auto *opt = app.add_option("--idx", vals)->delimiter(',');
    run();
    CHECK_EQ(std::vector<int>({1, 2}) , vals);

    app.remove_option(opt);

    app.add_option("--idx", vals, "")->delimiter(',')->capture_default_str();
    run();
    CHECK_EQ(std::vector<int>({1, 2}) , vals);
}

TEST_CASE_FIXTURE(TApp, "CustomUserSepParseFunction [app]") {

    std::vector<int> vals{1, 2, 3};
    args = {"--idx", "1,2,3"};
    app.add_option_function<std::vector<int>>("--idx", [&vals](std::vector<int> v) { vals = std::move(v); })
            ->delimiter(',');
    run();
    CHECK_EQ(std::vector<int>({1, 2, 3}) , vals);
}

// delimiter removal
TEST_CASE_FIXTURE(TApp, "CustomUserSepParseToggle [app]") {

    std::vector<std::string> vals;
    args = {"--idx", "1,2,3"};
    auto *opt = app.add_option("--idx", vals)->delimiter(',');
    run();
    CHECK_EQ(std::vector<std::string>({"1", "2", "3"}) , vals);
    opt->delimiter('\0');
    run();
    CHECK_EQ(std::vector<std::string>({"1,2,3"}) , vals);
    opt->delimiter(',');
    run();
    CHECK_EQ(std::vector<std::string>({"1", "2", "3"}) , vals);
}

// #209
TEST_CASE_FIXTURE(TApp, "CustomUserSepParse3 [app]") {

    std::vector<int> vals = {1, 2, 3};
    args = {"--idx",
            "1",
            ","
            "2"};
    auto *opt = app.add_option("--idx", vals)->delimiter(',');
    run();
    CHECK_EQ(std::vector<int>({1, 2}) , vals);
    app.remove_option(opt);

    app.add_option("--idx", vals)->delimiter(',');
    run();
    CHECK_EQ(std::vector<int>({1, 2}) , vals);
}

// #209
TEST_CASE_FIXTURE(TApp, "CustomUserSepParse4 [app]") {

    std::vector<int> vals;
    args = {"--idx", "1,    2"};
    auto *opt = app.add_option("--idx", vals)->delimiter(',')->capture_default_str();
    run();
    CHECK_EQ(std::vector<int>({1, 2}) , vals);

    app.remove_option(opt);

    app.add_option("--idx", vals)->delimiter(',');
    run();
    CHECK_EQ(std::vector<int>({1, 2}) , vals);
}

// #218
TEST_CASE_FIXTURE(TApp, "CustomUserSepParse5 [app]") {

    std::vector<std::string> bar;
    args = {"this", "is", "a", "test"};
    auto *opt = app.add_option("bar", bar, "bar");
    run();
    CHECK_EQ(std::vector<std::string>({"this", "is", "a", "test"}) , bar);

    app.remove_option(opt);
    args = {"this", "is", "a", "test"};
    app.add_option("bar", bar, "bar")->capture_default_str();
    run();
    CHECK_EQ(std::vector<std::string>({"this", "is", "a", "test"}) , bar);
}

// #218
TEST_CASE_FIXTURE(TApp, "logFormSingleDash [app]") {
    bool verbose{false};
    bool veryverbose{false};
    bool veryveryverbose{false};
    app.name("testargs");
    app.allow_extras();
    args = {"-v", "-vv", "-vvv"};
    app.final_callback([&]() {
        auto rem = app.remaining();
        for (auto &arg: rem) {
            if (arg == "-v") {
                verbose = true;
            }
            if (arg == "-vv") {
                veryverbose = true;
            }
            if (arg == "-vvv") {
                veryveryverbose = true;
            }
        }
    });
    run();
    CHECK_EQ(app.remaining().size() , 3U);
    CHECK(verbose);
    CHECK(veryverbose);
    CHECK(veryveryverbose);
}
