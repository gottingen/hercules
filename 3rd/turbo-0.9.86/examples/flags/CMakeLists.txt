#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

carbin_cc_binary(
        NAME
        callback_passthrough
        SOURCES
        "callback_passthrough.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        config_app
        SOURCES
        "config_app.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        custom_parse
        SOURCES
        "custom_parse.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        digit_args
        SOURCES
        "digit_args.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        enum
        SOURCES
        "enum.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        enum_ostream
        SOURCES
        "enum_ostream.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        formatter
        SOURCES
        "formatter.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        groups
        SOURCES
        "groups.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        inter_argument_order
        SOURCES
        "inter_argument_order.cc"
        DEPS
        turbo::turbo
)


carbin_cc_binary(
        NAME
        modhelp
        SOURCES
        "modhelp.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        nested
        SOURCES
        "nested.cc"
        DEPS
        turbo::turbo
)


carbin_cc_binary(
        NAME
        option_groups
        SOURCES
        "option_groups.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        positional_arity
        SOURCES
        "positional_arity.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        positional_validation
        SOURCES
        "positional_validation.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        prefix_command
        SOURCES
        "prefix_command.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        ranges
        SOURCES
        "ranges.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        retired
        SOURCES
        "retired.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        shapes
        SOURCES
        "shapes.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        subcom_help
        SOURCES
        "subcom_help.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        subcom_partitioned
        SOURCES
        "subcom_partitioned.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        subcommands
        SOURCES
        "subcommands.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        test_exec
        SOURCES
        "test_exec.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        validators
        SOURCES
        "validators.cc"
        DEPS
        turbo::turbo
)

carbin_cc_binary(
        NAME
        subcommand_main
        SOURCES
        "subcommand_a.cc"
        "subcommand_main.cc"
        DEPS
        turbo::turbo
)