#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
carbin_cc_binary(
        NAME config_app
        SOURCES config_app.cc
        COPTS ${USER_CXX_FLAGS}
)
add_test(NAME config_app1 COMMAND config_app -p)
set_property(TEST config_app1 PROPERTY PASS_REGULAR_EXPRESSION "file=")

add_test(NAME config_app2 COMMAND config_app -p -f /)
set_property(TEST config_app2 PROPERTY PASS_REGULAR_EXPRESSION "file=\"/\"")

add_test(NAME config_app3 COMMAND config_app -f "" -p)
set_property(TEST config_app3 PROPERTY PASS_REGULAR_EXPRESSION "file=\"\"")

add_test(NAME config_app4 COMMAND config_app -f "/" -p)
set_property(TEST config_app4 PROPERTY PASS_REGULAR_EXPRESSION "file=\"/\"")

carbin_cc_binary(
        NAME positional_arity
        SOURCES positional_arity.cc
        COPTS ${USER_CXX_FLAGS}
)

carbin_cc_binary(
        NAME positional_vector
        SOURCES positional_vector.cc
        COPTS ${USER_CXX_FLAGS}
)