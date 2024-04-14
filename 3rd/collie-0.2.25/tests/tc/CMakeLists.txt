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

set(source_files test.cc
                 arithmetic_policy.cc
                 boolean.cc
                 bounded_type.cc
                 compact_optional.cc
                 constrained_type.cc
                 constant_parser.cc
                 deferred_construction.cc
                 downcast.cc
                 flag.cc
                 flag_set.cc
                 floating_point.cc
                 index.cc
                 integer.cc
                 narrow_cast.cc
                 optional.cc
                 optional_ref.cc
                 output_parameter.cc
                 reference.cc
                 strong_typedef.cc
                 tagged_union.cc
                 variant.cc
                 visitor.cc)

carbin_cc_test(
        NAME type_safe_test
        SOURCES ${source_files}
        COPTS ${USER_CXX_FLAGS}
)

carbin_cc_test(
        NAME type_safe_std_test
        SOURCES std_tc.cc
        COPTS ${USER_CXX_FLAGS}
)