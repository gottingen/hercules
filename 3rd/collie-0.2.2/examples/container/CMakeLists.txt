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
        NAME span_basic
        SOURCES span_basic.cc
        DEPS Threads::Threads
        COPTS ${USER_CXX_FLAGS}
)

carbin_cc_binary(
        NAME span_advance
        SOURCES span_advance.cc
        DEPS Threads::Threads
        COPTS ${USER_CXX_FLAGS}
)

carbin_cc_binary(
        NAME inlined_vector
        SOURCES inlined_vector.cc
        DEPS Threads::Threads
        COPTS ${USER_CXX_FLAGS}
)
