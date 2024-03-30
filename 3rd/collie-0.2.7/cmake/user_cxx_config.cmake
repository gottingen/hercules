#
# Copyright 2023 The Carbin Authors.
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

##############################################################
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-g -O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if(DEFINED ENV{CARBIN_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CARBIN_CXX_FLAGS}")
endif ()

################################
# follow CC flag we provide
# ${CARBIN_DEFAULT_COPTS}
# ${CARBIN_TEST_COPTS}
# ${CARBIN_ARCH_OPTION} for arch option, by default, we set enable and
# ${CARBIN_RANDOM_RANDEN_COPTS}
# set it to haswell arch
##############################################################################
set(CARBIN_CXX_OPTIONS ${CARBIN_DEFAULT_COPTS} ${CARBIN_ARCH_OPTION} ${CARBIN_RANDOM_RANDEN_COPTS})
###############################
#
# define you options here
# eg.
# list(APPEND CARBIN_CXX_OPTIONS "-fopenmp")
list(REMOVE_DUPLICATES CARBIN_CXX_OPTIONS)
carbin_print_list_label("CXX_OPTIONS:" CARBIN_CXX_OPTIONS)