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

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/carbin_cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/carbin_cmake/package)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/carbin_cmake/recipes)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/carbin_cmake/arch)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/carbin_cmake/copts)
include(carbin_option)
include(carbin_config_cxx_opts)
include(carbin_arch)
include(carbin_print)
include(carbin_cc_library)
include(carbin_cc_test)
include(carbin_cc_binary)
include(carbin_cc_benchmark)
include(carbin_check)
include(carbin_variable)
include(carbin_include)
include(carbin_outof_source)
include(carbin_platform)
include(carbin_pkg_dump)

option(CARBIN_USE_SYSTEM_INCLUDES "" OFF)
if (VERBOSE_CMAKE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

if (CARBIN_USE_CXX11_ABI)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
elseif ()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif ()

if (CONDA_ENV_ENABLE)
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
endif ()

if (CARBIN_DEPS_ENABLE)
    list(APPEND CMAKE_PREFIX_PATH ${CARBIN_DEPS_PREFIX})
    include_directories(${CARBIN_DEPS_PREFIX}/include)
    link_directories(${CARBIN_DEPS_PREFIX}/lib)
endif ()

if (CARBIN_INSTALL_LIB)
    set(CMAKE_INSTALL_LIBDIR lib)
endif ()

if (CARBIN_USE_SYSTEM_INCLUDES)
    set(CARBIN_INTERNAL_INCLUDE_WARNING_GUARD SYSTEM)
else ()
    set(CARBIN_INTERNAL_INCLUDE_WARNING_GUARD "")
endif ()


set(CARBIN_SYSTEM_DYLINK)
if (APPLE)
    find_library(CoreFoundation CoreFoundation)
    list(APPEND CARBIN_SYSTEM_DYLINK ${CoreFoundation} pthread)
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND CARBIN_SYSTEM_DYLINK rt dl pthread)
endif ()

CARBIN_ENSURE_OUT_OF_SOURCE_BUILD("must out of source dir")

#if (NOT DEV_MODE AND ${PROJECT_VERSION} MATCHES "0.0.0")
#    carbin_error("PROJECT_VERSION must be set in file project_profile or set -DDEV_MODE=true for develop debug")
#endif()


