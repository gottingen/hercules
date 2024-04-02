#
# Copyright 2023 The EA Authors.
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

include_directories(${PROJECT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)



################
# ARM SETTINGS #
################

OPTION(CROSS_COMPILE_ARM "cross compile for ARM targets" OFF)

# Note: to compile on ARM (or cross compile), you may need to add the following:
# -DTARGET_ARCH="armv8-a -mfpu=neon -mfloat-abi=softfp -target arm-linux-gnueabi"
set(TARGET_ARCH "native" CACHE STRING "Target architecture arguments")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    if (NOT WIN32 AND NOT ANDROID)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-parameter -Wextra -Wreorder -Wno-maybe-uninitialized")
        if (NOT CROSS_COMPILE_ARM)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC")
        endif()
    endif()

    if (ANDROID)
        # Nothing to do here, we assume the cmake Android NDK toolchain sets the
        # correct options for arm and neon.
    elseif (CROSS_COMPILE_ARM)
        # We're cross-compiling with clang++ on Azure Pipelines, this is all pretty specific and just for testing
        set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
        set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
        set(CMAKE_THREAD_LIBS_INIT)

        set(CMAKE_SYSTEM_PROCESSOR arm)
        set(CMAKE_C_COMPILER_TARGET arm-linux-gnueabi)
        set(CMAKE_CXX_COMPILER_TARGET arm-linux-gnueabi)

        set(ARM_ARCH_DIRECTORY "arm-linux-gnueabi" CACHE STRING "ARM arch header dir")
        set(ARM_GCC_VER "4.7.3" CACHE STRING "ARM GCC header dir")
        include_directories(/usr/${ARM_ARCH_DIRECTORY}/include/c++/${ARM_GCC_VER}/${ARM_ARCH_DIRECTORY}/)
        include_directories(/usr/${ARM_ARCH_DIRECTORY}/include/c++/${ARM_GCC_VER}/)
        include_directories(/usr/${ARM_ARCH_DIRECTORY}/include/)
        if(NOT CMAKE_CXX_FLAGS MATCHES "-march")
            message(STATUS "SETTING ARCH TO ${TARGET_ARCH}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${TARGET_ARCH}")
        endif()
        if(ARM_ARCH_DIRECTORY MATCHES "arm-linux-gnueabi")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -mfloat-abi=softfp -target arm-linux-gnueabi")
        else ()
            # delegating to gcc here
        endif()
        message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
        message(STATUS "CMAKE_CXX_LINK_EXECUTABLE: ${CMAKE_CXX_LINK_EXECUTABLE}")
    elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^ppc64"  OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=${TARGET_ARCH} -mtune=${TARGET_ARCH}")
    elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
        # Nothing specific
    elseif(NOT WIN32 AND NOT EMSCRIPTEN)
        if(NOT CMAKE_CXX_FLAGS MATCHES "-march" AND NOT CMAKE_CXX_FLAGS MATCHES "-arch" AND NOT CMAKE_OSX_ARCHITECTURES)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${TARGET_ARCH}")
        endif()
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4005 /wd4146 /wd4800")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang AND MSVC AND WIN32) # We are using clang-cl
    add_compile_options(/EHsc /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

set(SIMD_TESTS
    main.cc
    test_api.cc
    test_arch.cc
    test_basic_math.cc
    test_batch.cc
    test_batch_bool.cc
    test_batch_cast.cc
    test_batch_complex.cc
    test_batch_float.cc
    test_batch_int.cc
    test_bitwise_cast.cc
    test_batch_constant.cc
    test_batch_manip.cc
    test_complex_exponential.cc
    test_complex_hyperbolic.cc
    test_complex_power.cc
    test_complex_trigonometric.cc
    test_conversion.cc
    test_custom_default_arch.cc
    test_error_gamma.cc
    test_explicit_batch_instantiation.cc
    test_exponential.cc
    test_extract_pair.cc
    test_fp_manipulation.cc
    test_hyperbolic.cc
    test_load_store.cc
    test_memory.cc
    test_poly_evaluation.cc
    test_power.cc
    test_rounding.cc
    test_select.cc
    test_shuffle.cc
    test_sum.cc
    test_traits.cc
    test_trigonometric.cc
        test_simd_api.cc
    test_utils.hpp
)

if(NOT MSVC)
    list(APPEND SIMD_TESTS test_gnu_source.cc)
endif()

add_executable(test_simd ${SIMD_TESTS})

add_test(NAME test_simd COMMAND test_simd)

if (CROSS_COMPILE_ARM)
    add_custom_target(test COMMAND qemu-arm -L /usr/arm-linux-gnueabi/ test_simd DEPENDS test_simd)
else()
    add_custom_target(xtest COMMAND test_simd DEPENDS test_simd)
endif()

add_subdirectory(doc)
add_subdirectory(architectures)

if(EMSCRIPTEN)
    set_target_properties(test_simd PROPERTIES LINK_FLAGS "-s MODULARIZE=1 -s EXPORT_NAME=test_simd_wasm -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -lembind")
    target_compile_options(test_simd
        PUBLIC --std=c++17
        PUBLIC "SHELL: -msimd128"
        PUBLIC "SHELL: -msse2"
    )
endif()
