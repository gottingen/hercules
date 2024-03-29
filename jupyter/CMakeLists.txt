cmake_minimum_required(VERSION 3.14)
project(
  HerculesJupyter
  VERSION "0.1"
  HOMEPAGE_URL "https://github.com/gottingen/hercules"
  DESCRIPTION "Jupyter support for Hercules")

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

if(NOT HERCULES_PATH)
    set(HERCULES_PATH "$ENV{HOME}/.hercules")
endif()
message(STATUS "Found Hercules in ${HERCULES_PATH}")
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${HERCULES_PATH}/lib/hercules/" CACHE PATH "Use the existing Hercules installation" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -pedantic -fvisibility-inlines-hidden -Wno-return-type-c-linkage -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations"
  )
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-limit-debug-info")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(.)

find_package(LLVM REQUIRED CONFIG)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(CPM_DOWNLOAD_VERSION 0.32.3)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})
CPMAddPackage(
    NAME xtl
    GITHUB_REPOSITORY "xtensor-stack/xtl"
    VERSION 0.7.5
    GIT_TAG 0.7.5
    EXCLUDE_FROM_ALL YES
    OPTIONS "BUILD_TESTS OFF")
CPMAddPackage(
    NAME json
    GITHUB_REPOSITORY "nlohmann/json"
    VERSION 3.11.2)
CPMAddPackage(
    NAME xeus
    GITHUB_REPOSITORY "jupyter-xeus/xeus"
    VERSION 3.0.5
    GIT_TAG 3.0.5
    EXCLUDE_FROM_ALL YES
    PATCH_COMMAND git apply --reject --whitespace=fix ${CMAKE_SOURCE_DIR}/xeus.patch
    OPTIONS "BUILD_EXAMPLES OFF"
            "XEUS_BUILD_SHARED_LIBS OFF"
            "XEUS_STATIC_DEPENDENCIES ON"
            "CMAKE_POSITION_INDEPENDENT_CODE ON"
            "XEUS_DISABLE_ARCH_NATIVE ON"
            "XEUS_USE_DYNAMIC_UUID ${XEUS_USE_DYNAMIC_UUID}")
if (xeus_ADDED)
    install(TARGETS nlohmann_json EXPORT xeus-targets)
endif()

CPMAddPackage(
  NAME libzmq
  VERSION 4.3.4
  URL https://github.com/zeromq/libzmq/releases/download/v4.3.4/zeromq-4.3.4.tar.gz
  EXCLUDE_FROM_ALL YES
  OPTIONS "WITH_PERF_TOOL OFF"
          "ZMQ_BUILD_TESTS OFF"
          "ENABLE_CPACK OFF"
          "BUILD_SHARED ON"
          "WITH_LIBSODIUM OFF"
          "WITH_TLS OFF"
          "WITH_DOC OFF")
CPMAddPackage(
    NAME cppzmq
    URL https://github.com/zeromq/cppzmq/archive/refs/tags/v4.8.1.tar.gz
    VERSION 4.9.0
    EXCLUDE_FROM_ALL YES
    OPTIONS "CPPZMQ_BUILD_TESTS OFF")
CPMAddPackage(
    NAME xeus-zmq
    GITHUB_REPOSITORY "jupyter-xeus/xeus-zmq"
    VERSION 1.0.3
    GIT_TAG 1.0.3
    EXCLUDE_FROM_ALL YES
    PATCH_COMMAND patch -N -u CMakeLists.txt --ignore-whitespace -b ${CMAKE_SOURCE_DIR}/xeus.patch || true
    OPTIONS "XEUS_ZMQ_BUILD_TESTS OFF"
            "XEUS_ZMQ_BUILD_SHARED_LIBS OFF"
            "XEUS_ZMQ_STATIC_DEPENDENCIES ON"
            "CMAKE_POSITION_INDEPENDENT_CODE ON")

# Hercules Jupyter library
set(HERCULES_JUPYTER_FILES jupyter.h jupyter.cpp)
add_library(hercules_jupyter SHARED ${HERCULES_JUPYTER_FILES})
target_include_directories(hercules_jupyter PRIVATE "${HERCULES_PATH}/include" ${LLVM_INCLUDE_DIRS})
add_dependencies(hercules_jupyter xeus-static xeus-zmq-static nlohmann_json)
target_link_directories(hercules_jupyter PRIVATE "${HERCULES_PATH}/lib/hercules")
target_link_libraries(hercules_jupyter PRIVATE xeus-static xeus-zmq-static herculesc)

install(TARGETS hercules_jupyter DESTINATION .)
