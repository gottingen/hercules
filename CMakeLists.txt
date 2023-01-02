cmake_minimum_required(VERSION 3.15)

# TODO modify this for you project
project(hercules CXX C ASM)

# Compiler id for Apple Clang is now AppleClang.
cmake_policy(SET CMP0025 NEW)

# if command can use IN_LIST
cmake_policy(SET CMP0057 NEW)

# Project version variables are the empty string if version is unspecified
cmake_policy(SET CMP0048 NEW)

set(CMAKE_MACOSX_RPATH 1)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/copts)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/recipes)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/recipes)

include(carbin_print)

# TODO modyify this for "-std=c++[x]"
set(CMAKE_CXX_STANDARD 17)

execute_process(
        COMMAND bash -c "${PROJECT_SOURCE_DIR}/tools/get_project_version.sh ${PROJECT_SOURCE_DIR} | tr -d '\n'"
        OUTPUT_VARIABLE PROJECT_REVISION
)

set(PROJECT_DESCRIPTION "carbin c++ lib")
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

option(CARBIN_STATUS_PRINT "cmake toolchain print" ON)
option(CARBIN_STATUS_DEBUG "cmake toolchain debug info" ON)

option(ENABLE_TESTING "enable unit test" OFF)
option(CARBIN_PACKAGE_GEN "enable package gen" OFF)
option(ENABLE_BENCHMARK "enable benchmark" OFF)
option(ENABLE_EXAMPLE "enable benchmark" OFF)
option(ENABLE_SUMMARY "enable summary output" ON)
# it is too many
option(ENABLE_SUMMARY_CXX_FLAG "enable cxx flags" OFF)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(PACKAGE_INSTALL_PREFIX "/usr/local")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
include(carbin_module)
include(CMakePackageConfigHelpers)

if (NOT DEV_MODE AND ${PROJECT_NAME} MATCHES "carbin")
    carbin_error("project name must be set to your project name \
                  or define -DDEV_MODE=true set to develop modeeg: \n\
                   cmake .. -DDEV_MODE=true")
endif ()


#########################
# default package

set(CARBIN_SYS_DYLINK)
if (APPLE)
    find_library(CoreFoundation CoreFoundation)
    list(APPEND CARBIN_SYS_DYLINK ${CoreFoundation} pthread)
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND CARBIN_SYS_DYLINK rt dl pthread)
endif ()


include(require_melon)
include(require_openssl)
include(require_gflags)
find_package(TensorFlow)
include(FindProtobuf)

set(DYNAMIC_LIB
        ${CMAKE_THREAD_LIBS_INIT}
        ${GFLAGS_LIBRARY}
        ${PROTOBUF_LIBRARIES}
        ${LEVELDB_LIB}
        ${OPENSSL_CRYPTO_LIBRARY}
        ${OPENSSL_SSL_LIBRARY}
        ${MELON_LIB}
        dl
        )

include(CompileProto)

set(PROTO_FILES
        hercules/proto/document.proto
        hercules/proto/plugin.proto
        hercules/proto/query.proto
        hercules/proto/recall_doc.proto
        hercules/proto/router.proto
        hercules/proto/search.proto
        hercules/proto/sort.proto
        hercules/proto/types.proto
        )
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/output/include/hercules)
set(PROTOC_FLAGS ${PROTOC_FLAGS} -I${PROTOBUF_INCLUDE_DIR} -I${CMAKE_SOURCE_DIR})
compile_proto(PROTO_HDRS PROTO_SRCS ${PROJECT_BINARY_DIR}
        ${PROJECT_BINARY_DIR}/output/include
        ${PROJECT_SOURCE_DIR}
        "${PROTO_FILES}")

carbin_cc_library(
        NAMESPACE hercules
        NAME proto
        SOURCES ${PROTO_SRCS}
        PUBLIC_LINKED_TARGETS ${CARBIN_SYS_DYLINK} ${DYNAMIC_LIB}
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
)


#TODO do add you own subdirs
add_subdirectory(hercules)
# TODO end

if (ENABLE_TESTING)
    set(CARBIN_RUN_TESTS true)
    enable_testing()
    add_subdirectory(test)
endif (ENABLE_TESTING)

################################################################################
# Installation of the library and all it's sub components. No need to edit this.
################################################################################
# Get the Default installation folders:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

# Layout. This works for all platforms:
#   * <prefix>/lib*/cmake/<PROJECT-NAME>
#   * <prefix>/lib*/
#   * <prefix>/include/
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Use:
#   * PROJECT_VERSION
write_basic_package_version_file(
        "${version_config}" COMPATIBILITY SameMajorVersion
)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
configure_package_config_file(
        "cmake/package/carbin_config.cmake.in"
        "${project_config}"
        INSTALL_DESTINATION "${config_install_dir}"
)

#Targets:
#  * <prefix>/lib/libbar.a
#  * <prefix>/lib/libbaz.a
#  * header location after install: <prefix>/include/foo/Bar.hpp
#  * headers can be included by C++ code `#include <foo/Bar.hpp>`


# Config
#   * <prefix>/lib/cmake/Foo/FooConfig.cmake
#   * <prefix>/lib/cmake/Foo/FooConfigVersion.cmake
install(
        FILES
        "${project_config}" "${version_config}"
        DESTINATION
        "${config_install_dir}"
)

# Config
#   * <prefix>/lib/cmake/Foo/FooTargets.cmake
install(
        EXPORT
        "${TARGETS_EXPORT_NAME}"
        NAMESPACE
        "${namespace}"
        DESTINATION
        "${config_install_dir}"
)


if (CARBIN_PACKAGE_GEN)
    include(carbin_package)
endif ()
if (ENABLE_SUMMARY)
    include(carbin_summary)
endif ()
