cmake_minimum_required(VERSION 3.2)
project(hercules)

option(BUILD_TESTING "Build the testing tree." OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
SET(CMAKE_LOCAL "${PROJECT_SOURCE_DIR}/cmake")
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_LOCAL}/)
SET(CMAKE_INSTALL_RPATH ".")

IF (ENABLE_RUNTIME_STRING_INVARIANT_CHECK)
    ADD_DEFINITIONS(-DHERCULES_RUNTIME_STRING_UNICODE_ENABLE_INVARIANT_CHECK)
ENDIF ()

ADD_DEFINITIONS(-DHVM_RUNTIME_ENABLE_STRINGREF)
# third-part
FIND_PACKAGE(PCRE REQUIRED)
INCLUDE_DIRECTORIES(${PCRE_INCLUDE_DIR})
# jemalloc
FIND_PACKAGE(Jemalloc)
IF (JEMALLOC_INCLUDE_DIR)
    INCLUDE_DIRECTORIES(${JEMALLOC_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_JEMALLOC)
ENDIF ()

IF (PCRE_FOUND)
    ADD_DEFINITIONS(-DHVM_ENABLE_PCRE_REGEX)
ENDIF ()

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(3rdparty/rapidjson/include)
INCLUDE_DIRECTORIES(3rdparty/filesystem/include)
INCLUDE_DIRECTORIES(3rdparty/alloc/include/brt)

find_path(RAPIDJSON_INCLUDE_DIR eabase/version.h)


IF (DEFINED ENV{CFLAGS})
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} $ENV{CFLAGS}")
ENDIF ()
IF (DEFINED ENV{CXXFLAGS})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXXFLAGS}")
ENDIF ()
IF (DEFINED ENV{CPPFLAGS})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CPPFLAGS}")
ENDIF ()
IF (NOT CMAKE_BUILD_TYPE)
    MESSAGE(STATUS "No build type selected, default to ReleaseWithDebugInfo")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -DNDEBUG")
ENDIF ()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -ftls-model=global-dynamic")
MESSAGE(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

FILE(GLOB_RECURSE
        HERCULES_OTHER_SRCS
        RELATIVE
        ${CMAKE_SOURCE_DIR}
        "hercules/runtime/*.cc"
        "hercules/runtime/*.cpp"
        "hercules/runtime/*.h"
        "hercules/codegen/*.cc"
        "hercules/codegen/*.cpp"
        "hercules/codegen/*.h"
        "hercules/pipeline/*.cc"
        "hercules/pipeline/*.cpp"
        "hercules/pipeline/*.h"
        "hercules/core/*.cc"
        "hercules/ir/*.cc"
        "hercules/ir/*.cpp"
        "hercules/ir/*.h"
        "hercules/printer/*.cc"
        "hercules/printer/*.cpp"
        "hercules/printer/*.h"
        "hercules/server/*.cc"
        "hercules/server/*.cpp"
        "hercules/server/*.h"
        "hercules/c_api/*.cc"
        "hercules/c_api/*.cpp"
        "hercules/c_api/*.h")

SET(HERCULES_SRCS  ${HERCULES_OTHER_SRCS})


LIST(REMOVE_ITEM HERCULES_SRCS ${HERCULES_CUDA_SRCS})
LIST(REMOVE_ITEM MEM_ALLOC ${MEM_CUDA_ALLOC})

ADD_LIBRARY(hercules SHARED ${HERCULES_SRCS} ${MEM_ALLOC})

TARGET_LINK_LIBRARIES(hercules pthread)

TARGET_LINK_LIBRARIES(hercules dl)
IF (PCRE_FOUND)
    TARGET_LINK_LIBRARIES(hercules ${PCRE_LIBRARIES})
ENDIF ()

IF (JEMALLOC_LIBRARIES)
    TARGET_LINK_LIBRARIES(hercules ${JEMALLOC_LIBRARIES})
ENDIF ()


SET(USE_LIBBACKTRACE OFF)
IF (DEFINED ENV{USE_LIBBACKTRACE})
    MESSAGE(STATUS "USE LIB_BACKTRACE: $ENV{USE_LIBBACKTRACE}")
    IF (CMAKE_SYSTEM_NAME MATCHES "Darwin" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
        SET(USE_LIBBACKTRACE $ENV{USE_LIBBACKTRACE})
    ENDIF ()
ENDIF ()

IF (USE_LIBBACKTRACE)
    MESSAGE("Building with libbacktrace...")
    include(cmake/Libbacktrace.cmake)
    TARGET_LINK_LIBRARIES(hercules libbacktrace)
    target_include_directories(hercules PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include)
    target_compile_definitions(hercules PUBLIC HVM_WITH_LIBBACKTRACE)
ENDIF ()

IF (BUILD_TESTING)
    MESSAGE(STATUS "BUILD_TESTING: ON")
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
ENDIF ()

SET(BUILD_BENCHMARK ON)
IF (DEFINED ENV{BUILD_BENCHMARK})
    MESSAGE(STATUS "USE ENV BUILD_BENCHMARK: $ENV{BUILD_BENCHMARK}")
    SET(BUILD_BENCHMARK $ENV{BUILD_BENCHMARK})
ENDIF ()


INSTALL(DIRECTORY hercules DESTINATION include FILES_MATCHING
        PATTERN "*.h"
    PATTERN "*.hpp")


INSTALL(TARGETS hercules
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

