set(CPM_DOWNLOAD_VERSION 0.32.3)
set(CPM_DOWNLOAD_LOCATION "${PROJECT_SOURCE_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME collie
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/collie-0.2.7"
        OPTIONS "CARBIN_ENABLE_INSTALL OFF"
        "CARBIN_BUILD_TEST OFF"
        "CARBIN_BUILD_EXAMPLES OFF"
        "CARBIN_BUILD_BENCHMARKS OFF"
        )
include_directories(${collie_SOURCE_DIR})

CPMAddPackage(
        NAME zlibng
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/zlibng-2.0.5"
        EXCLUDE_FROM_ALL YES
        OPTIONS "HAVE_OFF64_T ON"
        "ZLIB_COMPAT ON"
        "ZLIB_ENABLE_TESTS OFF"
        "CMAKE_POSITION_INDEPENDENT_CODE ON")
if (zlibng_ADDED)
    set_target_properties(zlib PROPERTIES EXCLUDE_FROM_ALL ON)
endif ()

CPMAddPackage(
        NAME xz
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/xz-5.2.5"
        EXCLUDE_FROM_ALL YES
        OPTIONS "BUILD_SHARED_LIBS OFF"
        "CMAKE_POSITION_INDEPENDENT_CODE ON")
if (xz_ADDED)
    set_target_properties(xz PROPERTIES EXCLUDE_FROM_ALL ON)
    set_target_properties(xzdec PROPERTIES EXCLUDE_FROM_ALL ON)
endif ()

CPMAddPackage(
        NAME bz2
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/bz2-1.0.8")
if (bz2_ADDED)
    add_library(bz2 STATIC
            "${bz2_SOURCE_DIR}/blocksort.c"
            "${bz2_SOURCE_DIR}/huffman.c"
            "${bz2_SOURCE_DIR}/crctable.c"
            "${bz2_SOURCE_DIR}/randtable.c"
            "${bz2_SOURCE_DIR}/compress.c"
            "${bz2_SOURCE_DIR}/decompress.c"
            "${bz2_SOURCE_DIR}/bzlib.c"
            "${bz2_SOURCE_DIR}/libbz2.def")
    set_target_properties(bz2 PROPERTIES
            COMPILE_FLAGS "-D_FILE_OFFSET_BITS=64"
            POSITION_INDEPENDENT_CODE ON)
endif ()

CPMAddPackage(
        NAME bdwgc
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/bdwgc-8.0.5"
        EXCLUDE_FROM_ALL YES
        OPTIONS "CMAKE_POSITION_INDEPENDENT_CODE ON"
        "BUILD_SHARED_LIBS OFF"
        "enable_threads ON"
        "enable_large_config ON"
        "enable_thread_local_alloc ON"
        "enable_handle_fork ON")
if (bdwgc_ADDED)
    set_target_properties(cord PROPERTIES EXCLUDE_FROM_ALL ON)
endif ()

CPMAddPackage(
        NAME openmp
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/openmp"
        EXCLUDE_FROM_ALL YES
        OPTIONS "CMAKE_BUILD_TYPE Release"
        "OPENMP_ENABLE_LIBOMPTARGET OFF"
        "OPENMP_STANDALONE_BUILD ON")

CPMAddPackage(
        NAME backtrace
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/backtrace"
)
if (backtrace_ADDED)
    set(backtrace_SOURCES
            "${backtrace_SOURCE_DIR}/atomic.c"
            "${backtrace_SOURCE_DIR}/backtrace.c"
            "${backtrace_SOURCE_DIR}/dwarf.c"
            "${backtrace_SOURCE_DIR}/fileline.c"
            "${backtrace_SOURCE_DIR}/mmapio.c"
            "${backtrace_SOURCE_DIR}/mmap.c"
            "${backtrace_SOURCE_DIR}/posix.c"
            "${backtrace_SOURCE_DIR}/print.c"
            "${backtrace_SOURCE_DIR}/simple.c"
            "${backtrace_SOURCE_DIR}/sort.c"
            "${backtrace_SOURCE_DIR}/state.c")

    # https://go.googlesource.com/gollvm/+/refs/heads/master/cmake/modules/LibbacktraceUtils.cmake
    set(BACKTRACE_SUPPORTED 1)
    set(BACKTRACE_ELF_SIZE 64)
    set(HAVE_GETIPINFO 1)
    set(BACKTRACE_USES_MALLOC 1)
    set(BACKTRACE_SUPPORTS_THREADS 1)
    set(BACKTRACE_SUPPORTS_DATA 1)
    set(HAVE_SYNC_FUNCTIONS 1)
    if (APPLE)
        set(HAVE_MACH_O_DYLD_H 1)
        list(APPEND backtrace_SOURCES "${backtrace_SOURCE_DIR}/macho.c")
    else ()
        set(HAVE_MACH_O_DYLD_H 0)
        list(APPEND backtrace_SOURCES "${backtrace_SOURCE_DIR}/elf.c")
    endif ()
    # Generate backtrace-supported.h based on the above.
    configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/backtrace-supported.h.in
            ${backtrace_SOURCE_DIR}/backtrace-supported.h)
    configure_file(
            ${CMAKE_SOURCE_DIR}/cmake/backtrace-config.h.in
            ${backtrace_SOURCE_DIR}/config.h)
    add_library(backtrace STATIC ${backtrace_SOURCES})
    target_include_directories(backtrace BEFORE PRIVATE "${backtrace_SOURCE_DIR}")
    set_target_properties(backtrace PROPERTIES
            COMPILE_FLAGS "-funwind-tables -D_GNU_SOURCE"
            POSITION_INDEPENDENT_CODE ON)
endif ()

CPMAddPackage(
        NAME re2
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/re2-2022-06-01"
        EXCLUDE_FROM_ALL YES
        OPTIONS "CMAKE_POSITION_INDEPENDENT_CODE ON"
        "BUILD_SHARED_LIBS OFF"
        "RE2_BUILD_TESTING OFF")

if (APPLE AND APPLE_ARM)
    enable_language(ASM)
    CPMAddPackage(
            NAME unwind
            SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/libunwind"
            OPTIONS "CMAKE_BUILD_TYPE Release"
            "LIBUNWIND_ENABLE_STATIC OFF"
            "LIBUNWIND_ENABLE_SHARED ON"
            "LIBUNWIND_INCLUDE_DOCS OFF")
endif ()
