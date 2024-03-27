cmake_minimum_required(VERSION 3.14)
project(
        Hercules
        VERSION "0.2.1"
        HOMEPAGE_URL "https://github.com/gottingen/hercules"
        DESCRIPTION "high-performance, extensible Python compiler")
set(HERCULES_JIT_PYTHON_VERSION "0.1.7")
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.h.in"
        "${PROJECT_SOURCE_DIR}/hercules/config/config.h")
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.py.in"
        "${PROJECT_SOURCE_DIR}/jit/hercules/version.py")

option(HERCULES_GPU "build Hercules GPU backend" OFF)

cmake_policy(SET CMP0028 NEW)
set(CMAKE_VERBOSE_MAKEFILE OFF)
#set_target_properties(xz PROPERTIES EXCLUDE_FROM_ALL ON)
#set_target_properties(xzdec PROPERTIES EXCLUDE_FROM_ALL ON)
link_directories(${CONDA_PREFIX}/lib)
include_directories(${CONDA_PREFIX}/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -pedantic -fvisibility-inlines-hidden -Wno-return-type-c-linkage -Wno-gnu-zero-variadic-macro-arguments -Wno-deprecated-declarations"
    )
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type")
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-limit-debug-info")
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(.)

set(APPLE_ARM OFF)
if (APPLE AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(APPLE_ARM ON)
endif ()

list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(LLVM 17 REQUIRED)
find_package(Clang 17 REQUIRED)
get_property(CLANG_LOCATION TARGET clang PROPERTY LOCATION)
message(STATUS "Using clang ${CLANG_LOCATION}")
include_directories(${CLANG_INCLUDE_DIRS})

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include(${CMAKE_SOURCE_DIR}/cmake/deps.cmake)

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib/hercules")
    set(STATIC_LIBCPP "")
else ()
    set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib/hercules")
    set(STATIC_LIBCPP "-static-libstdc++")
endif ()

add_executable(peg2cpp hercules/util/peg2cpp.cc)
target_include_directories(peg2cpp PRIVATE ${peglib_SOURCE_DIR})
target_link_libraries(peg2cpp PRIVATE Threads::Threads fmt)
add_custom_command(
        OUTPUT hercules_rules.cc
        COMMAND peg2cpp ${CMAKE_SOURCE_DIR}/hercules/parser/peg/grammar.peg
        hercules_rules.cc hercules
        DEPENDS peg2cpp hercules/parser/peg/grammar.peg)
add_custom_command(
        OUTPUT omp_rules.cc
        COMMAND peg2cpp ${CMAKE_SOURCE_DIR}/hercules/parser/peg/openmp.peg omp_rules.cc
        omp
        DEPENDS peg2cpp hercules/parser/peg/openmp.peg)

# Hercules Jupyter library
set(HERCULES_JUPYTER_FILES hercules/util/jupyter.h hercules/util/jupyter.cc)
add_library(hercules_jupyter SHARED ${HERCULES_JUPYTER_FILES})

# Hercules runtime library
set(HERCULESRT_FILES
        hercules/runtime/lib.h
        hercules/runtime/hs_str.h
        hercules/runtime/hs_str.cc
        hercules/runtime/lib.cc
        hercules/runtime/re.cc
        hercules/runtime/exc.cc
        hercules/runtime/gpu.cc
        hercules/runtime/libinfo.cc
)

add_library(herculesrt SHARED ${HERCULESRT_FILES})
add_dependencies(herculesrt zlibstatic gc backtrace bz2 liblzma re2)
if (APPLE AND APPLE_ARM)
    add_dependencies(herculesrt unwind_shared)
endif ()
target_include_directories(herculesrt PRIVATE ${backtrace_SOURCE_DIR}
        ${re2_SOURCE_DIR}
        "${gc_SOURCE_DIR}/include" runtime)
target_link_libraries(herculesrt PRIVATE fmt omp backtrace ${STATIC_LIBCPP}
        LLVMSupport)
if (APPLE)
    target_link_libraries(
            herculesrt
            PRIVATE -Wl,-force_load,$<TARGET_FILE:zlibstatic>
            -Wl,-force_load,$<TARGET_FILE:gc>
            -Wl,-force_load,$<TARGET_FILE:bz2>
            -Wl,-force_load,$<TARGET_FILE:liblzma>
            -Wl,-force_load,$<TARGET_FILE:re2>)
else ()
    target_link_libraries(
            herculesrt
            PRIVATE -Wl,--whole-archive $<TARGET_FILE:zlibstatic> $<TARGET_FILE:gc>
            $<TARGET_FILE:bz2> $<TARGET_FILE:liblzma> $<TARGET_FILE:re2>
            -Wl,--no-whole-archive)
endif ()
if (ASAN)
    target_compile_options(
            herculesrt PRIVATE "-fno-omit-frame-pointer" "-fsanitize=address"
            "-fsanitize-recover=address")
    target_link_libraries(
            herculesrt PRIVATE "-fno-omit-frame-pointer" "-fsanitize=address"
            "-fsanitize-recover=address")
endif ()
if (HERCULES_GPU)
    add_compile_definitions(HERCULES_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(herculesrt PRIVATE CUDA::cudart_static CUDA::cuda_driver)
endif ()
add_custom_command(
        TARGET herculesrt
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:omp>
        ${CMAKE_BINARY_DIR})
if (APPLE AND APPLE_ARM)
    add_custom_command(
            TARGET herculesrt
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:unwind_shared>
            ${CMAKE_BINARY_DIR})
endif ()

# Hercules compiler library
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
set(HERCULES_HPPFILES
        hercules/compiler/compiler.h
        hercules/compiler/debug_listener.h
        hercules/compiler/engine.h
        hercules/compiler/error.h
        hercules/compiler/jit.h
        hercules/compiler/memory_manager.h
        hercules/dsl/dsl.h
        hercules/dsl/plugins.h
        hercules/parser/ast.h
        hercules/parser/ast/expr.h
        hercules/parser/ast/stmt.h
        hercules/parser/ast/types.h
        hercules/parser/ast/types/type.h
        hercules/parser/ast/types/link.h
        hercules/parser/ast/types/class.h
        hercules/parser/ast/types/function.h
        hercules/parser/ast/types/union.h
        hercules/parser/ast/types/static.h
        hercules/parser/ast/types/traits.h
        hercules/parser/cache.h
        hercules/parser/common.h
        hercules/parser/ctx.h
        hercules/parser/peg/peg.h
        hercules/parser/peg/rules.h
        hercules/parser/visitors/doc/doc.h
        hercules/parser/visitors/format/format.h
        hercules/parser/visitors/simplify/simplify.h
        hercules/parser/visitors/simplify/ctx.h
        hercules/parser/visitors/translate/translate.h
        hercules/parser/visitors/translate/translate_ctx.h
        hercules/parser/visitors/typecheck/typecheck.h
        hercules/parser/visitors/typecheck/ctx.h
        hercules/parser/visitors/visitor.h
        hercules/hir/analyze/analysis.h
        hercules/hir/analyze/dataflow/capture.h
        hercules/hir/analyze/dataflow/cfg.h
        hercules/hir/analyze/dataflow/dominator.h
        hercules/hir/analyze/dataflow/reaching.h
        hercules/hir/analyze/module/global_vars.h
        hercules/hir/analyze/module/side_effect.h
        hercules/hir/attribute.h
        hercules/hir/base.h
        hercules/hir/const.h
        hercules/hir/dsl/codegen.h
        hercules/hir/dsl/nodes.h
        hercules/hir/flow.h
        hercules/hir/func.h
        hercules/hir/instr.h
        hercules/hir/llvm/gpu.h
        hercules/hir/llvm/llvisitor.h
        hercules/hir/llvm/llvm.h
        hercules/hir/llvm/optimize.h
        hercules/hir/module.h
        hercules/hir/pyextension.h
        hercules/hir/cir.h
        hercules/hir/transform/cleanup/canonical.h
        hercules/hir/transform/cleanup/dead_code.h
        hercules/hir/transform/cleanup/global_demote.h
        hercules/hir/transform/cleanup/replacer.h
        hercules/hir/transform/folding/const_fold.h
        hercules/hir/transform/folding/const_prop.h
        hercules/hir/transform/folding/folding.h
        hercules/hir/transform/folding/rule.h
        hercules/hir/transform/lowering/imperative.h
        hercules/hir/transform/lowering/pipeline.h
        hercules/hir/transform/manager.h
        hercules/hir/transform/parallel/openmp.h
        hercules/hir/transform/parallel/schedule.h
        hercules/hir/transform/pass.h
        hercules/hir/transform/pythonic/dict.h
        hercules/hir/transform/pythonic/generator.h
        hercules/hir/transform/pythonic/io.h
        hercules/hir/transform/pythonic/list.h
        hercules/hir/transform/pythonic/str.h
        hercules/hir/transform/rewrite.h
        hercules/hir/types/types.h
        hercules/hir/util/cloning.h
        hercules/hir/util/context.h
        hercules/hir/util/format.h
        hercules/hir/util/inlining.h
        hercules/hir/util/irtools.h
        hercules/hir/util/iterators.h
        hercules/hir/util/matching.h
        hercules/hir/util/operator.h
        hercules/hir/util/outlining.h
        hercules/hir/util/packs.h
        hercules/hir/util/side_effect.h
        hercules/hir/util/visitor.h
        hercules/hir/value.h
        hercules/hir/var.h
        hercules/util/common.h
        hercules/compiler/jit_extern.h)
file(GLOB_RECURSE HERCULES_CCAST_FILES hercules/ast/cc/*.cc)

set(HERCULES_CPPFILES
        hercules/compiler/compiler.cc
        hercules/compiler/debug_listener.cc
        hercules/compiler/engine.cc
        hercules/compiler/error.cc
        hercules/compiler/jit.cc
        hercules/compiler/memory_manager.cc
        hercules/dsl/plugins.cc
        hercules/engine/vm.cc
        hercules/parser/ast/expr.cc
        hercules/parser/ast/stmt.cc
        hercules/parser/ast/types/type.cc
        hercules/parser/ast/types/link.cc
        hercules/parser/ast/types/class.cc
        hercules/parser/ast/types/function.cc
        hercules/parser/ast/types/union.cc
        hercules/parser/ast/types/static.cc
        hercules/parser/ast/types/traits.cc
        hercules/parser/cache.cc
        hercules/parser/common.cc
        hercules/parser/peg/peg.cc
        hercules/parser/visitors/doc/doc.cc
        hercules/parser/visitors/format/format.cc
        hercules/parser/visitors/simplify/simplify.cc
        hercules/parser/visitors/simplify/ctx.cc
        hercules/parser/visitors/simplify/assign.cc
        hercules/parser/visitors/simplify/basic.cc
        hercules/parser/visitors/simplify/call.cc
        hercules/parser/visitors/simplify/class.cc
        hercules/parser/visitors/simplify/collections.cc
        hercules/parser/visitors/simplify/cond.cc
        hercules/parser/visitors/simplify/function.cc
        hercules/parser/visitors/simplify/access.cc
        hercules/parser/visitors/simplify/import.cc
        hercules/parser/visitors/simplify/loops.cc
        hercules/parser/visitors/simplify/op.cc
        hercules/parser/visitors/simplify/error.cc
        hercules/parser/visitors/translate/translate.cc
        hercules/parser/visitors/translate/translate_ctx.cc
        hercules/parser/visitors/typecheck/typecheck.cc
        hercules/parser/visitors/typecheck/infer.cc
        hercules/parser/visitors/typecheck/ctx.cc
        hercules/parser/visitors/typecheck/assign.cc
        hercules/parser/visitors/typecheck/basic.cc
        hercules/parser/visitors/typecheck/call.cc
        hercules/parser/visitors/typecheck/class.cc
        hercules/parser/visitors/typecheck/collections.cc
        hercules/parser/visitors/typecheck/cond.cc
        hercules/parser/visitors/typecheck/function.cc
        hercules/parser/visitors/typecheck/access.cc
        hercules/parser/visitors/typecheck/loops.cc
        hercules/parser/visitors/typecheck/op.cc
        hercules/parser/visitors/typecheck/error.cc
        hercules/parser/visitors/visitor.cc
        hercules/hir/attribute.cc
        hercules/hir/analyze/analysis.cc
        hercules/hir/analyze/dataflow/capture.cc
        hercules/hir/analyze/dataflow/cfg.cc
        hercules/hir/analyze/dataflow/dominator.cc
        hercules/hir/analyze/dataflow/reaching.cc
        hercules/hir/analyze/module/global_vars.cc
        hercules/hir/analyze/module/side_effect.cc
        hercules/hir/base.cc
        hercules/hir/const.cc
        hercules/hir/dsl/nodes.cc
        hercules/hir/flow.cc
        hercules/hir/func.cc
        hercules/hir/instr.cc
        hercules/hir/llvm/gpu.cc
        hercules/hir/llvm/llvisitor.cc
        hercules/hir/llvm/optimize.cc
        hercules/hir/module.cc
        hercules/hir/transform/cleanup/canonical.cc
        hercules/hir/transform/cleanup/dead_code.cc
        hercules/hir/transform/cleanup/global_demote.cc
        hercules/hir/transform/cleanup/replacer.cc
        hercules/hir/transform/folding/const_fold.cc
        hercules/hir/transform/folding/const_prop.cc
        hercules/hir/transform/folding/folding.cc
        hercules/hir/transform/lowering/imperative.cc
        hercules/hir/transform/lowering/pipeline.cc
        hercules/hir/transform/manager.cc
        hercules/hir/transform/parallel/openmp.cc
        hercules/hir/transform/parallel/schedule.cc
        hercules/hir/transform/pass.cc
        hercules/hir/transform/pythonic/dict.cc
        hercules/hir/transform/pythonic/generator.cc
        hercules/hir/transform/pythonic/io.cc
        hercules/hir/transform/pythonic/list.cc
        hercules/hir/transform/pythonic/str.cc
        hercules/hir/types/types.cc
        hercules/hir/util/cloning.cc
        hercules/hir/util/format.cc
        hercules/hir/util/inlining.cc
        hercules/hir/util/irtools.cc
        hercules/hir/util/matching.cc
        hercules/hir/util/outlining.cc
        hercules/hir/util/side_effect.cc
        hercules/hir/util/visitor.cc
        hercules/hir/value.cc
        hercules/hir/var.cc
        hercules/util/common.cc
        hercules/util/process.cc
        hercules/util/process_unix.cc
        hercules/util/process_win.cc
        ${HERCULES_CCAST_FILES}
)
add_library(herculesc SHARED ${HERCULES_HPPFILES})
target_include_directories(herculesc PRIVATE ${peglib_SOURCE_DIR} ${toml_SOURCE_DIR}/include ${semver_SOURCE_DIR}/include)
target_sources(herculesc PRIVATE ${HERCULES_CPPFILES} hercules_rules.cc omp_rules.cc)
if (ASAN)
    target_compile_options(
            herculesc PRIVATE "-fno-omit-frame-pointer" "-fsanitize=address"
            "-fsanitize-recover=address")
    target_link_libraries(
            herculesc PRIVATE "-fno-omit-frame-pointer" "-fsanitize=address"
            "-fsanitize-recover=address")
endif ()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    set_source_files_properties(hercules_rules.cc hercules/parser/peg/peg.cc
            PROPERTIES COMPILE_FLAGS "-O2")
endif ()
llvm_map_components_to_libnames(
        LLVM_LIBS
        AllTargetsAsmParsers
        AllTargetsCodeGens
        AllTargetsDescs
        AllTargetsInfos
        AggressiveInstCombine
        Analysis
        AsmParser
        BitWriter
        CodeGen
        Core
        Extensions
        IPO
        IRReader
        InstCombine
        Instrumentation
        MC
        MCJIT
        ObjCARCOpts
        OrcJIT
        Remarks
        ScalarOpts
        Support
        Symbolize
        Target
        TransformUtils
        Vectorize
        Passes)
if (APPLE)
    target_link_libraries(herculesc PRIVATE ${LLVM_LIBS} dl herculesrt)
else ()
    target_link_libraries(herculesc PRIVATE ${STATIC_LIBCPP} ${LLVM_LIBS} dl herculesrt)
endif ()

# Gather headers
add_custom_target(
        headers ALL
        COMMENT "Collecting headers"
        BYPRODUCTS "${CMAKE_BINARY_DIR}/include"
        VERBATIM
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/include/hercules"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/hercules"
        "${CMAKE_BINARY_DIR}/include/hercules"
        COMMAND find "${CMAKE_BINARY_DIR}/include" -type f ! -name "*.h" -exec rm {}
        \\;)
add_dependencies(headers herculesrt herculesc)

# Prepare lib directory for plugin compilation
add_custom_target(
        libs ALL
        COMMENT "Collecting libraries"
        BYPRODUCTS "${CMAKE_BINARY_DIR}/lib"
        VERBATIM
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/lib/hercules"
        COMMAND
        ${CMAKE_COMMAND} -E copy
        "${CMAKE_BINARY_DIR}/libherculesc${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CMAKE_BINARY_DIR}/lib/hercules"
        COMMAND
        ${CMAKE_COMMAND} -E copy
        "${CMAKE_BINARY_DIR}/libherculesrt${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CMAKE_BINARY_DIR}/lib/hercules"
        COMMAND
        ${CMAKE_COMMAND} -E copy
        "${CMAKE_BINARY_DIR}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CMAKE_BINARY_DIR}/lib/hercules")
add_dependencies(libs herculesrt herculesc)

if (APPLE AND APPLE_ARM)
    # add_custom_target(
    #   libs_apple_arm ALL
    #   COMMENT "Collecting Apple-specific libraries"
    #   BYPRODUCTS "${CMAKE_BINARY_DIR}/lib"
    #   VERBATIM
    #   COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/lib/hercules"
    #   COMMAND
    #     ${CMAKE_COMMAND} -E copy
    #     "${CMAKE_BINARY_DIR}/libunwind${CMAKE_SHARED_LIBRARY_SUFFIX}"
    #     "${CMAKE_BINARY_DIR}/lib/hercules")
    # add_dependencies(libs_apple_arm herculesrt)
endif ()

# Hercules command-line tool
add_executable(hercules hercules/app/main.cc)
target_link_libraries(hercules PUBLIC ${STATIC_LIBCPP} fmt herculesc hercules_jupyter Threads::Threads)

# Hercules test Download and unpack googletest at configure time
include(FetchContent)
FetchContent_Declare(
        googletest
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rd/googletest-release-1.12.1/googletest-release-1.12.1"
        #URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)
enable_testing()
set(HERCULES_TEST_CPPFILES
        test/hercules_test.cc
        test/types.cc)
add_executable(hercules_test ${HERCULES_TEST_CPPFILES})
target_include_directories(hercules_test PRIVATE test/hir
        "${gc_SOURCE_DIR}/include")
target_link_libraries(hercules_test fmt herculesc herculesrt gtest_main)
target_compile_definitions(hercules_test
        PRIVATE TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

set(HIR_TEST_CPPFILES
        test/hir_test.cc
        test/hir/analyze/dominator.cc
        test/hir/analyze/reaching.cc
        test/hir/base.cc
        test/hir/constant.cc
        test/hir/flow.cc
        test/hir/func.cc
        test/hir/instr.cc
        test/hir/module.cc
        test/hir/transform/manager.cc
        test/hir/types/types.cc
        test/hir/util/matching.cc
        test/hir/value.cc
        test/hir/var.cc
        test/types.cc)
add_executable(hir_test ${HIR_TEST_CPPFILES})
target_include_directories(hir_test PRIVATE test/hir
        "${gc_SOURCE_DIR}/include")
target_link_libraries(hir_test fmt herculesc herculesrt gtest_main)
target_compile_definitions(hir_test
        PRIVATE TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")


install(TARGETS herculesrt herculesc hercules_jupyter DESTINATION lib/hercules)
install(FILES ${CMAKE_BINARY_DIR}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib/hercules)
install(TARGETS hercules DESTINATION bin)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/hercules
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

install(DIRECTORY ${CMAKE_SOURCE_DIR}/stdlib DESTINATION lib/hercules)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/jit/ DESTINATION python)
install(DIRECTORY DESTINATION lib/hercules/plugins)
