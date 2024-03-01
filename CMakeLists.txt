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
set(CMAKE_VERBOSE_MAKEFILE ON)
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
find_package(LLVM REQUIRED)

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
        hercules/cir/analyze/analysis.h
        hercules/cir/analyze/dataflow/capture.h
        hercules/cir/analyze/dataflow/cfg.h
        hercules/cir/analyze/dataflow/dominator.h
        hercules/cir/analyze/dataflow/reaching.h
        hercules/cir/analyze/module/global_vars.h
        hercules/cir/analyze/module/side_effect.h
        hercules/cir/attribute.h
        hercules/cir/base.h
        hercules/cir/const.h
        hercules/cir/dsl/codegen.h
        hercules/cir/dsl/nodes.h
        hercules/cir/flow.h
        hercules/cir/func.h
        hercules/cir/instr.h
        hercules/cir/llvm/gpu.h
        hercules/cir/llvm/llvisitor.h
        hercules/cir/llvm/llvm.h
        hercules/cir/llvm/optimize.h
        hercules/cir/module.h
        hercules/cir/pyextension.h
        hercules/cir/cir.h
        hercules/cir/transform/cleanup/canonical.h
        hercules/cir/transform/cleanup/dead_code.h
        hercules/cir/transform/cleanup/global_demote.h
        hercules/cir/transform/cleanup/replacer.h
        hercules/cir/transform/folding/const_fold.h
        hercules/cir/transform/folding/const_prop.h
        hercules/cir/transform/folding/folding.h
        hercules/cir/transform/folding/rule.h
        hercules/cir/transform/lowering/imperative.h
        hercules/cir/transform/lowering/pipeline.h
        hercules/cir/transform/manager.h
        hercules/cir/transform/parallel/openmp.h
        hercules/cir/transform/parallel/schedule.h
        hercules/cir/transform/pass.h
        hercules/cir/transform/pythonic/dict.h
        hercules/cir/transform/pythonic/generator.h
        hercules/cir/transform/pythonic/io.h
        hercules/cir/transform/pythonic/list.h
        hercules/cir/transform/pythonic/str.h
        hercules/cir/transform/rewrite.h
        hercules/cir/types/types.h
        hercules/cir/util/cloning.h
        hercules/cir/util/context.h
        hercules/cir/util/format.h
        hercules/cir/util/inlining.h
        hercules/cir/util/irtools.h
        hercules/cir/util/iterators.h
        hercules/cir/util/matching.h
        hercules/cir/util/operator.h
        hercules/cir/util/outlining.h
        hercules/cir/util/packs.h
        hercules/cir/util/side_effect.h
        hercules/cir/util/visitor.h
        hercules/cir/value.h
        hercules/cir/var.h
        hercules/util/common.h
        hercules/compiler/jit_extern.h)
set(HERCULES_CPPFILES
        hercules/compiler/compiler.cc
        hercules/compiler/debug_listener.cc
        hercules/compiler/engine.cc
        hercules/compiler/error.cc
        hercules/compiler/jit.cc
        hercules/compiler/memory_manager.cc
        hercules/dsl/plugins.cc
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
        hercules/cir/attribute.cc
        hercules/cir/analyze/analysis.cc
        hercules/cir/analyze/dataflow/capture.cc
        hercules/cir/analyze/dataflow/cfg.cc
        hercules/cir/analyze/dataflow/dominator.cc
        hercules/cir/analyze/dataflow/reaching.cc
        hercules/cir/analyze/module/global_vars.cc
        hercules/cir/analyze/module/side_effect.cc
        hercules/cir/base.cc
        hercules/cir/const.cc
        hercules/cir/dsl/nodes.cc
        hercules/cir/flow.cc
        hercules/cir/func.cc
        hercules/cir/instr.cc
        hercules/cir/llvm/gpu.cc
        hercules/cir/llvm/llvisitor.cc
        hercules/cir/llvm/optimize.cc
        hercules/cir/module.cc
        hercules/cir/transform/cleanup/canonical.cc
        hercules/cir/transform/cleanup/dead_code.cc
        hercules/cir/transform/cleanup/global_demote.cc
        hercules/cir/transform/cleanup/replacer.cc
        hercules/cir/transform/folding/const_fold.cc
        hercules/cir/transform/folding/const_prop.cc
        hercules/cir/transform/folding/folding.cc
        hercules/cir/transform/lowering/imperative.cc
        hercules/cir/transform/lowering/pipeline.cc
        hercules/cir/transform/manager.cc
        hercules/cir/transform/parallel/openmp.cc
        hercules/cir/transform/parallel/schedule.cc
        hercules/cir/transform/pass.cc
        hercules/cir/transform/pythonic/dict.cc
        hercules/cir/transform/pythonic/generator.cc
        hercules/cir/transform/pythonic/io.cc
        hercules/cir/transform/pythonic/list.cc
        hercules/cir/transform/pythonic/str.cc
        hercules/cir/types/types.cc
        hercules/cir/util/cloning.cc
        hercules/cir/util/format.cc
        hercules/cir/util/inlining.cc
        hercules/cir/util/irtools.cc
        hercules/cir/util/matching.cc
        hercules/cir/util/outlining.cc
        hercules/cir/util/side_effect.cc
        hercules/cir/util/visitor.cc
        hercules/cir/value.cc
        hercules/cir/var.cc
        hercules/util/common.cc)
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
    target_link_libraries(herculesc PRIVATE ${LLVM_LIBS} fmt dl herculesrt)
else ()
    target_link_libraries(herculesc PRIVATE ${STATIC_LIBCPP} ${LLVM_LIBS} fmt dl herculesrt)
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
        test/cir/analyze/dominator.cc
        test/cir/analyze/reaching.cc
        test/cir/base.cc
        test/cir/constant.cc
        test/cir/flow.cc
        test/cir/func.cc
        test/cir/instr.cc
        test/cir/module.cc
        test/cir/transform/manager.cc
        test/cir/types/types.cc
        test/cir/util/matching.cc
        test/cir/value.cc
        test/cir/var.cc
        test/types.cc)
add_executable(hercules_test ${HERCULES_TEST_CPPFILES})
target_include_directories(hercules_test PRIVATE test/cir
        "${gc_SOURCE_DIR}/include")
target_link_libraries(hercules_test fmt herculesc herculesrt gtest_main)
target_compile_definitions(hercules_test
        PRIVATE TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

install(TARGETS herculesrt herculesc hercules_jupyter DESTINATION lib/hercules)
install(FILES ${CMAKE_BINARY_DIR}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION lib/hercules)
install(TARGETS hercules DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/hercules DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/stdlib DESTINATION lib/hercules)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/jit/ DESTINATION python)
install(DIRECTORY DESTINATION lib/hercules/plugins)
