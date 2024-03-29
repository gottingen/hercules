cmake_minimum_required(VERSION 3.14)
project(
        Hercules
        VERSION "0.1.1"
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

add_executable(peg2cpp hercules/util/peg2cpp.cpp)
target_include_directories(peg2cpp PRIVATE ${peglib_SOURCE_DIR})
target_link_libraries(peg2cpp PRIVATE Threads::Threads fmt)
add_custom_command(
        OUTPUT hercules_rules.cpp
        COMMAND peg2cpp ${CMAKE_SOURCE_DIR}/hercules/parser/peg/grammar.peg
        hercules_rules.cpp hercules
        DEPENDS peg2cpp hercules/parser/peg/grammar.peg)
add_custom_command(
        OUTPUT omp_rules.cpp
        COMMAND peg2cpp ${CMAKE_SOURCE_DIR}/hercules/parser/peg/openmp.peg omp_rules.cpp
        omp
        DEPENDS peg2cpp hercules/parser/peg/openmp.peg)

# Hercules Jupyter library
set(HERCULES_JUPYTER_FILES hercules/util/jupyter.h hercules/util/jupyter.cpp)
add_library(hercules_jupyter SHARED ${HERCULES_JUPYTER_FILES})

# Hercules runtime library
set(HERCULESRT_FILES hercules/runtime/lib.h hercules/runtime/lib.cpp
        hercules/runtime/re.cpp hercules/runtime/exc.cpp
        hercules/runtime/gpu.cpp)
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
        hercules/compiler/compiler.cpp
        hercules/compiler/debug_listener.cpp
        hercules/compiler/engine.cpp
        hercules/compiler/error.cpp
        hercules/compiler/jit.cpp
        hercules/compiler/memory_manager.cpp
        hercules/dsl/plugins.cpp
        hercules/parser/ast/expr.cpp
        hercules/parser/ast/stmt.cpp
        hercules/parser/ast/types/type.cpp
        hercules/parser/ast/types/link.cpp
        hercules/parser/ast/types/class.cpp
        hercules/parser/ast/types/function.cpp
        hercules/parser/ast/types/union.cpp
        hercules/parser/ast/types/static.cpp
        hercules/parser/ast/types/traits.cpp
        hercules/parser/cache.cpp
        hercules/parser/common.cpp
        hercules/parser/peg/peg.cpp
        hercules/parser/visitors/doc/doc.cpp
        hercules/parser/visitors/format/format.cpp
        hercules/parser/visitors/simplify/simplify.cpp
        hercules/parser/visitors/simplify/ctx.cpp
        hercules/parser/visitors/simplify/assign.cpp
        hercules/parser/visitors/simplify/basic.cpp
        hercules/parser/visitors/simplify/call.cpp
        hercules/parser/visitors/simplify/class.cpp
        hercules/parser/visitors/simplify/collections.cpp
        hercules/parser/visitors/simplify/cond.cpp
        hercules/parser/visitors/simplify/function.cpp
        hercules/parser/visitors/simplify/access.cpp
        hercules/parser/visitors/simplify/import.cpp
        hercules/parser/visitors/simplify/loops.cpp
        hercules/parser/visitors/simplify/op.cpp
        hercules/parser/visitors/simplify/error.cpp
        hercules/parser/visitors/translate/translate.cpp
        hercules/parser/visitors/translate/translate_ctx.cpp
        hercules/parser/visitors/typecheck/typecheck.cpp
        hercules/parser/visitors/typecheck/infer.cpp
        hercules/parser/visitors/typecheck/ctx.cpp
        hercules/parser/visitors/typecheck/assign.cpp
        hercules/parser/visitors/typecheck/basic.cpp
        hercules/parser/visitors/typecheck/call.cpp
        hercules/parser/visitors/typecheck/class.cpp
        hercules/parser/visitors/typecheck/collections.cpp
        hercules/parser/visitors/typecheck/cond.cpp
        hercules/parser/visitors/typecheck/function.cpp
        hercules/parser/visitors/typecheck/access.cpp
        hercules/parser/visitors/typecheck/loops.cpp
        hercules/parser/visitors/typecheck/op.cpp
        hercules/parser/visitors/typecheck/error.cpp
        hercules/parser/visitors/visitor.cpp
        hercules/cir/attribute.cpp
        hercules/cir/analyze/analysis.cpp
        hercules/cir/analyze/dataflow/capture.cpp
        hercules/cir/analyze/dataflow/cfg.cpp
        hercules/cir/analyze/dataflow/dominator.cpp
        hercules/cir/analyze/dataflow/reaching.cpp
        hercules/cir/analyze/module/global_vars.cpp
        hercules/cir/analyze/module/side_effect.cpp
        hercules/cir/base.cpp
        hercules/cir/const.cpp
        hercules/cir/dsl/nodes.cpp
        hercules/cir/flow.cpp
        hercules/cir/func.cpp
        hercules/cir/instr.cpp
        hercules/cir/llvm/gpu.cpp
        hercules/cir/llvm/llvisitor.cpp
        hercules/cir/llvm/optimize.cpp
        hercules/cir/module.cpp
        hercules/cir/transform/cleanup/canonical.cpp
        hercules/cir/transform/cleanup/dead_code.cpp
        hercules/cir/transform/cleanup/global_demote.cpp
        hercules/cir/transform/cleanup/replacer.cpp
        hercules/cir/transform/folding/const_fold.cpp
        hercules/cir/transform/folding/const_prop.cpp
        hercules/cir/transform/folding/folding.cpp
        hercules/cir/transform/lowering/imperative.cpp
        hercules/cir/transform/lowering/pipeline.cpp
        hercules/cir/transform/manager.cpp
        hercules/cir/transform/parallel/openmp.cpp
        hercules/cir/transform/parallel/schedule.cpp
        hercules/cir/transform/pass.cpp
        hercules/cir/transform/pythonic/dict.cpp
        hercules/cir/transform/pythonic/generator.cpp
        hercules/cir/transform/pythonic/io.cpp
        hercules/cir/transform/pythonic/list.cpp
        hercules/cir/transform/pythonic/str.cpp
        hercules/cir/types/types.cpp
        hercules/cir/util/cloning.cpp
        hercules/cir/util/format.cpp
        hercules/cir/util/inlining.cpp
        hercules/cir/util/irtools.cpp
        hercules/cir/util/matching.cpp
        hercules/cir/util/outlining.cpp
        hercules/cir/util/side_effect.cpp
        hercules/cir/util/visitor.cpp
        hercules/cir/value.cpp
        hercules/cir/var.cpp
        hercules/util/common.cpp)
add_library(herculesc SHARED ${HERCULES_HPPFILES})
target_include_directories(herculesc PRIVATE ${peglib_SOURCE_DIR} ${toml_SOURCE_DIR}/include ${semver_SOURCE_DIR}/include)
target_sources(herculesc PRIVATE ${HERCULES_CPPFILES} hercules_rules.cpp omp_rules.cpp)
if (ASAN)
    target_compile_options(
            herculesc PRIVATE "-fno-omit-frame-pointer" "-fsanitize=address"
            "-fsanitize-recover=address")
    target_link_libraries(
            herculesc PRIVATE "-fno-omit-frame-pointer" "-fsanitize=address"
            "-fsanitize-recover=address")
endif ()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    set_source_files_properties(hercules_rules.cpp hercules/parser/peg/peg.cpp
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
add_executable(hercules hercules/app/main.cpp)
target_link_libraries(hercules PUBLIC ${STATIC_LIBCPP} fmt herculesc hercules_jupyter Threads::Threads)

# Hercules test Download and unpack googletest at configure time
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)
enable_testing()
set(HERCULES_TEST_CPPFILES
        test/main.cpp
        test/cir/analyze/dominator.cpp
        test/cir/analyze/reaching.cpp
        test/cir/base.cpp
        test/cir/constant.cpp
        test/cir/flow.cpp
        test/cir/func.cpp
        test/cir/instr.cpp
        test/cir/module.cpp
        test/cir/transform/manager.cpp
        test/cir/types/types.cpp
        test/cir/util/matching.cpp
        test/cir/value.cpp
        test/cir/var.cpp
        test/types.cpp)
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
