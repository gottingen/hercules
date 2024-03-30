# Copyright (C) 2017-2023 Jonathan Müller and cppast contributors
# SPDX-License-Identifier: MIT
# found in the top-level directory of this distribution.

# Fetch catch.

set(TESTS_SRC
        code_generator.cc
        cpp_alias_template.cc
        cpp_attribute.cc
        cpp_class.cc
        cpp_class_template.cc
        cpp_concept.cc
        cpp_enum.cc
        cpp_friend.cc
        cpp_function.cc
        cpp_function_template.cc
        cpp_language_linkage.cc
        cpp_member_function.cc
        cpp_member_variable.cc
        cpp_namespace.cc
        cpp_preprocessor.cc
        cpp_static_assert.cc
        cpp_template_parameter.cc
        cpp_token.cc
        cpp_type_alias.cc
        cpp_variable.cc
        integration.cc
        libclang_parser.cc
        parser.cc
        preprocessor.cc
        visitor.cc)

# generate list of source files for the self parsing test
get_target_property(files ccast SOURCES)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ccast_files.h "// list of cppast source file includes\n")
foreach(file ${files})
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/ccast_files.h "\"${CMAKE_CURRENT_SOURCE_DIR}/../${file}\",\n")
endforeach()

add_executable(ccast_test test.cc test_parser.h ${TESTS_SRC})
target_include_directories(ccast_test PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ccast_test PUBLIC ccast hprocess)
target_compile_definitions(ccast_test PUBLIC CPPAST_INTEGRATION_FILE="${CMAKE_CURRENT_SOURCE_DIR}/integration.cc"
                                              CPPAST_COMPILE_COMMANDS="${CMAKE_BINARY_DIR}")

add_test(NAME unit_test COMMAND ccast_test "~[integration]")
add_test(NAME integration_test COMMAND ccast_test "[integration]")

