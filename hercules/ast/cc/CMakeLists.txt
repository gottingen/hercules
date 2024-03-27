# Copyright (C) 2017-2023 Jonathan Müller and cppast contributors
# SPDX-License-Identifier: MIT
# found in the top-level directory of this distribution.

set(detail_header
        ../include/cppast/detail/assert.h
        ../include/cppast/detail/intrusive_list.h)
set(header
    ../include/cppast/code_generator.h
    ../include/cppast/compile_config.h
    ../include/cppast/cpp_alias_template.h
    ../include/cppast/cpp_array_type.h
    ../include/cppast/cpp_attribute.h
    ../include/cppast/cpp_class.h
    ../include/cppast/cpp_class_template.h
    ../include/cppast/cpp_concept.h
    ../include/cppast/cpp_decltype_type.h
    ../include/cppast/cpp_entity.h
    ../include/cppast/cpp_entity_container.h
    ../include/cppast/cpp_entity_index.h
    ../include/cppast/cpp_entity_kind.h
    ../include/cppast/cpp_entity_ref.h
    ../include/cppast/cpp_enum.h
    ../include/cppast/cpp_expression.h
    ../include/cppast/cpp_file.h
    ../include/cppast/cpp_forward_declarable.h
    ../include/cppast/cpp_friend.h
    ../include/cppast/cpp_function.h
    ../include/cppast/cpp_function_template.h
    ../include/cppast/cpp_function_type.h
    ../include/cppast/cpp_language_linkage.h
    ../include/cppast/cpp_member_function.h
    ../include/cppast/cpp_member_variable.h
    ../include/cppast/cpp_namespace.h
    ../include/cppast/cpp_preprocessor.h
    ../include/cppast/cpp_static_assert.h
    ../include/cppast/cpp_storage_class_specifiers.h
    ../include/cppast/cpp_template.h
    ../include/cppast/cpp_template_parameter.h
    ../include/cppast/cpp_token.h
    ../include/cppast/cpp_type.h
    ../include/cppast/cpp_type_alias.h
    ../include/cppast/cpp_variable.h
    ../include/cppast/cpp_variable_base.h
    ../include/cppast/cpp_variable_template.h
    ../include/cppast/diagnostic.h
    ../include/cppast/diagnostic_logger.h
    ../include/cppast/cppast_fwd.h
    ../include/cppast/libclang_parser.h
    ../include/cppast/parser.h
    ../include/cppast/visitor.h)
set(source
        code_generator.cpp
        cpp_alias_template.cpp
        cpp_attribute.cpp
        cpp_class.cpp
        cpp_class_template.cpp
        cpp_concept.cpp
        cpp_entity.cpp
        cpp_entity_index.cpp
        cpp_entity_kind.cpp
        cpp_enum.cpp
        cpp_expression.cpp
        cpp_file.cpp
        cpp_forward_declarable.cpp
        cpp_friend.cpp
        cpp_function.cpp
        cpp_function_template.cpp
        cpp_language_linkage.cpp
        cpp_member_function.cpp
        cpp_member_variable.cpp
        cpp_namespace.cpp
        cpp_preprocessor.cpp
        cpp_static_assert.cpp
        cpp_template_parameter.cpp
        cpp_token.cpp
        cpp_type.cpp
        cpp_type_alias.cpp
        cpp_variable.cpp
        cpp_variable_template.cpp
        diagnostic_logger.cpp
        visitor.cpp)
set(libclang_source
        libclang/class_parser.cpp
        libclang/concept_parser.cpp
        libclang/cxtokenizer.cpp
        libclang/cxtokenizer.h
        libclang/debug_helper.cpp
        libclang/debug_helper.h
        libclang/enum_parser.cpp
        libclang/expression_parser.cpp
        libclang/friend_parser.cpp
        libclang/function_parser.cpp
        libclang/language_linkage_parser.cpp
        libclang/libclang_parser.cpp
        libclang/libclang_visitor.h
        libclang/namespace_parser.cpp
        libclang/parse_error.h
        libclang/parse_functions.cpp
        libclang/parse_functions.h
        libclang/preprocessor.cpp
        libclang/preprocessor.h
        libclang/raii_wrapper.h
        libclang/template_parser.cpp
        libclang/type_parser.cpp
        libclang/variable_parser.cpp)

add_library(cppast ${detail_header} ${header} ${source} ${libclang_source})
target_compile_features(cppast PUBLIC cxx_std_11)
target_include_directories(cppast PRIVATE ../include SYSTEM INTERFACE ../include)
target_link_libraries(cppast PUBLIC type_safe _cppast_tiny_process _cppast_libclang)
target_compile_definitions(cppast PUBLIC
                                CPPAST_VERSION_MINOR="${cppast_VERSION_MINOR}"
                                CPPAST_VERSION_MAJOR="${cppast_VERSION_MAJOR}"
                                CPPAST_VERSION_STRING="${cppast_VERSION}")

if(${is_top_level_project})
    target_compile_options(cppast PRIVATE
                               # clang/GCC warnings
                               $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
                               -pedantic-errors -Werror -Wall -Wextra -Wconversion -Wsign-conversion>
                               # disable noexcept type warning on GCC
                               $<$<CXX_COMPILER_ID:GNU>: -Wno-noexcept-type>
                               # MSVC warnings
                               $<$<CXX_COMPILER_ID:MSVC>:
                               /W3>)
endif()

install(TARGETS cppast)
install(DIRECTORY ../include/ DESTINATION include)
