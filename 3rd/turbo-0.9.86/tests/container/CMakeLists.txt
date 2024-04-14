#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME        btree_test_common
        HEADERS
        "btree_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)

carbin_cc_test(
        NAME
        btree_test
        SOURCES
        "btree_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
        turbo::btree_test_common
        turbo::hash_testing
        turbo::test_instance_tracker
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        compressed_tuple_test
        SOURCES
        "compressed_tuple_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::test_instance_tracker
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        fixed_array_test
        SOURCES
        "fixed_array_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::exception_testing
        turbo::hash_testing
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        fixed_array_exception_safety_test
        SOURCES
        "fixed_array_exception_safety_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::exception_safety_testing
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        inlined_vector_test
        SOURCES
        "inlined_vector_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::test_instance_tracker
        turbo::exception_testing
        turbo::hash_testing
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        inlined_vector_exception_safety_test
        SOURCES
        "inlined_vector_exception_safety_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        turbo::exception_safety_testing
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        test_instance_tracker
        HEADERS
        "test_instance_tracker.h"
        SOURCES
        "test_instance_tracker.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}
)

carbin_cc_test(
        NAME
        test_instance_tracker_test
        SOURCES
        "test_instance_tracker_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::test_instance_tracker
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        flat_hash_map_test
        SOURCES
        "flat_hash_map_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::hash_generator_testing
        turbo::unordered_map_constructor_test
        turbo::unordered_map_lookup_test
        turbo::unordered_map_members_test
        turbo::unordered_map_modifiers_test
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)
#TODO disabled now(jeff)
#[[
carbin_cc_test(
        NAME
        flat_hash_set_test
        SOURCES
        "flat_hash_set_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-DUNORDERED_SET_CXX17"
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
        turbo::hash_generator_testing
        turbo::unordered_set_constructor_test
        turbo::unordered_set_lookup_test
        turbo::unordered_set_members_test
        turbo::unordered_set_modifiers_test
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)
]]
carbin_cc_test(
        NAME
        node_hash_map_test
        SOURCES
        "node_hash_map_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::hash_generator_testing
        turbo::turbo
        turbo::unordered_map_constructor_test
        turbo::unordered_map_lookup_test
        turbo::unordered_map_members_test
        turbo::unordered_map_modifiers_test
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        node_hash_set_test
        SOURCES
        "node_hash_set_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-DUNORDERED_SET_CXX17"
        DEPS
        turbo::hash_generator_testing
        turbo::turbo
        turbo::unordered_set_constructor_test
        turbo::unordered_set_lookup_test
        turbo::unordered_set_members_test
        turbo::unordered_set_modifiers_test
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        container_memory_test
        SOURCES
        "container_memory_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
        turbo::test_instance_tracker
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        hash_function_defaults_test
        SOURCES
        "hash_function_defaults_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        hash_generator_testing
        HEADERS
        "hash_generator_testing.h"
        SOURCES
        "hash_generator_testing.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        turbo::hash_policy_testing
        turbo::turbo
)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        hash_policy_testing
        HEADERS
        "hash_policy_testing.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        turbo::turbo
)

carbin_cc_test(
        NAME
        hash_policy_testing_test
        SOURCES
        "hash_policy_testing_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        hash_policy_traits_test
        SOURCES
        "hash_policy_traits_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        ${GMOCK_LIB}
        turbo::turbo
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        common_policy_traits_test
        SOURCES
        "common_policy_traits_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        ${GMOCK_LIB}
        turbo::turbo
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        hashtablez_sampler_test
        SOURCES
        "hashtablez_sampler_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        ${GMOCK_LIB}
        turbo::turbo
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        node_slot_policy_test
        SOURCES
        "node_slot_policy_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        raw_hash_set_test
        SOURCES
        "raw_hash_set_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        ${GMOCK_LIB}
        turbo::hash_policy_testing
        turbo::turbo
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        raw_hash_set_allocator_test
        SOURCES
        "raw_hash_set_allocator_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        layout_test
        SOURCES
        "layout_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        tracked
        HEADERS
        "tracked.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${GTEST_LIB}
        turbo::turbo

)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_map_constructor_test
        HEADERS
        "unordered_map_constructor_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::hash_generator_testing
        turbo::hash_policy_testing
        ${GTEST_LIB}
        ${GMOCK_LIB}

)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_map_lookup_test
        HEADERS
        "unordered_map_lookup_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::hash_generator_testing
        turbo::hash_policy_testing
        ${GTEST_LIB}
        ${GMOCK_LIB}

)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_map_members_test
        HEADERS
        "unordered_map_members_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}

)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_map_modifiers_test
        HEADERS
        "unordered_map_modifiers_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::hash_generator_testing
        turbo::hash_policy_testing
        ${GMOCK_LIB}
        ${GTEST_LIB}

)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_set_constructor_test
        HEADERS
        "unordered_set_constructor_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::hash_generator_testing
        turbo::hash_policy_testing
        ${GMOCK_LIB}
        ${GTEST_LIB}

)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_set_lookup_test
        HEADERS
        "unordered_set_lookup_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::hash_generator_testing
        turbo::hash_policy_testing
        ${GMOCK_LIB}
        ${GTEST_LIB}
)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_set_members_test
        HEADERS
        "unordered_set_members_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GMOCK_LIB}
        ${GTEST_LIB}

)

# Internal-only target, do not depend on directly.
carbin_cc_library(
        NAMESPACE
        turbo
        NAME
        unordered_set_modifiers_test
        HEADERS
        "unordered_set_modifiers_test.h"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::hash_generator_testing
        turbo::hash_policy_testing
        ${GMOCK_LIB}
)

carbin_cc_test(
        NAME
        unordered_set_test
        SOURCES
        "unordered_set_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::unordered_set_constructor_test
        turbo::unordered_set_lookup_test
        turbo::unordered_set_members_test
        turbo::unordered_set_modifiers_test
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        unordered_map_test
        SOURCES
        "unordered_map_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::unordered_map_constructor_test
        turbo::unordered_map_lookup_test
        turbo::unordered_map_members_test
        turbo::unordered_map_modifiers_test
        ${GTEST_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        sample_element_size_test
        SOURCES
        "sample_element_size_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)

carbin_cc_test(
        NAME
        ring_buffer_test
        SOURCES
        "ring_buffer_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
        ${GTEST_LIB}
        ${GMOCK_LIB}
        ${GTEST_MAIN_LIB}
)


carbin_cc_test(
        NAME
        intrusive_list_test
        SOURCES
        "intrusive_list_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)

carbin_cc_test(
        NAME
        bounded_queue_test
        SOURCES
        "bounded_queue_test.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        turbo::turbo
)

carbin_cc_test(
        NAME container_batch_test
        SOURCES
        batch_test.cc
        COPTS ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo::turbo
)
