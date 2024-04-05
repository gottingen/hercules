# Only test under some architecture, because it's just a sanity check, no full
# coverage is needed.

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT CMAKE_OSX_ARCHITECTURES)

add_library(test_doc_any_arch OBJECT
            explicit_use_of_an_instruction_set_mean_aligned.cc
            explicit_use_of_an_instruction_set_mean_arch_independent.cc
            explicit_use_of_an_instruction_set_mean.cc
            explicit_use_of_an_instruction_set_mean_tag_dispatch.cc
            manipulating_abstract_batches.cc
            manipulating_parametric_batches.cc
            writing_vectorized_code.cc)
target_compile_options(test_doc_any_arch PRIVATE -mavx)

add_library(test_doc_avx2 OBJECT
            explicit_use_of_an_instruction_set.cc
            sum_avx2.cc)
target_compile_options(test_doc_avx2 PRIVATE -mavx2)

add_library(test_doc_sse2 OBJECT
            sum_sse2.cc)
target_compile_options(test_doc_sse2 PRIVATE -msse2)

add_dependencies(xtest test_doc_any_arch test_doc_avx2 test_doc_sse2)

endif()
