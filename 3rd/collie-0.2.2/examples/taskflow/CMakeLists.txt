list(APPEND TF_EXAMPLES
        simple
        attach_data
        async
        subflow_async
        dependent_async
        dependent_async_algorithm
        observer
        subflow
        fibonacci
        condition
        multi_condition
        switch_case
        do_while_loop
        while_loop
        if_else
        nested_if_else
        priority
        visualization
        parallel_for
        parallel_sort
        reduce
        inclusive_scan
        exclusive_scan
        pipeline
        scalable_pipeline
        scalable_pipeline_with_deferred_tokens
        text_pipeline
        taskflow_pipeline
        parallel_graph_pipeline
        parallel_data_pipeline
        pipeline_with_deferred_tokens
        run
        corun
        runtime
        composition
        limited_concurrency
        cancel
        exception
)

foreach (example IN LISTS TF_EXAMPLES)
    carbin_cc_binary(
            NAME ${example}
            SOURCES ${example}.cc
            COPTS ${USER_CXX_FLAGS}
            DEPS pthread
    )
endforeach ()

# -----------------------------------------------------------------------------
# cuda examples
# -----------------------------------------------------------------------------
option(TF_BUILD_CUDA "Build CUDA examples" OFF)
if (TF_BUILD_CUDA)
    add_subdirectory(cuda)
endif ()

# -----------------------------------------------------------------------------
# sycl examples
# -----------------------------------------------------------------------------
if (TF_BUILD_SYCL)
    add_subdirectory(sycl)
endif ()
