# -----------------------------------------------------------------------------
# cuda examples
# -----------------------------------------------------------------------------
include(CheckLanguage)
message(STATUS "Configuring CUDA ...")
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "\nNo CUDA compiler found")
endif()
enable_language(CUDA)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
  message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  message(FATAL_ERROR "\nTaskflow requires CUDA at least v11")
endif()

add_library(error_settings INTERFACE)
add_library(tf::error_settings ALIAS error_settings)

message(STATUS "Configuring CUDA compiler options ...")
target_compile_options(
        error_settings
        BEFORE
        INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=-Wall,-Wextra,-Wfatal-errors,-Wshadow>
)

list(APPEND TF_CUDA_EXAMPLES 
  cuda_device_property 
  cuda_saxpy 
  cuda_matmul 
  cuda_knn
  cuda_capturer
  cuda_reduce
  cuda_scan
  cuda_merge
  cuda_sort
  cuda_transform
  cuda_rebind
  cuda_find
)

foreach(cuda_example IN LISTS TF_CUDA_EXAMPLES)
  add_executable(${cuda_example} ${cuda_example}.cu)
  target_link_libraries(${cuda_example}
    ${PROJECT_NAME} error_settings Threads::Threads
  )

  # avoid cmake 3.18+ warning
  # we let nvcc to decide the flag if the architecture is not given
  if(NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${cuda_example} PROPERTY CUDA_ARCHITECTURES OFF)
  endif()
endforeach()

