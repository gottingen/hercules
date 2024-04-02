set(INTEL_PROCESSORS
    knl knm skylake-avx512 cannonlake icelake-client
    icelake-server cascadelake cooperlake tigerlake sapphirerapids alderlake
    rocketlake graniterapids graniterapids-d znver4)
set(CMAKE_CXX_FLAGS "")

foreach(INTEL_PROCESSOR ${INTEL_PROCESSORS})
    check_cxx_compiler_flag(-march=${INTEL_PROCESSOR} FLAG_SUPPORTED_${INTEL_PROCESSOR})
    if(FLAG_SUPPORTED_${INTEL_PROCESSOR})
        message(STATUS ${INTEL_PROCESSOR})
        add_library(test_${INTEL_PROCESSOR} OBJECT dummy.cc)
        target_compile_options(test_${INTEL_PROCESSOR} PRIVATE -march=${INTEL_PROCESSOR})
    endif()
endforeach()

