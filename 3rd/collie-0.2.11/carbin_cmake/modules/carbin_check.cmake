
include(CMakeParseArguments)
include(carbin_config_cxx_opts)
include(carbin_install_dirs)


function(carbin_check_target my_target)
    if(NOT TARGET ${my_target})
        message(FATAL_ERROR " CARBIN: compiling ${PROJECT_NAME} requires a ${my_target} CMake target in your project,
                   see CMake/README.md for more details")
    endif(NOT TARGET ${my_target})
endfunction()