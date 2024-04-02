
include(carbin_print)
macro(CARBIN_ENSURE_OUT_OF_SOURCE_BUILD errorMessage)

    string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" is_insource)
    if (is_insource)
        carbin_error(${errorMessage} "In-source builds are not allowed.
    CMake would overwrite the makefiles distributed with Compiler-RT.
    Please create a directory and run cmake from there, passing the path
    to this source directory as the last argument.
    This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
    Please delete them.")

    endif (is_insource)

endmacro(CARBIN_ENSURE_OUT_OF_SOURCE_BUILD)