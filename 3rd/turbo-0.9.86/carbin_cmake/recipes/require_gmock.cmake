
find_path(GMOCK_INCLUDE_PATH NAMES gmock/gmock.h)
find_library(GMOCK_LIB NAMES libgmock.a gmock)
find_library(GMOCK__MAIN_LIB NAMES libgmock_main.a gmock_main)
include_directories(${GMOCK_INCLUDE_PATH})
if((NOT GMOCK_INCLUDE_PATH) OR (NOT GMOCK_LIB))
    carbin_error("Fail to find gmock")
endif()