
find_path(GTEST_INCLUDE_PATH NAMES gtest/gtest.h)
find_library(GTEST_LIB NAMES libgtest.a gtest)
find_library(GTEST_MAIN_LIB NAMES libgtest_main.a gtest_main)
include_directories(${GTEST_INCLUDE_PATH})
if((NOT GTEST_INCLUDE_PATH) OR (NOT GTEST_LIB))
    carbin_error("Fail to find gtest")
endif()
