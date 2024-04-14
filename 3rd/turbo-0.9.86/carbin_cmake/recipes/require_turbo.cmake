
find_path(TURBO_INCLUDE_PATH NAMES turbo/version.h)
find_library(TURBO_LIB NAMES libturbo.a turbo)
include_directories(${TURBO_INCLUDE_PATH})
if((NOT TURBO_INCLUDE_PATH) OR (NOT TURBO_LIB))
    carbin_error("Fail to find turbo")
endif()
