
find_path(MELON_INCLUDE_PATH NAMES melon/idl_options.pb.h)
find_library(MELON_LIB NAMES melon)
include_directories(${MELON_INCLUDE_PATH})
if((NOT MELON_INCLUDE_PATH) OR (NOT MELON_LIB))
    message(FATAL_ERROR "Fail to find melon")
endif()
