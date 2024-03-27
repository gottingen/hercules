
find_path(ROCKSDB_INCLUDE_PATH NAMES rocksdb/env.h)
find_library(ROCKSDB_LIB NAMES librocksdb.a rocksdb)
include_directories(${ROCKSDB_INCLUDE_PATH})
if((NOT ROCKSDB_INCLUDE_PATH) OR (NOT ROCKSDB_LIB))
    carbin_error("Fail to find rocksdb")
endif()