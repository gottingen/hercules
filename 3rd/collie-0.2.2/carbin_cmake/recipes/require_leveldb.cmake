

find_path(LEVELDB_INCLUDE_PATH NAMES leveldb/db.h)
find_library(LEVELDB_LIB NAMES leveldb)
if ((NOT LEVELDB_INCLUDE_PATH) OR (NOT LEVELDB_LIB))
    carbin_error("Fail to find leveldb")
endif()
include_directories(${LEVELDB_INCLUDE_PATH})