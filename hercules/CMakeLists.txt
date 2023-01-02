################################################################
#
# Copyright (c) 2022, liyinbin
# All rights reserved.
# Author by liyibin (jeff.li)
#
#################################################################

FILE(GLOB_RECURSE ENGINE_SRC "engine/*.cc")


SET(HERCULES_ENGINE_SRC
        ${ENGINE_SRC}
        )

carbin_cc_library(
        NAMESPACE hercules
        NAME engine
        SOURCES ${HERCULES_ENGINE_SRC}
        PUBLIC_LINKED_TARGETS ${CAEBIN_SYS_DYLINK} ${DYNAMIC_LIB}
        PRIVATE_LINKED_TARGETS hercules::proto
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
        SHARED
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/hercules
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hercules
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        )
