# Check if the processor is an ARM and if Neon instruction are available on the machine where
# the project is compiled.

IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
    EXEC_PROGRAM(cat ARGS "/proc/cpuinfo" OUTPUT_VARIABLE CPUINFO)

    #neon instruction can be found on the majority part of modern ARM processor
    STRING(REGEX REPLACE "^.*(neon).*$" "\\1" NEON_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "neon" "${NEON_THERE}" NEON_TRUE)
    IF (NEON_TRUE)
        set(NEON_FOUND true  BOOL "NEON available on host")
    ELSE ()
        set(NEON_FOUND false  BOOL "NEON available on host")
    ENDIF ()

    #Find the processor type (for now OMAP3 or OMAP4)
    STRING(REGEX REPLACE "^.*(OMAP3).*$" "\\1" OMAP3_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "OMAP3" "${OMAP3_THERE}" OMAP3_TRUE)
    IF (OMAP3_TRUE)
        set(CORTEXA8_FOUND true  BOOL "OMAP3 available on host")
    ELSE ()
        set(CORTEXA8_FOUND false  BOOL "OMAP3 available on host")
    ENDIF ()

    #Find the processor type (for now OMAP3 or OMAP4)
    STRING(REGEX REPLACE "^.*(OMAP4).*$" "\\1" OMAP4_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "OMAP4" "${OMAP4_THERE}" OMAP4_TRUE)
    IF (OMAP4_TRUE)
        set(CORTEXA9_FOUND true  BOOL "OMAP4 available on host")
    ELSE ()
        set(CORTEXA9_FOUND false  BOOL "OMAP4 available on host")
    ENDIF ()

ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    EXEC_PROGRAM("/usr/sbin/sysctl -n machdep.cpu.features" OUTPUT_VARIABLE
            CPUINFO)

    #neon instruction can be found on the majority part of modern ARM processor
    STRING(REGEX REPLACE "^.*(neon).*$" "\\1" NEON_THERE ${CPUINFO})
    STRING(COMPARE EQUAL "neon" "${NEON_THERE}" NEON_TRUE)
    IF (NEON_TRUE)
        set(NEON_FOUND true  BOOL "NEON available on host")
    ELSE ()
        set(NEON_FOUND false  BOOL "NEON available on host")
    ENDIF ()

ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # TODO
    set(CORTEXA8_FOUND   false  BOOL "OMAP3 not available on host")
    set(CORTEXA9_FOUND   false  BOOL "OMAP4 not available on host")
    set(NEON_FOUND   false  BOOL "NEON not available on host")
ELSE(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CORTEXA8_FOUND   false  BOOL "OMAP3 not available on host")
    set(CORTEXA9_FOUND   false  BOOL "OMAP4 not available on host")
    set(NEON_FOUND   false  BOOL "NEON not available on host")
ENDIF()

if(NOT NEON_FOUND)
    MESSAGE(STATUS "Could not find hardware support for NEON on this machine.")
endif()
if(NOT CORTEXA8_FOUND)
    MESSAGE(STATUS "No OMAP3 processor on this on this machine.")
endif()
if(NOT CORTEXA9_FOUND)
    MESSAGE(STATUS "No OMAP4 processor on this on this machine.")
endif()
mark_as_advanced(NEON_FOUND)