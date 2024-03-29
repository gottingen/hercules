
set(CARBIN_COMPILER_NAME "")
set(CARBIN_SYSTEM_VERSION_NAME "")
set(CARBIN_LINUX_NAME "Unknown")

#For Linux, the folowing varaible will be set
#CARBIN_LINUX_NAME		= Fedora, CentOS, RedHat, Ubuntu, openSUSE
#LINUX_VER		= 12 (Fedora), 5.0 (Debian, CentOS)
#RPM_SYSTEM_NAME	= fc12, rhel5, ..
#BIT_MODE		= 32 | 64
#CPACK_DEBIAN_PACKAGE_ARCHITECTURE	= i386, amd64, ...  (value from dpkg utility)

if(WIN32)
    # information taken from
    # http://www.codeguru.com/cpp/w-p/system/systeminformation/article.php/c8973/
    # Win9x series
    if(CMAKE_SYSTEM_VERSION MATCHES "4.0")
        set(CARBIN_SYSTEM_VERSION_NAME "Win95")
    endif(CMAKE_SYSTEM_VERSION MATCHES "4.0")
    if(CMAKE_SYSTEM_VERSION MATCHES "4.10")
        set(CARBIN_SYSTEM_VERSION_NAME "Win98")
    endif(CMAKE_SYSTEM_VERSION MATCHES "4.10")
    if(CMAKE_SYSTEM_VERSION MATCHES "4.90")
        set(CARBIN_SYSTEM_VERSION_NAME "WinME")
    endif(CMAKE_SYSTEM_VERSION MATCHES "4.90")

    # WinNTyyy series
    if(CMAKE_SYSTEM_VERSION MATCHES "3.0")
        set(CARBIN_SYSTEM_VERSION_NAME "WinNT351")
    endif(CMAKE_SYSTEM_VERSION MATCHES "3.0")
    if(CMAKE_SYSTEM_VERSION MATCHES "4.1")
        set(CARBIN_SYSTEM_VERSION_NAME "WinNT4")
    endif(CMAKE_SYSTEM_VERSION MATCHES "4.1")

    # Win2000/XP series
    if(CMAKE_SYSTEM_VERSION MATCHES "5.0")
        set(CARBIN_SYSTEM_VERSION_NAME "Win2000")
    endif(CMAKE_SYSTEM_VERSION MATCHES "5.0")
    if(CMAKE_SYSTEM_VERSION MATCHES "5.1")
        set(CARBIN_SYSTEM_VERSION_NAME "WinXP")
    endif(CMAKE_SYSTEM_VERSION MATCHES "5.1")
    if(CMAKE_SYSTEM_VERSION MATCHES "5.2")
        set(CARBIN_SYSTEM_VERSION_NAME "Win2003")
    endif(CMAKE_SYSTEM_VERSION MATCHES "5.2")

    # WinVista/7 series
    if(CMAKE_SYSTEM_VERSION MATCHES "6.0")
        set(CARBIN_SYSTEM_VERSION_NAME "WinVISTA")
    endif(CMAKE_SYSTEM_VERSION MATCHES "6.0")
    if(CMAKE_SYSTEM_VERSION MATCHES "6.1")
        set(CARBIN_SYSTEM_VERSION_NAME "Win7")
    endif(CMAKE_SYSTEM_VERSION MATCHES "6.1")

    # Compilers
    # taken from http://predef.sourceforge.net/precomp.html#sec34
    IF (MSVC)
        if(MSVC_VERSION EQUAL 1200)
            set(CARBIN_COMPILER_NAME "MSVC-6.0")
        endif(MSVC_VERSION EQUAL 1200)
        if(MSVC_VERSION EQUAL 1300)
            set(CARBIN_COMPILER_NAME "MSVC-7.0")
        endif(MSVC_VERSION EQUAL 1300)
        if(MSVC_VERSION EQUAL 1310)
            set(CARBIN_COMPILER_NAME "MSVC-7.1-2003") #Visual Studio 2003
        endif(MSVC_VERSION EQUAL 1310)
        if(MSVC_VERSION EQUAL 1400)
            set(CARBIN_COMPILER_NAME "MSVC-8.0-2005") #Visual Studio 2005
        endif(MSVC_VERSION EQUAL 1400)
        if(MSVC_VERSION EQUAL 1500)
            set(CARBIN_COMPILER_NAME "MSVC-9.0-2008") #Visual Studio 2008
        endif(MSVC_VERSION EQUAL 1500)
    endif(MSVC)
    IF (MINGW)
        set(CARBIN_COMPILER_NAME "MinGW")
    endif(MINGW)
    IF (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(CARBIN_SYSTEM_VERSION_NAME "${CARBIN_SYSTEM_VERSION_NAME}-x86_64")
    endif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
endif(WIN32)

if(UNIX)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(CARBIN_SYSTEM_VERSION_NAME "${CMAKE_SYSTEM_NAME}")
        if(EXISTS "/etc/issue")
            set(CARBIN_LINUX_NAME "")
            file(READ "/etc/issue" LINUX_ISSUE)
            # Fedora case
            if(LINUX_ISSUE MATCHES "Fedora")
                string(REGEX MATCH "release ([0-9]+)" FEDORA "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "Fedora")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "fc${CMAKE_MATCH_1}")
                set(RPM_SYSTEM_NAME "${SYSTEM_NAME}")
            endif(LINUX_ISSUE MATCHES "Fedora")
            # Scientific Linux case
            # Scientific Linux SL release 5.5 (Boron)
            if(LINUX_ISSUE MATCHES "Scientific Linux")
                string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" CENTOS "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "ScientificLinux")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "sl${CMAKE_MATCH_1}")
                set(RPM_SYSTEM_NAME "${SYSTEM_NAME}")
            endif(LINUX_ISSUE MATCHES "Scientific Linux")
            # CentOS case
            # CentOS release 5.5 (Final)
            if(LINUX_ISSUE MATCHES "CentOS")
                string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" CENTOS "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "CentOS")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "centos${CMAKE_MATCH_1}")
                set(å "${SYSTEM_NAME}")
            endif(LINUX_ISSUE MATCHES "CentOS")
            # Redhat case
            # Red Hat Enterprise Linux Server release 5 (Tikanga)
            if(LINUX_ISSUE MATCHES "Red Hat")
                string(REGEX MATCH "release ([0-9]+\\.*[0-9]*)" REDHAT "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "RedHat")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "rhel${CMAKE_MATCH_1}")
                set(RPM_SYSTEM_NAME "${SYSTEM_NAME}")
            endif(LINUX_ISSUE MATCHES "Red Hat")
            # Ubuntu case
            if(LINUX_ISSUE MATCHES "Ubuntu")
                string(REGEX MATCH "buntu ([0-9]+\\.[0-9]+)" UBUNTU "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "Ubuntu")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "Ubuntu-${CMAKE_MATCH_1}")
                set(DEB_SYSTEM_NAME "ubuntu_${CMAKE_MATCH_1}")
            endif(LINUX_ISSUE MATCHES "Ubuntu")
            # Debian case
            if(LINUX_ISSUE MATCHES "Debian")
                string(REGEX MATCH "Debian .*ux ([0-9]+\\.[0-9]+)" DEBIAN "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "Debian")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "Debian-${CMAKE_MATCH_1}")
                set(DEB_SYSTEM_NAME "deb_${CMAKE_MATCH_1}")
            endif(LINUX_ISSUE MATCHES "Debian")
            # SuSE / openSUSE case
            if(LINUX_ISSUE MATCHES "openSUSE")
                string(REGEX MATCH "openSUSE ([0-9]+\\.[0-9]+)" OPENSUSE "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "openSUSE")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "opensuse_${CMAKE_MATCH_1}")
                set(RPM_SYSTEM_NAME "${SYSTEM_NAME}")
                if (LINUX_VER MATCHES "/")
                    string(REPLACE "/" "_" LINUX_VER ${LINUX_VER})
                endif()
            elseif(LINUX_ISSUE MATCHES "SUSE")
                string(REGEX MATCH "Server ([0-9]+)" SUSE "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "sles")
                set(LINUX_VER "${CMAKE_MATCH_1}")
                set(SYSTEM_NAME "sles${CMAKE_MATCH_1}")
                set(RPM_SYSTEM_NAME "${SYSTEM_NAME}")
                if (LINUX_VER MATCHES "/")
                    string(REPLACE "/" "_" LINUX_VER ${LINUX_VER})
                endif()
            endif()
            # Guess at the name for other Linux distros
            if(NOT SYSTEM_NAME)
                string(REGEX MATCH "([^ ]+) [^0-9]*([0-9.]+)" DISTRO "${LINUX_ISSUE}")
                set(CARBIN_LINUX_NAME "${CMAKE_MATCH_1}")
                set(LINUX_VER "${CMAKE_MATCH_2}")
                set(SYSTEM_NAME "${CARBIN_LINUX_NAME}")
                if(EXISTS "/etc/debian_version")
                    set(DEB_SYSTEM_NAME "${CARBIN_LINUX_NAME}")
                endif()
            endif()

            #Find CPU Arch
            if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "64")
                set ( BIT_MODE "64")
            else()
                set ( BIT_MODE "32")
            endif ()

            #Find CPU Arch for Debian system
            if ((CARBIN_LINUX_NAME STREQUAL "Debian") OR (CARBIN_LINUX_NAME STREQUAL "Ubuntu"))

                # There is no such thing as i686 architecture on debian, you should use i386 instead
                # $ dpkg --print-architecture
                FIND_PROGRAM(DPKG_CMD dpkg)
                IF(NOT DPKG_CMD)
                    # Cannot find dpkg in your path, default to i386
                    # Try best guess
                    if (BIT_MODE STREQUAL "32")
                        SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE i386)
                    elseif (BIT_MODE STREQUAL "64")
                        SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
                    endif()
                ENDIF(NOT DPKG_CMD)
                EXECUTE_PROCESS(COMMAND "${DPKG_CMD}" --print-architecture
                        OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        )
            endif ()

            #Find Codename for Debian system
            if ((CARBIN_LINUX_NAME STREQUAL "Debian") OR (CARBIN_LINUX_NAME STREQUAL "Ubuntu"))
                # $ lsb_release -cs
                FIND_PROGRAM(LSB_CMD lsb_release)
                IF(NOT LSB_CMD)
                    # Cannot find lsb_release in your path, default to none
                    SET(DEBIAN_CODENAME "")
                ENDIF(NOT LSB_CMD)
                EXECUTE_PROCESS(COMMAND "${LSB_CMD}" -cs
                        OUTPUT_VARIABLE DEBIAN_CODENAME
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        )
            endif ()

            if(CARBIN_LINUX_NAME)
                set(CARBIN_SYSTEM_VERSION_NAME "${CMAKE_SYSTEM_NAME}-${CARBIN_LINUX_NAME}-${LINUX_VER}")
            else()
                set(CARBIN_LINUX_NAME "NOT-FOUND")
            endif(CARBIN_LINUX_NAME)
        endif(EXISTS "/etc/issue")

    elseif(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
        string(REGEX MATCH "(([0-9]+)\\.([0-9]+))-RELEASE" FREEBSD "${CMAKE_SYSTEM_VERSION}")
        set( FREEBSD_RELEASE "${CMAKE_MATCH_1}" )
        set( FREEBSD_MAJOR "${CMAKE_MATCH_2}" )
        set( FREEBSD_MINOR "${CMAKE_MATCH_3}" )
        set( FREEBSD_VERSION "${CMAKE_SYSTEM_VERSION}" )
        set( SYSTEM_NAME "freebsd_${FREEBSD_RELEASE}" )
        set( CONDOR_FREEBSD ON )
        set( BSD_UNIX ON )
        if(FREEBSD_MAJOR MATCHES "4" )
            set( CONDOR_FREEBSD4 ON )
        elseif(FREEBSD_MAJOR MATCHES "5" )
            set( CONDOR_FREEBSD5 ON )
        elseif(FREEBSD_MAJOR MATCHES "6" )
            set( CONDOR_FREEBSD6 ON )
        elseif(FREEBSD_MAJOR MATCHES "7" )
            set( CONDOR_FREEBSD7 ON )
        elseif(FREEBSD_MAJOR MATCHES "8" )
            set( CONDOR_FREEBSD8 ON )
        endif()
        if( CMAKE_SYSTEM_PROCESSOR MATCHES "amd64" )
            set( SYS_ARCH "x86_64")
        elseif( CMAKE_SYSTEM_PROCESSOR MATCHES "i386" )
            set( SYS_ARCH "x86")
        endif( )
        set( PLATFORM "${SYS_ARCH}_freebsd_${FREEBSD_RELEASE}")

    elseif(OS_NAME MATCHES "DARWIN")
        set( BSD_UNIX ON )

    endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

    set(CARBIN_SYSTEM_VERSION_NAME "${CARBIN_SYSTEM_VERSION_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
    set(CARBIN_COMPILER_NAME "")

endif(UNIX)