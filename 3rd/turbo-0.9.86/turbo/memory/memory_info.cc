// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "turbo/memory/memory_info.h"

#ifdef TURBO_PLATFORM_WINDOWS
#include <windows.h>
#include <psapi.h>
#undef min
#undef max
#include <windows.h>
#include <Psapi.h>
#undef min
#undef max
#else

#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/resource.h>

#if defined(TURBO_PLATFORM_LINUX)

#include <sys/types.h>
#include <sys/sysinfo.h>

#elif defined(TURBO_PLATFORM_FREEBSD)
#include <paths.h>
#include <fcntl.h>
#include <kvm.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <sys/user.h>
#elif defined(TURBO_PLATFORM_OSX)
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#include <mach/vm_map.h>
#endif
#endif

namespace turbo {

    uint64_t get_system_memory() {
#ifdef TURBO_PLATFORM_WINDOWS
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return static_cast<uint64_t>(memInfo.ullTotalPageFile);
#elif defined(TURBO_PLATFORM_LINUX)
        struct sysinfo memInfo;
        sysinfo (&memInfo);
        auto totalVirtualMem = memInfo.totalram;

        totalVirtualMem += memInfo.totalswap;
        totalVirtualMem *= memInfo.mem_unit;
        return static_cast<uint64_t>(totalVirtualMem);
#elif defined(TURBO_PLATFORM_FREEBSD)
        kvm_t *kd;
        u_int pageCnt;
        size_t pageCntLen = sizeof(pageCnt);
        u_int pageSize;
        struct kvm_swap kswap;
        uint64_t totalVirtualMem;

        pageSize = static_cast<u_int>(getpagesize());

        sysctlbyname("vm.stats.vm.v_page_count", &pageCnt, &pageCntLen, NULL, 0);
        totalVirtualMem = pageCnt * pageSize;

        kd = kvm_open(NULL, _PATH_DEVNULL, NULL, O_RDONLY, "kvm_open");
        kvm_getswapinfo(kd, &kswap, 1, 0);
        kvm_close(kd);
        totalVirtualMem += kswap.ksw_total * pageSize;

        return totalVirtualMem;
#else
        return 0;
#endif
    }

    size_t get_total_memory_used() {
#ifdef TURBO_PLATFORM_WINDOWS
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return static_cast<uint64_t>(memInfo.ullTotalPageFile - memInfo.ullAvailPageFile);
#elif defined(TURBO_PLATFORM_LINUX)
        struct sysinfo memInfo;
        sysinfo(&memInfo);
        auto virtualMemUsed = memInfo.totalram - memInfo.freeram;

        virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
        virtualMemUsed *= memInfo.mem_unit;

        return static_cast<uint64_t>(virtualMemUsed);
#elif defined(TURBO_PLATFORM_FREEBSD)
        kvm_t *kd;
        u_int pageSize;
        u_int pageCnt, freeCnt;
        size_t pageCntLen = sizeof(pageCnt);
        size_t freeCntLen = sizeof(freeCnt);
        struct kvm_swap kswap;
        uint64_t virtualMemUsed;

        pageSize = static_cast<u_int>(getpagesize());

        sysctlbyname("vm.stats.vm.v_page_count", &pageCnt, &pageCntLen, NULL, 0);
        sysctlbyname("vm.stats.vm.v_free_count", &freeCnt, &freeCntLen, NULL, 0);
        virtualMemUsed = (pageCnt - freeCnt) * pageSize;

        kd = kvm_open(NULL, _PATH_DEVNULL, NULL, O_RDONLY, "kvm_open");
        kvm_getswapinfo(kd, &kswap, 1, 0);
        kvm_close(kd);
        virtualMemUsed += kswap.ksw_used * pageSize;

        return virtualMemUsed;
#else
        return 0;
#endif
    }

    uint64_t get_process_memory_used() {
#ifdef TURBO_PLATFORM_WINDOWS
        PROCESS_MEMORY_COUNTERS_EX pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc), sizeof(pmc));
        return static_cast<uint64_t>(pmc.PrivateUsage);
#elif defined(TURBO_PLATFORM_LINUX)
        auto parseLine =
                [](char *line) -> int {
                    auto i = strlen(line);

                    while (*line < '0' || *line > '9') {
                        line++;
                    }

                    line[i - 3] = '\0';
                    i = atoi(line);
                    return i;
                };

        auto file = fopen("/proc/self/status", "r");
        auto result = -1;
        char line[128];

        while (fgets(line, 128, file) != nullptr) {
            if (strncmp(line, "VmSize:", 7) == 0) {
                result = parseLine(line);
                break;
            }
        }

        fclose(file);
        return static_cast<uint64_t>(result) * 1024;
#elif defined(TURBO_PLATFORM_FREEBSD)
        struct kinfo_proc info;
        size_t infoLen = sizeof(info);
        int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid() };

        sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &infoLen, NULL, 0);
        return static_cast<uint64_t>(info.ki_rssize * getpagesize());
#else
        return 0;
#endif
    }

    size_t get_physical_memory() {
#ifdef TURBO_PLATFORM_WINDOWS
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&memInfo);
        return static_cast<uint64_t>(memInfo.ullTotalPhys);
#elif defined(TURBO_PLATFORM_LINUX)
        struct sysinfo memInfo;
        sysinfo(&memInfo);

        auto totalPhysMem = memInfo.totalram;

        totalPhysMem *= memInfo.mem_unit;
        return static_cast<uint64_t>(totalPhysMem);
#elif defined(TURBO_PLATFORM_FREEBSD)
        u_long physMem;
        size_t physMemLen = sizeof(physMem);
        int mib[] = { CTL_HW, HW_PHYSMEM };

        sysctl(mib, sizeof(mib) / sizeof(*mib), &physMem, &physMemLen, NULL, 0);
        return physMem;
#else
        return 0;
#endif
    }

    size_t get_page_size_impl() {
#ifdef _WIN32
        SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return std::max(system_info.dwPageSize, system_info.dwAllocationGranularity);
#elif defined(__wasm__) || defined(__asmjs__)
        return getpagesize();
#else
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    }

    size_t get_page_size() {
        static const size_t page_size = get_page_size_impl();
        return page_size;
    }

    size_t get_peak_memory_used() {
#if defined(TURBO_PLATFORM_WINDOWS)
        /* Windows -------------------------------------------------- */
        PROCESS_MEMORY_COUNTERS info;
        GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
        return (size_t)info.PeakWorkingSetSize;
#elif defined(TURBO_PLATFORM_UNIX) || defined(TURBO_PLATFORM_APPLE) || defined(TURBO_PLATFORM_LINUX)
        /* BSD, Linux, and OSX -------------------------------------- */
        struct rusage rusage;
        getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
        return (size_t)rusage.ru_maxrss;
#else
        return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
        /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;			/* Unsupported. */
#endif
    }

    size_t get_current_memory_used() {
#if defined(TURBO_PLATFORM_WINDOWS)
        /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.WorkingSetSize;

#elif defined(TURBO_PLATFORM_OSX)
        /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount ) != KERN_SUCCESS )
        return (size_t)0L;		/* Can't access? */
    return (size_t)info.resident_size;

#elif defined(TURBO_PLATFORM_LINUX)
        /* Linux ---------------------------------------------------- */
        long rss = 0L;
        FILE *fp = nullptr;
        if ((fp = fopen("/proc/self/statm", "r")) == nullptr)
            return (size_t) 0L;        /* Can't open? */
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            fclose(fp);
            return (size_t) 0L;        /* Can't read? */
        }
        fclose(fp);
        return (size_t) rss * (size_t) get_page_size();

#else
        /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
        return (size_t)0L;			/* Unsupported. */
#endif
    }

}  // namespace turbo
