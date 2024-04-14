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

#include "turbo/system/command.h"
#include "turbo/base/internal/raw_logging.h"
#include <stdio.h>
#include "turbo/flags/flag.h"
#include "turbo/platform/port.h"
#include "turbo/system/io.h"
#include "turbo/status/error.h"  // errno
#if defined(TURBO_PLATFORM_LINUX)
// clone is a linux specific syscall
#include <sched.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

TURBO_FLAG(bool, run_command_through_clone, false, "Run command with clone syscall to avoid the costly page table duplication");

namespace turbo {

#if defined(TURBO_PLATFORM_LINUX)

    const int CHILD_STACK_SIZE = 256 * 1024;

struct ChildArgs {
    const char* cmd;
    int pipe_fd0;
    int pipe_fd1;
};

int launch_child_process(void* args) {
    ChildArgs* cargs = (ChildArgs*)args;
    dup2(cargs->pipe_fd1, STDOUT_FILENO);
    close(cargs->pipe_fd0);
    close(cargs->pipe_fd1);
    execl("/bin/sh", "sh", "-c", cargs->cmd, nullptr);
    _exit(1);
}

int read_command_output_through_clone(std::ostream& os, const char* cmd) {
    int pipe_fd[2];
    if (pipe(pipe_fd) != 0) {
        TURBO_RAW_LOG(ERROR, "Fail to pipe");
        return -1;
    }
    int saved_errno = 0;
    int wstatus = 0;
    pid_t cpid;
    int rc = 0;
    ChildArgs args = { cmd, pipe_fd[0], pipe_fd[1] };
    char buffer[1024];

    char* child_stack = nullptr;
    char* child_stack_mem = (char*)malloc(CHILD_STACK_SIZE);
    if (!child_stack_mem) {
        TURBO_RAW_LOG(ERROR, "Fail to alloc stack for the child process");
        rc = -1;
        goto END;
    }
    child_stack = child_stack_mem + CHILD_STACK_SIZE;
                               // ^ Assume stack grows downward
    cpid = clone(launch_child_process, child_stack,
                 __WCLONE | CLONE_VM | SIGCHLD | CLONE_UNTRACED, &args);
    if (cpid < 0) {
        TURBO_RAW_LOG(ERROR, "Fail to clone child process");
        rc = -1;
        goto END;
    }
    close(pipe_fd[1]);
    pipe_fd[1] = -1;

    for (;;) {
        const ssize_t nr = read(pipe_fd[0], buffer, sizeof(buffer));
        if (nr > 0) {
            os.write(buffer, nr);
            continue;
        } else if (nr == 0) {
            break;
        } else if (errno != EINTR) {
            TURBO_RAW_LOG(ERROR, "Encountered error while reading for the pipe");
            break;
        }
    }

    close(pipe_fd[0]);
    pipe_fd[0] = -1;

    for (;;) {
        pid_t wpid = waitpid(cpid, &wstatus, WNOHANG | __WALL);
        if (wpid > 0) {
            break;
        }
        if (wpid == 0) {
                usleep(1000);
            continue;
        }
        rc = -1;
        goto END;
    }

    if (WIFEXITED(wstatus)) {
        rc = WEXITSTATUS(wstatus);
        goto END;
    }

    if (WIFSIGNALED(wstatus)) {
        os << "Child process(" << cpid << ") was killed by signal "
           << WTERMSIG(wstatus);
    }

    rc = -1;
    errno = ECHILD;

END:
    saved_errno = errno;
    if (child_stack_mem) {
        free(child_stack_mem);
    }
    if (pipe_fd[0] >= 0) {
        close(pipe_fd[0]);
    }
    if (pipe_fd[1] >= 0) {
        close(pipe_fd[1]);
    }
    errno = saved_errno;
    return rc;
}

#endif // TURBO_PLATFORM_LINUX

    int read_command_output_through_popen(std::ostream& os, const char* cmd) {
        FILE* pipe = ::popen(cmd, "r");
        if (pipe == nullptr) {
            return -1;
        }
        char buffer[1024];
        for (;;) {
            size_t nr = ::fread(buffer, 1, sizeof(buffer), pipe);
            if (nr != 0) {
                os.write(buffer, nr);
            }
            if (nr != sizeof(buffer)) {
                if (feof(pipe)) {
                    break;
                } else if (::ferror(pipe)) {
                    TURBO_RAW_LOG(ERROR, "Encountered error while reading for the pipe");
                    break;
                }
                // retry;
            }
        }

        const int wstatus = pclose(pipe);

        if (wstatus < 0) {
            return wstatus;
        }
        if (WIFEXITED(wstatus)) {
            return WEXITSTATUS(wstatus);
        }
        if (WIFSIGNALED(wstatus)) {
            os << "Child process was killed by signal "
               << WTERMSIG(wstatus);
        }
        errno = ECHILD;
        return -1;
    }

    int read_command_output(std::ostream& os, const char* cmd) {
#if !defined(TURBO_PLATFORM_LINUX)
        return read_command_output_through_popen(os, cmd);
#else
        return get_flag(FLAGS_run_command_through_clone)
        ? read_command_output_through_clone(os, cmd)
        : read_command_output_through_popen(os, cmd);
#endif
    }


    bool self_command_line(std::string &cmd, bool with_args, size_t max_len) {
#if defined(TURBO_PLATFORM_LINUX)
        FDGuard fd(open("/proc/self/cmdline", O_RDONLY));
    if (fd < 0) {
        TURBO_RAW_LOG(ERROR, "Fail to open /proc/self/cmdline");
        return false;
    }
    auto ns = file_size(fd);
    if (ns < 0) {
        TURBO_RAW_LOG(ERROR, "Fail to get file size of /proc/self/cmdline");
        return false;
    }
    if (ns == 0) {
        TURBO_RAW_LOG(ERROR, "File size of /proc/self/cmdline is 0");
        return false;
    }

    if(ns > max_len) {
        TURBO_RAW_LOG(ERROR, "File size of /proc/self/cmdline is too large");
        return false;
    }
    cmd.resize(ns);

    ssize_t nr = ::read(fd, &cmd[0], ns);
    if (nr <= 0) {
        TURBO_RAW_LOG(ERROR, "Fail to read /proc/self/cmdline");
        return false;
    }
#elif defined(TURBO_PLATFORM_OSX)
        static pid_t pid = getpid();
    std::ostringstream oss;
    char cmdbuf[32];
    snprintf(cmdbuf, sizeof(cmdbuf), "ps -p %ld -o command=", (long)pid);
    if (read_command_output(oss, cmdbuf) != 0) {
        LOG(ERROR) << "Fail to read cmdline";
        return false;
    }
    cmd = oss.str();
#else
#error Not Implemented
#endif

        if (with_args) {
            for (ssize_t i = 0; i < cmd.size(); ++i) {
                if (cmd[i] == '\0') {
                    cmd[i] = '\n';
                }
            }
            return true;
        } else {
            for (ssize_t i = 0; i < cmd.size(); ++i) {
                // The command in macos is separated with space and ended with '\n'
                if (cmd[i] == '\0' || cmd[i] == '\n' || cmd[i] == ' ') {
                    cmd.resize(i);
                    return true;
                }
            }
            return true;
        }
    }

}  // namespace turbo
