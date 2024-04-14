// Copyright 2023 The Turbo Authors.
//
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

#include "includes.h"
#include "turbo/testing/test.h"
#ifdef _WIN32
#    include <windows.h>
#else
#    include <sys/types.h>
#    include <dirent.h>
#endif

void prepare_logdir()
{
    turbo::tlog::drop_all();
#ifdef _WIN32
    system("rmdir /S /Q test_logs");
#else
    auto rv = system("rm -rf test_logs");
    if (rv != 0)
    {
        throw std::runtime_error("Failed to rm -rf test_logs");
    }
#endif
}

std::string file_contents(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios_base::binary);
    if (!ifs)
    {
        throw std::runtime_error("Failed open file ");
    }
    return std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
}

std::size_t count_lines(const std::string &filename, bool dump)
{
    std::ifstream ifs(filename);
    if (!ifs)
    {
        throw std::runtime_error("Failed open file ");
    }

    std::string line;
    size_t counter = 0;
    while (std::getline(ifs, line)) {
        if(dump) {
            turbo::println(line);
        }
        counter++;
    }
    return counter;
}

void require_message_count(const std::string &filename, const std::size_t messages)
{
    if (strlen(turbo::tlog::details::os::default_eol) == 0)
    {
        REQUIRE(count_lines(filename) == 1);
    }
    else
    {
        REQUIRE(count_lines(filename) == messages);
    }
}

std::size_t get_filesize(const std::string &filename)
{
    std::ifstream ifs(filename, std::ifstream::ate | std::ifstream::binary);
    if (!ifs)
    {
        throw std::runtime_error("Failed open file ");
    }

    return static_cast<std::size_t>(ifs.tellg());
}

// source: https://stackoverflow.com/a/2072890/192001
bool ends_with(std::string const &value, std::string const &ending)
{
    if (ending.size() > value.size())
    {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

#ifdef _WIN32
// Based on: https://stackoverflow.com/a/37416569/192001
std::size_t count_files(const std::string &folder)
{
    size_t counter = 0;
    WIN32_FIND_DATAA ffd;

    // Start iterating over the files in the folder directory.
    HANDLE hFind = ::FindFirstFileA((folder + "\\*").c_str(), &ffd);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do // Managed to locate and create an handle to that folder.
        {
            if (ffd.cFileName[0] != '.')
                counter++;
        } while (::FindNextFileA(hFind, &ffd) != 0);
        ::FindClose(hFind);
    }
    else
    {
        throw std::runtime_error("Failed open folder " + folder);
    }

    return counter;
}
#else
// Based on: https://stackoverflow.com/a/2802255/192001
std::size_t count_files(const std::string &folder)
{
    size_t counter = 0;
    DIR *dp = opendir(folder.c_str());
    if (dp == nullptr)
    {
        throw std::runtime_error("Failed open folder " + folder);
    }

    struct dirent *ep = nullptr;
    while ((ep = readdir(dp)) != nullptr)
    {
        if (ep->d_name[0] != '.')
            counter++;
    }
    (void)closedir(dp);
    return counter;
}
#endif
