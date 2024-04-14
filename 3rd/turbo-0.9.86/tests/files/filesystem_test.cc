//---------------------------------------------------------------------------------------
//
// Copyright (c) 2018, Steffen Schümann <s.schuemann@pobox.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//---------------------------------------------------------------------------------------
#include "turbo/log/logging.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <thread>

#if (defined(WIN32) || defined(_WIN32)) && !defined(__GNUC__)
#define NOMINMAX 1
#endif

#ifdef USE_STD_FS
#include <filesystem>
namespace fs {
using namespace std::filesystem;
using ifstream = std::ifstream;
using ofstream = std::ofstream;
using fstream = std::fstream;
} // namespace fs
#ifdef __GNUC__
#define GCC_VERSION                                                            \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif
#ifdef _MSC_VER
#define IS_WCHAR_PATH
#endif
#ifdef WIN32
#define TURBO_PLATFORM_WINDOWS
#endif
#else
#ifdef TURBO_FILESYSTEM_FWD_TEST
#include "turbo/files/fs_fwd.h"
#else

#include "turbo/files/filesystem.h"

#endif
namespace fs {
    using namespace turbo::filesystem;
    using ifstream = turbo::filesystem::ifstream;
    using ofstream = turbo::filesystem::ofstream;
    using fstream = turbo::filesystem::fstream;
} // namespace fs
#endif

#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#else

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#endif

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//- - - - - - - - -
// Behaviour Switches (should match the config in turbo/filesystem.hpp):
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//- - - - - - - - -
// LWG #2682 disables the since then invalid use of the copy option
// create_symlinks on directories
#define TEST_LWG_2682_BEHAVIOUR
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//- - - - - - - - -
// LWG #2395 makes crate_directory/create_directories not emit an error if there
// is a regular file with that name, it is superceded by P1164R1, so only
// activate if really needed #define TEST_LWG_2935_BEHAVIOUR
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//- - - - - - - - -
// LWG #2937 enforces that fs::equivalent emits an error, if
// !fs::exists(p1)||!exists(p2)
#define TEST_LWG_2937_BEHAVIOUR
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//- - - - - - - - -

template<typename TP>
std::time_t to_time_t(TP tp) {
    using namespace std::chrono;
    auto sctp = time_point_cast<system_clock::duration>(tp - TP::clock::now() +
                                                        system_clock::now());
    return system_clock::to_time_t(sctp);
}

template<typename TP>
TP from_time_t(std::time_t t) {
    using namespace std::chrono;
    auto sctp = system_clock::from_time_t(t);
    auto tp = time_point_cast<typename TP::duration>(sctp - system_clock::now() +
                                                     TP::clock::now());
    return tp;
}

enum class TempOpt {
    none, change_path
};

class TemporaryDirectory {
public:
    TemporaryDirectory(TempOpt opt = TempOpt::none) {
        static auto seed =
                std::chrono::high_resolution_clock::now().time_since_epoch().count();
        static auto rng =
                std::bind(std::uniform_int_distribution<int>(0, 35),
                          std::mt19937(static_cast<unsigned int>(seed) ^
                                       static_cast<unsigned int>(
                                               reinterpret_cast<ptrdiff_t>(&opt))));
        std::string filename;
        do {
            filename = "test_";
            for (int i = 0; i < 8; ++i) {
                filename += "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[rng()];
            }
            _path = fs::canonical(fs::temp_directory_path()) / filename;
        } while (fs::exists(_path));
        fs::create_directories(_path);
        if (opt == TempOpt::change_path) {
            _orig_dir = fs::current_path();
            fs::current_path(_path);
        }
    }

    ~TemporaryDirectory() {
        if (!_orig_dir.empty()) {
            fs::current_path(_orig_dir);
        }
        fs::remove_all(_path);
    }

    const fs::path &path() const { return _path; }

private:
    fs::path _path;
    fs::path _orig_dir;
};

static void generateFile(const fs::path &pathname, int withSize = -1) {
    fs::ofstream outfile(pathname);
    if (withSize < 0) {
        outfile << "Hello world!" << std::endl;
    } else {
        outfile << std::string(size_t(withSize), '*');
    }
}

#ifdef TURBO_PLATFORM_WINDOWS
inline bool isWow64Proc() {
  typedef BOOL(WINAPI * IsWow64Process_t)(HANDLE, PBOOL);
  BOOL bIsWow64 = FALSE;
  auto fnIsWow64Process = (IsWow64Process_t)GetProcAddress(
      GetModuleHandle(TEXT("kernel32")), "IsWow64Process");
  if (NULL != fnIsWow64Process) {
    if (!fnIsWow64Process(GetCurrentProcess(), &bIsWow64)) {
      bIsWow64 = FALSE;
    }
  }
  return bIsWow64 == TRUE;
}

static bool is_symlink_creation_supported() {
  bool result = true;
  HKEY key;
  REGSAM flags = KEY_READ;
#ifdef _WIN64
  flags |= KEY_WOW64_64KEY;
#elif defined(KEY_WOW64_64KEY)
  if (isWow64Proc()) {
    flags |= KEY_WOW64_64KEY;
  } else {
    flags |= KEY_WOW64_32KEY;
  }
#else
  result = false;
#endif
  if (result) {
    auto err = RegOpenKeyExW(
        HKEY_LOCAL_MACHINE,
        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock", 0,
        flags, &key);
    if (err == ERROR_SUCCESS) {
      DWORD val = 0, size = sizeof(DWORD);
      err = RegQueryValueExW(key, L"AllowDevelopmentWithoutDevLicense", 0, NULL,
                             reinterpret_cast<LPBYTE>(&val), &size);
      RegCloseKey(key);
      if (err != ERROR_SUCCESS) {
        result = false;
      } else {
        result = (val != 0);
      }
    } else {
      result = false;
    }
  }
  if (!result) {
    std::clog << "Warning: Symlink creation not supported." << std::endl;
  }
  return result;
}
#else

static bool is_symlink_creation_supported() { return true; }

#endif

static bool has_host_root_name_support() {
    return fs::path("//host").has_root_name();
}

template<class T>
class TestAllocator {
public:
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using difference_type = ptrdiff_t;
    using size_type = size_t;

    TestAllocator() noexcept {}

    template<class U>
    TestAllocator(TestAllocator<U> const &) noexcept {}

    value_type *allocate(std::size_t n) {
        return static_cast<value_type *>(::operator new(n * sizeof(value_type)));
    }

    void deallocate(value_type *p, std::size_t) noexcept { ::operator delete(p); }

    template<class U>
    struct rebind {
        typedef TestAllocator<U> other;
    };
};

template<class T, class U>
bool operator==(TestAllocator<T> const &, TestAllocator<U> const &) noexcept {
    return true;
}

template<class T, class U>
bool operator!=(TestAllocator<T> const &x, TestAllocator<U> const &y) noexcept {
    return !(x == y);
}

TEST_CASE("TemporaryDirectory, fsTestTempdir") {
    fs::path tempPath;
    {
        TemporaryDirectory t;
        tempPath = t.path();
        REQUIRE(fs::exists(fs::path(t.path())));
        REQUIRE(fs::is_directory(t.path()));
    }
    REQUIRE(!fs::exists(tempPath));
}

#ifdef TURBO_FILESYSTEM_VERSION

TEST_CASE("Filesystem, detail_utf8") {
    REQUIRE(fs::detail::fromUtf8<std::wstring>("foobar").length() == 6);
    REQUIRE(fs::detail::fromUtf8<std::wstring>("foobar") == L"foobar");
    REQUIRE(fs::detail::fromUtf8<std::wstring>(u8"föobar").length() == 6);
    REQUIRE(fs::detail::fromUtf8<std::wstring>(u8"föobar") == L"föobar");

    REQUIRE(fs::detail::toUtf8(std::wstring(L"foobar")).length() == 6);
    REQUIRE(fs::detail::toUtf8(std::wstring(L"foobar")) == "foobar");
    REQUIRE(fs::detail::toUtf8(std::wstring(L"föobar")).length() == 7);
    // REQUIRE(fs::detail::toUtf8(std::wstring(L"föobar")) == u8"föobar");

#ifdef TURBO_RAISE_UNICODE_ERRORS
    REQUIRE_THROWS_AS(
        fs::detail::fromUtf8<std::u16string>(std::string("\xed\xa0\x80")),
        fs::filesystem_error);
    REQUIRE_THROWS_AS(fs::detail::fromUtf8<std::u16string>(std::string("\xc3")),
                 fs::filesystem_error);
#else
    REQUIRE(std::u16string(2, 0xfffd) == fs::detail::fromUtf8<std::u16string>(
            std::string("\xed\xa0\x80")));
    REQUIRE(std::u16string(1, 0xfffd) ==
            fs::detail::fromUtf8<std::u16string>(std::string("\xc3")));
#endif
}

TEST_CASE("fs_utf, detail_utf8") {
    std::string t;
    REQUIRE(std::string("\xc3\xa4/\xe2\x82\xac\xf0\x9d\x84\x9e") ==
            fs::detail::toUtf8(std::u16string(u"\u00E4/\u20AC\U0001D11E")));
#ifdef TURBO_RAISE_UNICODE_ERRORS
    REQUIRE_THROWS_AS(fs::detail::toUtf8(std::u16string(1, 0xd800)),
                 fs::filesystem_error);
    REQUIRE_THROWS_AS(fs::detail::appendUTF8(t, 0x200000), fs::filesystem_error);
#else
    REQUIRE(std::string("\xEF\xBF\xBD") ==
            fs::detail::toUtf8(std::u16string(1, 0xd800)));
    fs::detail::appendUTF8(t, 0x200000);
    REQUIRE(std::string("\xEF\xBF\xBD") == t);
#endif
}

#endif

TEST_CASE("Filesystem, generic") {
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path::preferred_separator == '\\');
#else
    REQUIRE(fs::path::preferred_separator == '/');
#endif
}

#ifndef TURBO_PLATFORM_WINDOWS

TEST_CASE("Filesystem, path_gen") {
    if (!has_host_root_name_support()) {
        TLOG_WARN("This implementation doesn't support "
                  "path(\"//host\").has_root_name() == true [C++17 "
                  "30.12.8.1 par. 4] on this platform, tests based on "
                  "this are skipped. (Should be okay.)");
    }
}

#endif

TEST_CASE("Filesystem, construct") {
    REQUIRE("/usr/local/bin" == fs::path("/usr/local/bin").generic_string());
    std::string str = "/usr/local/bin";
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    std::u8string u8str = u8"/usr/local/bin";
#endif
    std::u16string u16str = u"/usr/local/bin";
    std::u32string u32str = U"/usr/local/bin";
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    REQUIRE(u8str == fs::path(u8str).generic_u8string());
#endif
    REQUIRE(u16str == fs::path(u16str).generic_u16string());
    REQUIRE(u32str == fs::path(u32str).generic_u32string());
    REQUIRE(str == fs::path(str, fs::path::format::generic_format));
    REQUIRE(str == fs::path(str.begin(), str.end()));
    REQUIRE(fs::path(std::wstring(3, 67)) == "CCC");
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    REQUIRE(str == fs::path(u8str.begin(), u8str.end()));
#endif
    REQUIRE(str == fs::path(u16str.begin(), u16str.end()));
    REQUIRE(str == fs::path(u32str.begin(), u32str.end()));
#ifdef TURBO_FILESYSTEM_VERSION
    REQUIRE(fs::path("///foo/bar") == "/foo/bar");
    REQUIRE(fs::path("//foo//bar") == "//foo/bar");
#endif
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE("\\usr\\local\\bin" == fs::path("/usr/local/bin"));
    REQUIRE("C:\\usr\\local\\bin" == fs::path("C:\\usr\\local\\bin"));
#else
    REQUIRE("/usr/local/bin" == fs::path("/usr/local/bin"));
#endif
    if (has_host_root_name_support()) {
        REQUIRE("//host/foo/bar" == fs::path("//host/foo/bar"));
    }

#if !defined(TURBO_PLATFORM_WINDOWS) && \
    !(defined(__GLIBCXX__) && \
      !(defined(_GLIBCXX_RELEASE) && (_GLIBCXX_RELEASE >= 8))) && \
    !defined(USE_STD_FS)
    std::locale loc;
    bool testUTF8Locale = false;
    try {
        if (const char *lang = std::getenv("LANG")) {
            loc = std::locale(lang);
        } else {
            loc = std::locale("en_US.UTF-8");
        }
        std::string name = loc.name();
        if (name.length() > 5 && (name.substr(name.length() - 5) == "UTF-8" ||
                                  name.substr(name.length() - 5) == "utf-8")) {
            testUTF8Locale = true;
        }
    } catch (std::runtime_error &) {
        TLOG_WARN("Couldn't create an UTF-8 locale!");
    }
    if (testUTF8Locale) {
        REQUIRE("/usr/local/bin" == fs::path("/usr/local/bin", loc));
        REQUIRE(str == fs::path(str.begin(), str.end(), loc));
        REQUIRE(str == fs::path(u16str.begin(), u16str.end(), loc));
        REQUIRE(str == fs::path(u32str.begin(), u32str.end(), loc));
    }
#endif
}

TEST_CASE("FilesystemPath, assign") {
    fs::path p1{"/foo/bar"};
    fs::path p2{"/usr/local"};
    fs::path p3;
    p3 = p1;
    REQUIRE(p1 == p3);
    p3 = fs::path{"/usr/local"};
    REQUIRE(p2 == p3);
    p3 = fs::path{L"/usr/local"};
    REQUIRE(p2 == p3);
    p3.assign(L"/usr/local");
    REQUIRE(p2 == p3);
#if defined(IS_WCHAR_PATH) || defined(GHC_USE_WCHAR_T)
    p3 = fs::path::string_type{L"/foo/bar"};
    REQUIRE(p1 == p3);
    p3.assign(fs::path::string_type{L"/usr/local"});
    REQUIRE(p2 == p3);
#else
    p3 = fs::path::string_type{"/foo/bar"};
    REQUIRE(p1 == p3);
    p3.assign(fs::path::string_type{"/usr/local"});
    REQUIRE(p2 == p3);
#endif
    p3 = std::u16string(u"/foo/bar");
    REQUIRE(p1 == p3);
    p3 = U"/usr/local";
    REQUIRE(p2 == p3);
    p3.assign(std::u16string(u"/foo/bar"));
    REQUIRE(p1 == p3);
    std::string s{"/usr/local"};
    p3.assign(s.begin(), s.end());
    REQUIRE(p2 == p3);
}

TEST_CASE("FilesystemPath, append") {
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("foo") / "c:/bar" == "c:/bar");
    REQUIRE(fs::path("foo") / "c:" == "c:");
    REQUIRE(fs::path("c:") / "" == "c:");
    REQUIRE(fs::path("c:foo") / "/bar" == "c:/bar");
    REQUIRE(fs::path("c:foo") / "c:bar" == "c:foo/bar");
#else
    REQUIRE(fs::path("foo") / "" == "foo/");
    REQUIRE(fs::path("foo") / "/bar" == "/bar");
    REQUIRE(fs::path("/foo") / "/" == "/");
    if (has_host_root_name_support()) {
        REQUIRE(fs::path("//host/foo") / "/bar" == "/bar");
        REQUIRE(fs::path("//host") / "/" == "//host/");
        REQUIRE(fs::path("//host/foo") / "/" == "/");
    }
#endif
    REQUIRE(fs::path("/foo/bar") / "some///other" == "/foo/bar/some/other");
    fs::path p1{"/tmp/test"};
    fs::path p2{"foobar.txt"};
    fs::path p3 = p1 / p2;
    REQUIRE("/tmp/test/foobar.txt" == p3);
    // TODO: append(first, last)
}

TEST_CASE("FilesystemPath, concat") {
    REQUIRE((fs::path("foo") += fs::path("bar")) == "foobar");
    REQUIRE((fs::path("foo") += fs::path("/bar")) == "foo/bar");

    REQUIRE((fs::path("foo") += std::string("bar")) == "foobar");
    REQUIRE((fs::path("foo") += std::string("/bar")) == "foo/bar");

    REQUIRE((fs::path("foo") += "bar") == "foobar");
    REQUIRE((fs::path("foo") += "/bar") == "foo/bar");
    REQUIRE((fs::path("foo") += L"bar") == "foobar");
    REQUIRE((fs::path("foo") += L"/bar") == "foo/bar");

    REQUIRE((fs::path("foo") += 'b') == "foob");
    REQUIRE((fs::path("foo") += '/') == "foo/");
    REQUIRE((fs::path("foo") += L'b') == "foob");
    REQUIRE((fs::path("foo") += L'/') == "foo/");

    REQUIRE((fs::path("foo") += std::string("bar")) == "foobar");
    REQUIRE((fs::path("foo") += std::string("/bar")) == "foo/bar");

    REQUIRE((fs::path("foo") += std::u16string(u"bar")) == "foobar");
    REQUIRE((fs::path("foo") += std::u16string(u"/bar")) == "foo/bar");

    REQUIRE((fs::path("foo") += std::u32string(U"bar")) == "foobar");
    REQUIRE((fs::path("foo") += std::u32string(U"/bar")) == "foo/bar");

    REQUIRE(fs::path("foo").concat("bar") == "foobar");
    REQUIRE(fs::path("foo").concat("/bar") == "foo/bar");
    REQUIRE(fs::path("foo").concat(L"bar") == "foobar");
    REQUIRE(fs::path("foo").concat(L"/bar") == "foo/bar");
    std::string bar = "bar";
    REQUIRE(fs::path("foo").concat(bar.begin(), bar.end()) == "foobar");
#ifndef USE_STD_FS
    REQUIRE((fs::path("/foo/bar") += "/some///other") ==
            "/foo/bar/some/other");
#endif
    // TODO: contat(first, last)
}

TEST_CASE("FilesystemPath, modifiers") {
    fs::path p = fs::path("/foo/bar");
    p.clear();
    REQUIRE(p == "");

    // make_preferred() is a no-op
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("foo\\bar") == "foo/bar");
    REQUIRE(fs::path("foo\\bar").make_preferred() == "foo/bar");
#else
    REQUIRE(fs::path("foo\\bar") == "foo\\bar");
    REQUIRE(fs::path("foo\\bar").make_preferred() == "foo\\bar");
#endif
    REQUIRE(fs::path("foo/bar").make_preferred() == "foo/bar");

    REQUIRE(fs::path("foo/bar").remove_filename() == "foo/");
    REQUIRE(fs::path("foo/").remove_filename() == "foo/");
    REQUIRE(fs::path("/foo").remove_filename() == "/");
    REQUIRE(fs::path("/").remove_filename() == "/");

    REQUIRE(fs::path("/foo").replace_filename("bar") == "/bar");
    REQUIRE(fs::path("/").replace_filename("bar") == "/bar");
    REQUIRE(fs::path("/foo").replace_filename("b//ar") == "/b/ar");

    REQUIRE(fs::path("/foo/bar.txt").replace_extension("odf") ==
            "/foo/bar.odf");
    REQUIRE(fs::path("/foo/bar.txt").replace_extension() == "/foo/bar");
    REQUIRE(fs::path("/foo/bar").replace_extension("odf") == "/foo/bar.odf");
    REQUIRE(fs::path("/foo/bar").replace_extension(".odf") == "/foo/bar.odf");
    REQUIRE(fs::path("/foo/bar.").replace_extension(".odf") ==
            "/foo/bar.odf");
    REQUIRE(fs::path("/foo/bar/").replace_extension("odf") ==
            "/foo/bar/.odf");

    fs::path p1 = "foo";
    fs::path p2 = "bar";
    p1.swap(p2);
    REQUIRE(p1 == "bar");
    REQUIRE(p2 == "foo");
}

TEST_CASE("FilesystemPath, obs") {
#ifdef TURBO_PLATFORM_WINDOWS
#if defined(IS_WCHAR_PATH) || defined(GHC_USE_WCHAR_T)
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").native() ==
                fs::path::string_type(L"\u00E4\\\u20AC"));
    // REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").string() ==
    // std::string("ä\\€")); // MSVCs returns local DBCS encoding
#else
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").native() ==
                fs::path::string_type("\xc3\xa4\\\xe2\x82\xac"));
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").string() ==
                std::string("\xc3\xa4\\\xe2\x82\xac"));
    REQUIRE(!::strcmp(fs::u8path("\xc3\xa4\\\xe2\x82\xac").c_str(),
                          "\xc3\xa4\\\xe2\x82\xac"));
    REQUIRE((std::string)fs::u8path("\xc3\xa4\\\xe2\x82\xac") ==
                std::string("\xc3\xa4\\\xe2\x82\xac"));
#endif
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").wstring() ==
                std::wstring(L"\u00E4\\\u20AC"));
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").u8string() ==
                std::u8string(u8"\u00E4\\\u20AC"));
#else
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").u8string() ==
                std::string("\xc3\xa4\\\xe2\x82\xac"));
#endif
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").u16string() ==
                std::u16string(u"\u00E4\\\u20AC"));
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").u32string() ==
                std::u32string(U"\U000000E4\\\U000020AC"));
#else
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").native() ==
            fs::path::string_type("\xc3\xa4/\xe2\x82\xac"));
    REQUIRE(!::strcmp(fs::u8path("\xc3\xa4/\xe2\x82\xac").c_str(),
                      "\xc3\xa4/\xe2\x82\xac"));
    REQUIRE((std::string) fs::u8path("\xc3\xa4/\xe2\x82\xac") ==
            std::string("\xc3\xa4/\xe2\x82\xac"));
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").string() ==
            std::string("\xc3\xa4/\xe2\x82\xac"));
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").wstring() ==
            std::wstring(L"ä/€"));
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").u8string() ==
                std::u8string(u8"\xc3\xa4/\xe2\x82\xac"));
#else
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").u8string() ==
            std::string("\xc3\xa4/\xe2\x82\xac"));
#endif
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").u16string() ==
            std::u16string(u"\u00E4/\u20AC"));
    TLOG_WARN("This check might fail on GCC8 (with \"Illegal byte sequence\") due "
              "to not detecting the valid unicode codepoint U+1D11E.");
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac\xf0\x9d\x84\x9e").u16string() ==
            std::u16string(u"\u00E4/\u20AC\U0001D11E"));
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").u32string() ==
            std::u32string(U"\U000000E4/\U000020AC"));
#endif
}

TEST_CASE("FilesystemPath, generic_obs") {
#ifdef TURBO_PLATFORM_WINDOWS
#ifndef IS_WCHAR_PATH
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").generic_string() ==
                std::string("\xc3\xa4/\xe2\x82\xac"));
#endif
#ifndef USE_STD_FS
    auto t =
        fs::u8path("\xc3\xa4\\\xe2\x82\xac")
            .generic_string<char, std::char_traits<char>, TestAllocator<char>>();
    REQUIRE(t.c_str() == std::string("\xc3\xa4/\xe2\x82\xac"));
#endif
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").generic_wstring() ==
                std::wstring(L"\U000000E4/\U000020AC"));
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").generic_u8string() ==
                std::u8string(u8"\u00E4/\u20AC"));
#else
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").generic_u8string() ==
                std::string("\xc3\xa4/\xe2\x82\xac"));
#endif
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").generic_u16string() ==
                std::u16string(u"\u00E4/\u20AC"));
    REQUIRE(fs::u8path("\xc3\xa4\\\xe2\x82\xac").generic_u32string() ==
                std::u32string(U"\U000000E4/\U000020AC"));
#else
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").generic_string() ==
            std::string("\xc3\xa4/\xe2\x82\xac"));
#ifndef USE_STD_FS
    auto t =
            fs::u8path("\xc3\xa4/\xe2\x82\xac")
                    .generic_string<char, std::char_traits<char>, TestAllocator<char>>();
    REQUIRE(t.c_str() == std::string("\xc3\xa4/\xe2\x82\xac"));
#endif
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").generic_wstring() ==
            std::wstring(L"ä/€"));
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").generic_u8string() ==
                std::u8string(u8"\xc3\xa4/\xe2\x82\xac"));
#else
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").generic_u8string() ==
            std::string("\xc3\xa4/\xe2\x82\xac"));
#endif
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").generic_u16string() ==
            std::u16string(u"\u00E4/\u20AC"));
    REQUIRE(fs::u8path("\xc3\xa4/\xe2\x82\xac").generic_u32string() ==
            std::u32string(U"\U000000E4/\U000020AC"));
#endif
}

TEST_CASE("FilesystemPath, compare") {
    REQUIRE(fs::path("/foo/b").compare("/foo/a") > 0);
    REQUIRE(fs::path("/foo/b").compare("/foo/b") == 0);
    REQUIRE(fs::path("/foo/b").compare("/foo/c") < 0);

    REQUIRE(fs::path("/foo/b").compare(std::string("/foo/a")) > 0);
    REQUIRE(fs::path("/foo/b").compare(std::string("/foo/b")) == 0);
    REQUIRE(fs::path("/foo/b").compare(std::string("/foo/c")) < 0);

    REQUIRE(fs::path("/foo/b").compare(fs::path("/foo/a")) > 0);
    REQUIRE(fs::path("/foo/b").compare(fs::path("/foo/b")) == 0);
    REQUIRE(fs::path("/foo/b").compare(fs::path("/foo/c")) < 0);

#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("c:\\a\\b").compare("C:\\a\\b") == 0);
    REQUIRE(fs::path("c:\\a\\b").compare("d:\\a\\b") != 0);
    REQUIRE(fs::path("c:\\a\\b").compare("C:\\A\\b") != 0);
#endif

#ifdef LWG_2936_BEHAVIOUR
    REQUIRE(fs::path("/a/b/").compare("/a/b/c") < 0);
    REQUIRE(fs::path("/a/b/").compare("a/c") > 0);
#endif // LWG_2936_BEHAVIOUR
}

TEST_CASE("FilesystemPath, decompose") {
    // root_name()
    REQUIRE(fs::path("").root_name() == "");
    REQUIRE(fs::path(".").root_name() == "");
    REQUIRE(fs::path("..").root_name() == "");
    REQUIRE(fs::path("foo").root_name() == "");
    REQUIRE(fs::path("/").root_name() == "");
    REQUIRE(fs::path("/foo").root_name() == "");
    REQUIRE(fs::path("foo/").root_name() == "");
    REQUIRE(fs::path("/foo/").root_name() == "");
    REQUIRE(fs::path("foo/bar").root_name() == "");
    REQUIRE(fs::path("/foo/bar").root_name() == "");
    REQUIRE(fs::path("///foo/bar").root_name() == "");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:/foo").root_name() == "C:");
    REQUIRE(fs::path("C:\\foo").root_name() == "C:");
    REQUIRE(fs::path("C:foo").root_name() == "C:");
#endif

    // root_directory()
    REQUIRE(fs::path("").root_directory() == "");
    REQUIRE(fs::path(".").root_directory() == "");
    REQUIRE(fs::path("..").root_directory() == "");
    REQUIRE(fs::path("foo").root_directory() == "");
    REQUIRE(fs::path("/").root_directory() == "/");
    REQUIRE(fs::path("/foo").root_directory() == "/");
    REQUIRE(fs::path("foo/").root_directory() == "");
    REQUIRE(fs::path("/foo/").root_directory() == "/");
    REQUIRE(fs::path("foo/bar").root_directory() == "");
    REQUIRE(fs::path("/foo/bar").root_directory() == "/");
    REQUIRE(fs::path("///foo/bar").root_directory() == "/");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:/foo").root_directory() == "/");
    REQUIRE(fs::path("C:\\foo").root_directory() == "/");
    REQUIRE(fs::path("C:foo").root_directory() == "");
#endif

    // root_path()
    REQUIRE(fs::path("").root_path() == "");
    REQUIRE(fs::path(".").root_path() == "");
    REQUIRE(fs::path("..").root_path() == "");
    REQUIRE(fs::path("foo").root_path() == "");
    REQUIRE(fs::path("/").root_path() == "/");
    REQUIRE(fs::path("/foo").root_path() == "/");
    REQUIRE(fs::path("foo/").root_path() == "");
    REQUIRE(fs::path("/foo/").root_path() == "/");
    REQUIRE(fs::path("foo/bar").root_path() == "");
    REQUIRE(fs::path("/foo/bar").root_path() == "/");
    REQUIRE(fs::path("///foo/bar").root_path() == "/");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:/foo").root_path() == "C:/");
    REQUIRE(fs::path("C:\\foo").root_path() == "C:/");
    REQUIRE(fs::path("C:foo").root_path() == "C:");
#endif

    // relative_path()
    REQUIRE(fs::path("").relative_path() == "");
    REQUIRE(fs::path(".").relative_path() == ".");
    REQUIRE(fs::path("..").relative_path() == "..");
    REQUIRE(fs::path("foo").relative_path() == "foo");
    REQUIRE(fs::path("/").relative_path() == "");
    REQUIRE(fs::path("/foo").relative_path() == "foo");
    REQUIRE(fs::path("foo/").relative_path() == "foo/");
    REQUIRE(fs::path("/foo/").relative_path() == "foo/");
    REQUIRE(fs::path("foo/bar").relative_path() == "foo/bar");
    REQUIRE(fs::path("/foo/bar").relative_path() == "foo/bar");
    REQUIRE(fs::path("///foo/bar").relative_path() == "foo/bar");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:/foo").relative_path() == "foo");
    REQUIRE(fs::path("C:\\foo").relative_path() == "foo");
    REQUIRE(fs::path("C:foo").relative_path() == "foo");
#endif

    // parent_path()
    REQUIRE(fs::path("").parent_path() == "");
    REQUIRE(fs::path(".").parent_path() == "");
    REQUIRE(fs::path("..").parent_path() ==
            ""); // unintuitive but as defined in the standard
    REQUIRE(fs::path("foo").parent_path() == "");
    REQUIRE(fs::path("/").parent_path() == "/");
    REQUIRE(fs::path("/foo").parent_path() == "/");
    REQUIRE(fs::path("foo/").parent_path() == "foo");
    REQUIRE(fs::path("/foo/").parent_path() == "/foo");
    REQUIRE(fs::path("foo/bar").parent_path() == "foo");
    REQUIRE(fs::path("/foo/bar").parent_path() == "/foo");
    REQUIRE(fs::path("///foo/bar").parent_path() == "/foo");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:/foo").parent_path() == "C:/");
    REQUIRE(fs::path("C:\\foo").parent_path() == "C:/");
    REQUIRE(fs::path("C:foo").parent_path() == "C:");
#endif

    // filename()
    REQUIRE(fs::path("").filename() == "");
    REQUIRE(fs::path(".").filename() == ".");
    REQUIRE(fs::path("..").filename() == "..");
    REQUIRE(fs::path("foo").filename() == "foo");
    REQUIRE(fs::path("/").filename() == "");
    REQUIRE(fs::path("/foo").filename() == "foo");
    REQUIRE(fs::path("foo/").filename() == "");
    REQUIRE(fs::path("/foo/").filename() == "");
    REQUIRE(fs::path("foo/bar").filename() == "bar");
    REQUIRE(fs::path("/foo/bar").filename() == "bar");
    REQUIRE(fs::path("///foo/bar").filename() == "bar");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:/foo").filename() == "foo");
    REQUIRE(fs::path("C:\\foo").filename() == "foo");
    REQUIRE(fs::path("C:foo").filename() == "foo");
#endif

    // stem()
    REQUIRE(fs::path("/foo/bar.txt").stem() == "bar");
    {
        fs::path p = "foo.bar.baz.tar";
        REQUIRE(p.extension() == ".tar");
        p = p.stem();
        REQUIRE(p.extension() == ".baz");
        p = p.stem();
        REQUIRE(p.extension() == ".bar");
        p = p.stem();
        REQUIRE(p == "foo");
    }
    REQUIRE(fs::path("/foo/.profile").stem() == ".profile");
    REQUIRE(fs::path(".bar").stem() == ".bar");
    REQUIRE(fs::path("..bar").stem() == ".");

    // extension()
    REQUIRE(fs::path("/foo/bar.txt").extension() == ".txt");
    REQUIRE(fs::path("/foo/bar").extension() == "");
    REQUIRE(fs::path("/foo/.profile").extension() == "");
    REQUIRE(fs::path(".bar").extension() == "");
    REQUIRE(fs::path("..bar").extension() == ".bar");

    if (has_host_root_name_support()) {
        // //host-based root-names
        REQUIRE(fs::path("//host").root_name() == "//host");
        REQUIRE(fs::path("//host/foo").root_name() == "//host");
        REQUIRE(fs::path("//host").root_directory() == "");
        REQUIRE(fs::path("//host/foo").root_directory() == "/");
        REQUIRE(fs::path("//host").root_path() == "//host");
        REQUIRE(fs::path("//host/foo").root_path() == "//host/");
        REQUIRE(fs::path("//host").relative_path() == "");
        REQUIRE(fs::path("//host/foo").relative_path() == "foo");
        REQUIRE(fs::path("//host").parent_path() == "//host");
        REQUIRE(fs::path("//host/foo").parent_path() == "//host/");
        REQUIRE(fs::path("//host").filename() == "");
        REQUIRE(fs::path("//host/foo").filename() == "foo");
    }
}

TEST_CASE("FilesystemPath, query") {
    // empty
    REQUIRE(fs::path("").empty());
    REQUIRE(!fs::path("foo").empty());

    // has_root_path()
    REQUIRE(!fs::path("foo").has_root_path());
    REQUIRE(!fs::path("foo/bar").has_root_path());
    REQUIRE(fs::path("/foo").has_root_path());
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:foo").has_root_path());
    REQUIRE(fs::path("C:/foo").has_root_path());
#endif

    // has_root_name()
    REQUIRE(!fs::path("foo").has_root_name());
    REQUIRE(!fs::path("foo/bar").has_root_name());
    REQUIRE(!fs::path("/foo").has_root_name());
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("C:foo").has_root_name());
    REQUIRE(fs::path("C:/foo").has_root_name());
#endif

    // has_root_directory()
    REQUIRE(!fs::path("foo").has_root_directory());
    REQUIRE(!fs::path("foo/bar").has_root_directory());
    REQUIRE(fs::path("/foo").has_root_directory());
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(!fs::path("C:foo").has_root_directory());
    REQUIRE(fs::path("C:/foo").has_root_directory());
#endif

    // has_relative_path()
    REQUIRE(!fs::path("").has_relative_path());
    REQUIRE(!fs::path("/").has_relative_path());
    REQUIRE(fs::path("/foo").has_relative_path());

    // has_parent_path()
    REQUIRE(!fs::path("").has_parent_path());
    REQUIRE(!fs::path(".").has_parent_path());
    REQUIRE(
            !fs::path("..")
                    .has_parent_path()); // unintuitive but as defined in the standard
    REQUIRE(!fs::path("foo").has_parent_path());
    REQUIRE(fs::path("/").has_parent_path());
    REQUIRE(fs::path("/foo").has_parent_path());
    REQUIRE(fs::path("foo/").has_parent_path());
    REQUIRE(fs::path("/foo/").has_parent_path());

    // has_filename()
    REQUIRE(fs::path("foo").has_filename());
    REQUIRE(fs::path("foo/bar").has_filename());
    REQUIRE(!fs::path("/foo/bar/").has_filename());

    // has_stem()
    REQUIRE(fs::path("foo").has_stem());
    REQUIRE(fs::path("foo.bar").has_stem());
    REQUIRE(fs::path(".profile").has_stem());
    REQUIRE(!fs::path("/foo/").has_stem());

    // has_extension()
    REQUIRE(!fs::path("foo").has_extension());
    REQUIRE(fs::path("foo.bar").has_extension());
    REQUIRE(!fs::path(".profile").has_extension());

    // is_absolute()
    REQUIRE(!fs::path("foo/bar").is_absolute());
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(!fs::path("/foo").is_absolute());
    REQUIRE(!fs::path("c:foo").is_absolute());
    REQUIRE(fs::path("c:/foo").is_absolute());
#else
    REQUIRE(fs::path("/foo").is_absolute());
#endif

    // is_relative()
    REQUIRE(fs::path("foo/bar").is_relative());
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("/foo").is_relative());
    REQUIRE(fs::path("c:foo").is_relative());
    REQUIRE(!fs::path("c:/foo").is_relative());
#else
    REQUIRE(!fs::path("/foo").is_relative());
#endif

    if (has_host_root_name_support()) {
        REQUIRE(fs::path("//host").has_root_name());
        REQUIRE(fs::path("//host/foo").has_root_name());
        REQUIRE(fs::path("//host").has_root_path());
        REQUIRE(fs::path("//host/foo").has_root_path());
        REQUIRE(!fs::path("//host").has_root_directory());
        REQUIRE(fs::path("//host/foo").has_root_directory());
        REQUIRE(!fs::path("//host").has_relative_path());
        REQUIRE(fs::path("//host/foo").has_relative_path());
        REQUIRE(fs::path("//host/foo").is_absolute());
        REQUIRE(!fs::path("//host/foo").is_relative());
    }
}

TEST_CASE("FilesystemPath, fs_path_gen") {
    // lexically_normal()
    REQUIRE(fs::path("foo/./bar/..").lexically_normal() == "foo/");
    REQUIRE(fs::path("foo/.///bar/../").lexically_normal() == "foo/");
    REQUIRE(fs::path("/foo/../..").lexically_normal() == "/");
    REQUIRE(fs::path("foo/..").lexically_normal() == ".");
    REQUIRE(fs::path("ab/cd/ef/../../qw").lexically_normal() == "ab/qw");
    REQUIRE(fs::path("a/b/../../../c").lexically_normal() == "../c");
    REQUIRE(fs::path("../").lexically_normal() == "..");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("\\/\\///\\/").lexically_normal() == "/");
    REQUIRE(fs::path("a/b/..\\//..///\\/../c\\\\/").lexically_normal() ==
                "../c/");
    REQUIRE(fs::path("..a/b/..\\//..///\\/../c\\\\/").lexically_normal() ==
                "../c/");
    REQUIRE(fs::path("..\\").lexically_normal() == "..");
#endif

    // lexically_relative()
    REQUIRE(fs::path("/a/d").lexically_relative("/a/b/c") == "../../d");
    REQUIRE(fs::path("/a/b/c").lexically_relative("/a/d") == "../b/c");
    REQUIRE(fs::path("a/b/c").lexically_relative("a") == "b/c");
    REQUIRE(fs::path("a/b/c").lexically_relative("a/b/c/x/y") == "../..");
    REQUIRE(fs::path("a/b/c").lexically_relative("a/b/c") == ".");
    REQUIRE(fs::path("a/b").lexically_relative("c/d") == "../../a/b");
    REQUIRE(fs::path("a/b").lexically_relative("a/") == "b");
    if (has_host_root_name_support()) {
        REQUIRE(fs::path("//host1/foo").lexically_relative("//host2.bar") ==
                "");
    }
#ifdef TURBO_PLATFORM_WINDOWS
        REQUIRE(fs::path("c:/foo").lexically_relative("/bar") == "");
        REQUIRE(fs::path("c:foo").lexically_relative("c:/bar") == "");
        REQUIRE(fs::path("foo").lexically_relative("/bar") == "");
        REQUIRE(fs::path("c:/foo/bar.txt").lexically_relative("c:/foo/") ==
                    "bar.txt");
        REQUIRE(fs::path("c:/foo/bar.txt").lexically_relative("C:/foo/") ==
                    "bar.txt");
#else
    REQUIRE(fs::path("/foo").lexically_relative("bar") == "");
    REQUIRE(fs::path("foo").lexically_relative("/bar") == "");
#endif

    // lexically_proximate()
    REQUIRE(fs::path("/a/d").lexically_proximate("/a/b/c") == "../../d");
    if (has_host_root_name_support()) {
        REQUIRE(fs::path("//host1/a/d").lexically_proximate("//host2/a/b/c") ==
                "//host1/a/d");
    }
    REQUIRE(fs::path("a/d").lexically_proximate("/a/b/c") == "a/d");
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE(fs::path("c:/a/d").lexically_proximate("c:/a/b/c") == "../../d");
    REQUIRE(fs::path("c:/a/d").lexically_proximate("d:/a/b/c") == "c:/a/d");
    REQUIRE(fs::path("c:/foo").lexically_proximate("/bar") == "c:/foo");
    REQUIRE(fs::path("c:foo").lexically_proximate("c:/bar") == "c:foo");
    REQUIRE(fs::path("foo").lexically_proximate("/bar") == "foo");
#else
    REQUIRE(fs::path("/foo").lexically_proximate("bar") == "/foo");
    REQUIRE(fs::path("foo").lexically_proximate("/bar") == "foo");
#endif
}

static std::string iterateResult(const fs::path &path) {
    std::ostringstream result;
    for (fs::path::const_iterator i = path.begin(); i != path.end(); ++i) {
        if (i != path.begin()) {
            result << ",";
        }
        result << i->generic_string();
    }
    return result.str();
}

static std::string reverseIterateResult(const fs::path &path) {
    std::ostringstream result;
    fs::path::const_iterator iter = path.end();
    bool first = true;
    if (iter != path.begin()) {
        do {
            --iter;
            if (!first) {
                result << ",";
            }
            first = false;
            result << iter->generic_string();
        } while (iter != path.begin());
    }
    return result.str();
}

TEST_CASE("FilesystemPath, itr") {
    REQUIRE(iterateResult(fs::path()).empty());
    REQUIRE("." == iterateResult(fs::path(".")));
    REQUIRE(".." == iterateResult(fs::path("..")));
    REQUIRE("foo" == iterateResult(fs::path("foo")));
    REQUIRE("/" == iterateResult(fs::path("/")));
    REQUIRE("/,foo" == iterateResult(fs::path("/foo")));
    REQUIRE("foo," == iterateResult(fs::path("foo/")));
    REQUIRE("/,foo," == iterateResult(fs::path("/foo/")));
    REQUIRE("foo,bar" == iterateResult(fs::path("foo/bar")));
    REQUIRE("/,foo,bar" == iterateResult(fs::path("/foo/bar")));
#ifndef USE_STD_FS
    // turbo::filesystem enforces redundant slashes to be reduced to one
    REQUIRE("/,foo,bar" == iterateResult(fs::path("///foo/bar")));
#else
    // typically std::filesystem keeps them
    REQUIRE("///,foo,bar" == iterateResult(fs::path("///foo/bar")));
#endif
    REQUIRE("/,foo,bar," == iterateResult(fs::path("/foo/bar///")));
    REQUIRE("foo,.,bar,..," == iterateResult(fs::path("foo/.///bar/../")));
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE("C:,/,foo" == iterateResult(fs::path("C:/foo")));
#endif

    REQUIRE(reverseIterateResult(fs::path()).empty());
    REQUIRE("." == reverseIterateResult(fs::path(".")));
    REQUIRE(".." == reverseIterateResult(fs::path("..")));
    REQUIRE("foo" == reverseIterateResult(fs::path("foo")));
    REQUIRE("/" == reverseIterateResult(fs::path("/")));
    REQUIRE("foo,/" == reverseIterateResult(fs::path("/foo")));
    REQUIRE(",foo" == reverseIterateResult(fs::path("foo/")));
    REQUIRE(",foo,/" == reverseIterateResult(fs::path("/foo/")));
    REQUIRE("bar,foo" == reverseIterateResult(fs::path("foo/bar")));
    REQUIRE("bar,foo,/" == reverseIterateResult(fs::path("/foo/bar")));
#ifndef USE_STD_FS
    // turbo::filesystem enforces redundant slashes to be reduced to one
    REQUIRE("bar,foo,/" == reverseIterateResult(fs::path("///foo/bar")));
#else
    // typically std::filesystem keeps them
    REQUIRE("bar,foo,///" == reverseIterateResult(fs::path("///foo/bar")));
#endif
    REQUIRE(",bar,foo,/" == reverseIterateResult(fs::path("/foo/bar///")));
    REQUIRE(",..,bar,.,foo" ==
            reverseIterateResult(fs::path("foo/.///bar/../")));
#ifdef TURBO_PLATFORM_WINDOWS
    REQUIRE("foo,/,C:" == reverseIterateResult(fs::path("C:/foo")));
    REQUIRE("foo,C:" == reverseIterateResult(fs::path("C:foo")));
#endif
    {
        fs::path p1 = "/foo/bar/test.txt";
        fs::path p2;
        for (auto pe: p1) {
            p2 /= pe;
        }
        REQUIRE(p1 == p2);
        REQUIRE("bar" == *(--fs::path("/foo/bar").end()));
        auto p = fs::path("/foo/bar");
        auto pi = p.end();
        pi--;
        REQUIRE("bar" == *pi);
    }

    if (has_host_root_name_support()) {
        REQUIRE("foo" == *(--fs::path("//host/foo").end()));
        auto p = fs::path("//host/foo");
        auto pi = p.end();
        pi--;
        REQUIRE("foo" == *pi);
        REQUIRE("//host" == iterateResult(fs::path("//host")));
        REQUIRE("//host,/,foo" == iterateResult(fs::path("//host/foo")));
        REQUIRE("//host" == reverseIterateResult(fs::path("//host")));
        REQUIRE("foo,/,//host" == reverseIterateResult(fs::path("//host/foo")));
        {
            fs::path p1 = "//host/foo/bar/test.txt";
            fs::path p2;
            for (auto pe: p1) {
                p2 /= pe;
            }
            REQUIRE(p1 == p2);
        }
    }
}

TEST_CASE("FilesystemPath, nonmember") {
    fs::path p1("foo/bar");
    fs::path p2("some/other");
    fs::swap(p1, p2);
    REQUIRE(p1 == "some/other");
    REQUIRE(p2 == "foo/bar");
    REQUIRE(hash_value(p1));
    REQUIRE(p2 < p1);
    REQUIRE(p2 <= p1);
    REQUIRE(p1 <= p1);
    REQUIRE(!(p1 < p2));
    REQUIRE(!(p1 <= p2));
    REQUIRE(p1 > p2);
    REQUIRE(p1 >= p2);
    REQUIRE(p1 >= p1);
    REQUIRE(!(p2 > p1));
    REQUIRE(!(p2 >= p1));
    REQUIRE(p1 != p2);
    REQUIRE(p1 / p2 == "some/other/foo/bar");
}

TEST_CASE("FilesystemPath, io") {
    {
        std::ostringstream os;
        os << fs::path("/root/foo bar");
#ifdef TURBO_PLATFORM_WINDOWS
        REQUIRE(os.str() == "\"\\\\root\\\\foo bar\"");
#else
        REQUIRE(os.str() == "\"/root/foo bar\"");
#endif
    }
    {
        std::ostringstream os;
        os << fs::path("/root/foo\"bar");
#ifdef TURBO_PLATFORM_WINDOWS
        REQUIRE(os.str() == "\"\\\\root\\\\foo\\\"bar\"");
#else
        REQUIRE(os.str() == "\"/root/foo\\\"bar\"");
#endif
    }

    {
        std::istringstream is("\"/root/foo bar\"");
        fs::path p;
        is >> p;
        REQUIRE(p == fs::path("/root/foo bar"));
        REQUIRE((is.flags() & std::ios_base::skipws) == std::ios_base::skipws);
    }
    {
        std::istringstream is("\"/root/foo bar\"");
        is >> std::noskipws;
        fs::path p;
        is >> p;
        REQUIRE(p == fs::path("/root/foo bar"));
        REQUIRE((is.flags() & std::ios_base::skipws) != std::ios_base::skipws);
    }
    {
        std::istringstream is("\"/root/foo\\\"bar\"");
        fs::path p;
        is >> p;
        REQUIRE(p == fs::path("/root/foo\"bar"));
    }
    {
        std::istringstream is("/root/foo");
        fs::path p;
        is >> p;
        REQUIRE(p == fs::path("/root/foo"));
    }
}

TEST_CASE("FilesystemPath, factory") {
    REQUIRE(fs::u8path("foo/bar") == fs::path("foo/bar"));
    REQUIRE(fs::u8path("foo/bar") == fs::path("foo/bar"));
    std::string str("/foo/bar/test.txt");
    REQUIRE(fs::u8path(str.begin(), str.end()) == str);
}

TEST_CASE("FilesystemPath, filesystem_error") {
    std::error_code ec(1, std::system_category());
    fs::filesystem_error fse("None", std::error_code());
    fse = fs::filesystem_error("Some error", ec);
    REQUIRE(fse.code().value() == 1);
    REQUIRE(!std::string(fse.what()).empty());
    REQUIRE(fse.path1().empty());
    REQUIRE(fse.path2().empty());
    fse = fs::filesystem_error("Some error", fs::path("foo/bar"), ec);
    REQUIRE(!std::string(fse.what()).empty());
    REQUIRE(fse.path1() == "foo/bar");
    REQUIRE(fse.path2().empty());
    fse = fs::filesystem_error("Some error", fs::path("foo/bar"),
                               fs::path("some/other"), ec);
    REQUIRE(!std::string(fse.what()).empty());
    REQUIRE(fse.path1() == "foo/bar");
    REQUIRE(fse.path2() == "some/other");
}

constexpr fs::perms constExprOwnerAll() {
    return fs::perms::owner_read | fs::perms::owner_write | fs::perms::owner_exec;
}

TEST_CASE("FilesystemPath, fs_enum") {
    static_assert(constExprOwnerAll() == fs::perms::owner_all,
                  "constexpr didn't result in owner_all");
    REQUIRE((fs::perms::owner_read | fs::perms::owner_write |
             fs::perms::owner_exec) == fs::perms::owner_all);
    REQUIRE((fs::perms::group_read | fs::perms::group_write |
             fs::perms::group_exec) == fs::perms::group_all);
    REQUIRE((fs::perms::others_read | fs::perms::others_write |
             fs::perms::others_exec) == fs::perms::others_all);
    REQUIRE((fs::perms::owner_all | fs::perms::group_all |
             fs::perms::others_all) == fs::perms::all);
    REQUIRE((fs::perms::all | fs::perms::set_uid | fs::perms::set_gid |
             fs::perms::sticky_bit) == fs::perms::mask);
}

TEST_CASE("FilesystemPath, file_status") {
    {
        fs::file_status fs;
        REQUIRE(fs.type() == fs::file_type::none);
        REQUIRE(fs.permissions() == fs::perms::unknown);
    }
    {
        fs::file_status fs{fs::file_type::regular};
        REQUIRE(fs.type() == fs::file_type::regular);
        REQUIRE(fs.permissions() == fs::perms::unknown);
    }
    {
        fs::file_status fs{fs::file_type::directory, fs::perms::owner_read |
                                                     fs::perms::owner_write |
                                                     fs::perms::owner_exec};
        REQUIRE(fs.type() == fs::file_type::directory);
        REQUIRE(fs.permissions() == fs::perms::owner_all);
        fs.type(fs::file_type::block);
        REQUIRE(fs.type() == fs::file_type::block);
        fs.type(fs::file_type::character);
        REQUIRE(fs.type() == fs::file_type::character);
        fs.type(fs::file_type::fifo);
        REQUIRE(fs.type() == fs::file_type::fifo);
        fs.type(fs::file_type::symlink);
        REQUIRE(fs.type() == fs::file_type::symlink);
        fs.type(fs::file_type::socket);
        REQUIRE(fs.type() == fs::file_type::socket);
        fs.permissions(fs.permissions() | fs::perms::group_all |
                       fs::perms::others_all);
        REQUIRE(fs.permissions() == fs::perms::all);
    }
    {
        fs::file_status fst(fs::file_type::regular);
        fs::file_status fs(std::move(fst));
        REQUIRE(fs.type() == fs::file_type::regular);
        REQUIRE(fs.permissions() == fs::perms::unknown);
    }
#if !defined(USE_STD_FS) || defined(TURBO_FILESYSTEM_RUNNING_CPP20)
    {
        fs::file_status fs1{fs::file_type::regular, fs::perms::owner_read |
                                                    fs::perms::owner_write |
                                                    fs::perms::owner_exec};
        fs::file_status fs2{fs::file_type::regular, fs::perms::owner_read |
                                                    fs::perms::owner_write |
                                                    fs::perms::owner_exec};
        fs::file_status fs3{fs::file_type::directory, fs::perms::owner_read |
                                                      fs::perms::owner_write |
                                                      fs::perms::owner_exec};
        fs::file_status fs4{fs::file_type::regular,
                            fs::perms::owner_read | fs::perms::owner_write};
        REQUIRE(fs1 == fs2);
        REQUIRE_FALSE(fs1 == fs3);
        REQUIRE_FALSE(fs1 == fs4);
    }
#endif
}

TEST_CASE("FilesystemDir, dir_entry") {
    TemporaryDirectory t;
    std::error_code ec;
    auto de = fs::directory_entry(t.path());
    REQUIRE(de.path() == t.path());
    REQUIRE((fs::path) de == t.path());
    REQUIRE(de.exists());
    REQUIRE(!de.is_block_file());
    REQUIRE(!de.is_character_file());
    REQUIRE(de.is_directory());
    REQUIRE(!de.is_fifo());
    REQUIRE(!de.is_other());
    REQUIRE(!de.is_regular_file());
    REQUIRE(!de.is_socket());
    REQUIRE(!de.is_symlink());
    REQUIRE(de.status().type() == fs::file_type::directory);
    ec.clear();
    REQUIRE(de.status(ec).type() == fs::file_type::directory);
    REQUIRE(!ec);
    REQUIRE_NOTHROW(de.refresh());
    fs::directory_entry none;
    REQUIRE_THROWS_AS(none.refresh(), fs::filesystem_error);
    ec.clear();
    REQUIRE_NOTHROW(none.refresh(ec));
    REQUIRE(ec);
    REQUIRE_THROWS_AS(de.assign(""), fs::filesystem_error);
    ec.clear();
    REQUIRE_NOTHROW(de.assign("", ec));
    REQUIRE(ec);
    generateFile(t.path() / "foo", 1234);
    auto now = fs::file_time_type::clock::now();
    REQUIRE_NOTHROW(de.assign(t.path() / "foo"));
    REQUIRE_NOTHROW(de.assign(t.path() / "foo", ec));
    REQUIRE(!ec);
    de = fs::directory_entry(t.path() / "foo");
    REQUIRE(de.path() == t.path() / "foo");
    REQUIRE(de.exists());
    REQUIRE(de.exists(ec));
    REQUIRE(!ec);
    REQUIRE(!de.is_block_file());
    REQUIRE(!de.is_block_file(ec));
    REQUIRE(!ec);
    REQUIRE(!de.is_character_file());
    REQUIRE(!de.is_character_file(ec));
    REQUIRE(!ec);
    REQUIRE(!de.is_directory());
    REQUIRE(!de.is_directory(ec));
    REQUIRE(!ec);
    REQUIRE(!de.is_fifo());
    REQUIRE(!de.is_fifo(ec));
    REQUIRE(!ec);
    REQUIRE(!de.is_other());
    REQUIRE(!de.is_other(ec));
    REQUIRE(!ec);
    REQUIRE(de.is_regular_file());
    REQUIRE(de.is_regular_file(ec));
    REQUIRE(!ec);
    REQUIRE(!de.is_socket());
    REQUIRE(!de.is_socket(ec));
    REQUIRE(!ec);
    REQUIRE(!de.is_symlink());
    REQUIRE(!de.is_symlink(ec));
    REQUIRE(!ec);
    REQUIRE(de.file_size() == 1234);
    REQUIRE(de.file_size(ec) == 1234);
    REQUIRE(std::abs(std::chrono::duration_cast<std::chrono::seconds>(
            de.last_write_time() - now)
                             .count()) < 3);
    ec.clear();
    REQUIRE(std::abs(std::chrono::duration_cast<std::chrono::seconds>(
            de.last_write_time(ec) - now)
                             .count()) < 3);
    REQUIRE(!ec);
#ifndef TURBO_PLATFORM_WEB
    REQUIRE(de.hard_link_count() == 1);
    REQUIRE(de.hard_link_count(ec) == 1);
    REQUIRE(!ec);
#endif
    REQUIRE_THROWS_AS(de.replace_filename("bar"), fs::filesystem_error);
    REQUIRE_NOTHROW(de.replace_filename("foo"));
    ec.clear();
    REQUIRE_NOTHROW(de.replace_filename("bar", ec));
    REQUIRE(ec);
    auto de2none = fs::directory_entry();
    ec.clear();
#ifndef TURBO_PLATFORM_WEB
    REQUIRE(de2none.hard_link_count(ec) == static_cast<uintmax_t>(-1));
    REQUIRE_THROWS_AS(de2none.hard_link_count(), fs::filesystem_error);
    REQUIRE(ec);
#endif
    ec.clear();
    REQUIRE_NOTHROW(de2none.last_write_time(ec));
    REQUIRE_THROWS_AS(de2none.last_write_time(), fs::filesystem_error);
    REQUIRE(ec);
    ec.clear();
    REQUIRE_THROWS_AS(de2none.file_size(), fs::filesystem_error);
    REQUIRE(de2none.file_size(ec) == static_cast<uintmax_t>(-1));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(de2none.status().type() == fs::file_type::not_found);
    REQUIRE(de2none.status(ec).type() == fs::file_type::not_found);
    REQUIRE(ec);
    generateFile(t.path() / "a");
    generateFile(t.path() / "b");
    auto d1 = fs::directory_entry(t.path() / "a");
    auto d2 = fs::directory_entry(t.path() / "b");
    REQUIRE(d1 < d2);
    REQUIRE(!(d2 < d1));
    REQUIRE(d1 <= d2);
    REQUIRE(!(d2 <= d1));
    REQUIRE(d2 > d1);
    REQUIRE(!(d1 > d2));
    REQUIRE(d2 >= d1);
    REQUIRE(!(d1 >= d2));
    REQUIRE(d1 != d2);
    REQUIRE(!(d2 != d2));
    REQUIRE(d1 == d1);
    REQUIRE(!(d1 == d2));
}

TEST_CASE("FilesystemDir, directory_iterator") {
    {
        TemporaryDirectory t;
        REQUIRE(fs::directory_iterator(t.path()) == fs::directory_iterator());
        generateFile(t.path() / "test", 1234);
        REQUIRE(fs::directory_iterator(t.path()) != fs::directory_iterator());
        auto iter = fs::directory_iterator(t.path());
        fs::directory_iterator iter2(iter);
        fs::directory_iterator iter3, iter4;
        iter3 = iter;
        REQUIRE(iter->path().filename() == "test");
        REQUIRE(iter2->path().filename() == "test");
        REQUIRE(iter3->path().filename() == "test");
        iter4 = std::move(iter3);
        REQUIRE(iter4->path().filename() == "test");
        REQUIRE(iter->path() == t.path() / "test");
        REQUIRE(!iter->is_symlink());
        REQUIRE(iter->is_regular_file());
        REQUIRE(!iter->is_directory());
        REQUIRE(iter->file_size() == 1234);
        REQUIRE(++iter == fs::directory_iterator());
        REQUIRE_THROWS_AS(fs::directory_iterator(t.path() / "non-existing"),
                          fs::filesystem_error);
        int cnt = 0;
        for (auto de: fs::directory_iterator(t.path())) {
            ++cnt;
        }
        REQUIRE(cnt == 1);
    }
    if (is_symlink_creation_supported()) {
        TemporaryDirectory t;
        fs::path td = t.path() / "testdir";
        REQUIRE(fs::directory_iterator(t.path()) == fs::directory_iterator());
        generateFile(t.path() / "test", 1234);
        fs::create_directory(td);
        REQUIRE_NOTHROW(fs::create_symlink(t.path() / "test", td / "testlink"));
        std::error_code ec;
        REQUIRE(fs::directory_iterator(td) != fs::directory_iterator());
        auto iter = fs::directory_iterator(td);
        REQUIRE(iter->path().filename() == "testlink");
        REQUIRE(iter->path() == td / "testlink");
        REQUIRE(iter->is_symlink());
        REQUIRE(iter->is_regular_file());
        REQUIRE(!iter->is_directory());
        REQUIRE(iter->file_size() == 1234);
        REQUIRE(++iter == fs::directory_iterator());
    }
    {
        // Issue #8: check if resources are freed when iterator reaches end()
        TemporaryDirectory t(TempOpt::change_path);
        auto p = fs::path("test/");
        fs::create_directory(p);
        auto iter = fs::directory_iterator(p);
        while (iter != fs::directory_iterator()) {
            ++iter;
        }
        REQUIRE(fs::remove_all(p) == 1);
        REQUIRE_NOTHROW(fs::create_directory(p));
    }
}

TEST_CASE("FilesystemDir, rec_dir_itr") {
    {
        auto iter = fs::recursive_directory_iterator(".");
        iter.pop();
        REQUIRE(iter == fs::recursive_directory_iterator());
    }
    {
        TemporaryDirectory t;
        REQUIRE(fs::recursive_directory_iterator(t.path()) ==
                fs::recursive_directory_iterator());
        generateFile(t.path() / "test", 1234);
        REQUIRE(fs::recursive_directory_iterator(t.path()) !=
                fs::recursive_directory_iterator());
        auto iter = fs::recursive_directory_iterator(t.path());
        REQUIRE(iter->path().filename() == "test");
        REQUIRE(iter->path() == t.path() / "test");
        REQUIRE(!iter->is_symlink());
        REQUIRE(iter->is_regular_file());
        REQUIRE(!iter->is_directory());
        REQUIRE(iter->file_size() == 1234);
        REQUIRE(++iter == fs::recursive_directory_iterator());
    }

    {
        TemporaryDirectory t;
        fs::path td = t.path() / "testdir";
        fs::create_directories(td);
        generateFile(td / "test", 1234);
        REQUIRE(fs::recursive_directory_iterator(t.path()) !=
                fs::recursive_directory_iterator());
        auto iter = fs::recursive_directory_iterator(t.path());

        REQUIRE(iter->path().filename() == "testdir");
        REQUIRE(iter->path() == td);
        REQUIRE(!iter->is_symlink());
        REQUIRE(!iter->is_regular_file());
        REQUIRE(iter->is_directory());

        REQUIRE(++iter != fs::recursive_directory_iterator());

        REQUIRE(iter->path().filename() == "test");
        REQUIRE(iter->path() == td / "test");
        REQUIRE(!iter->is_symlink());
        REQUIRE(iter->is_regular_file());
        REQUIRE(!iter->is_directory());
        REQUIRE(iter->file_size() == 1234);

        REQUIRE(++iter == fs::recursive_directory_iterator());
    }
    {
        TemporaryDirectory t;
        std::error_code ec;
        REQUIRE(fs::recursive_directory_iterator(t.path(),
                                                 fs::directory_options::none) ==
                fs::recursive_directory_iterator());
        REQUIRE(fs::recursive_directory_iterator(
                t.path(), fs::directory_options::none, ec) ==
                fs::recursive_directory_iterator());
        REQUIRE(!ec);
        REQUIRE(fs::recursive_directory_iterator(t.path(), ec) ==
                fs::recursive_directory_iterator());
        REQUIRE(!ec);
        generateFile(t.path() / "test");
        fs::recursive_directory_iterator rd1(t.path());
        REQUIRE(fs::recursive_directory_iterator(rd1) !=
                        fs::recursive_directory_iterator());
        fs::recursive_directory_iterator rd2(t.path());
        REQUIRE(fs::recursive_directory_iterator(std::move(rd2)) !=
                fs::recursive_directory_iterator());
        fs::recursive_directory_iterator rd3(
                t.path(), fs::directory_options::skip_permission_denied);
        REQUIRE(rd3.options() == fs::directory_options::skip_permission_denied);
        fs::recursive_directory_iterator rd4;
        rd4 = std::move(rd3);
        REQUIRE(rd4 != fs::recursive_directory_iterator());
        REQUIRE_NOTHROW(++rd4);
        REQUIRE(rd4 == fs::recursive_directory_iterator());
        fs::recursive_directory_iterator rd5;
        rd5 = rd4;
    }
    {
        TemporaryDirectory t(TempOpt::change_path);
        generateFile("a");
        fs::create_directory("d1");
        fs::create_directory("d1/d2");
        generateFile("d1/b");
        generateFile("d1/c");
        generateFile("d1/d2/d");
        generateFile("e");
        auto iter = fs::recursive_directory_iterator(".");
        std::multimap<std::string, int> result;
        while (iter != fs::recursive_directory_iterator()) {
            result.insert(
                    std::make_pair(iter->path().generic_string(), iter.depth()));
            ++iter;
        }
        std::stringstream os;
        for (auto p: result) {
            os << "[" << p.first << "," << p.second << "],";
        }
        REQUIRE(os.str() == "[./a,0],[./d1,0],[./d1/b,1],[./d1/c,1],[./d1/"
                            "d2,1],[./d1/d2/d,2],[./e,0],");
    }
    {
        TemporaryDirectory t(TempOpt::change_path);
        generateFile("a");
        fs::create_directory("d1");
        fs::create_directory("d1/d2");
        generateFile("d1/b");
        generateFile("d1/c");
        generateFile("d1/d2/d");
        generateFile("e");
        std::multiset<std::string> result;
        for (auto de: fs::recursive_directory_iterator(".")) {
            result.insert(de.path().generic_string());
        }
        std::stringstream os;
        for (auto p: result) {
            os << p << ",";
        }
        REQUIRE(os.str() == "./a,./d1,./d1/b,./d1/c,./d1/d2,./d1/d2/d,./e,");
    }
    {
        TemporaryDirectory t(TempOpt::change_path);
        generateFile("a");
        fs::create_directory("d1");
        fs::create_directory("d1/d2");
        generateFile("d1/d2/b");
        generateFile("e");
        auto iter = fs::recursive_directory_iterator(".");
        std::multimap<std::string, int> result;
        while (iter != fs::recursive_directory_iterator()) {
            result.insert(
                    std::make_pair(iter->path().generic_string(), iter.depth()));
            if (iter->path() == "./d1/d2") {
                iter.disable_recursion_pending();
            }
            ++iter;
        }
        std::stringstream os;
        for (auto p: result) {
            os << "[" << p.first << "," << p.second << "],";
        }
        REQUIRE(os.str() == "[./a,0],[./d1,0],[./d1/d2,1],[./e,0],");
    }
    {
        TemporaryDirectory t(TempOpt::change_path);
        generateFile("a");
        fs::create_directory("d1");
        fs::create_directory("d1/d2");
        generateFile("d1/d2/b");
        generateFile("e");
        auto iter = fs::recursive_directory_iterator(".");
        std::multimap<std::string, int> result;
        while (iter != fs::recursive_directory_iterator()) {
            result.insert(
                    std::make_pair(iter->path().generic_string(), iter.depth()));
            if (iter->path() == "./d1/d2") {
                iter.pop();
            } else {
                ++iter;
            }
        }
        std::stringstream os;
        for (auto p: result) {
            os << "[" << p.first << "," << p.second << "],";
        }
        REQUIRE(os.str() == "[./a,0],[./d1,0],[./d1/d2,1],[./e,0],");
    }
    if (is_symlink_creation_supported()) {
        TemporaryDirectory t(TempOpt::change_path);
        fs::create_directory("d1");
        generateFile("d1/a");
        fs::create_directory("d2");
        generateFile("d2/b");
        fs::create_directory_symlink("../d1", "d2/ds1");
        fs::create_directory_symlink("d3", "d2/ds2");
        std::multiset<std::string> result;
        REQUIRE_NOTHROW([&]() {
            for (const auto &de: fs::recursive_directory_iterator(
                    "d2", fs::directory_options::follow_directory_symlink)) {
                result.insert(de.path().generic_string());
            }
        }());
        std::stringstream os;
        for (const auto &p: result) {
            os << p << ",";
        }
        REQUIRE(os.str() == "d2/b,d2/ds1,d2/ds1/a,d2/ds2,");
        os.str("");
        result.clear();
        REQUIRE_NOTHROW([&]() {
            for (const auto &de: fs::recursive_directory_iterator("d2")) {
                result.insert(de.path().generic_string());
            }
        }());
        for (const auto &p: result) {
            os << p << ",";
        }
        REQUIRE(os.str() == "d2/b,d2/ds1,d2/ds2,");
    }
}

TEST_CASE("FilesystemDir, op_absolute") {
    REQUIRE(fs::absolute("") == fs::current_path() / "");
    REQUIRE(fs::absolute(fs::current_path()) == fs::current_path());
    REQUIRE(fs::absolute(".") == fs::current_path() / ".");
    REQUIRE((fs::absolute("..") == fs::current_path().parent_path() ||
             fs::absolute("..") == fs::current_path() / ".."));
    REQUIRE(fs::absolute("foo") == fs::current_path() / "foo");
    std::error_code ec;
    REQUIRE(fs::absolute("", ec) == fs::current_path() / "");
    REQUIRE(!ec);
    REQUIRE(fs::absolute("foo", ec) == fs::current_path() / "foo");
    REQUIRE(!ec);
}

TEST_CASE("FilesystemDir, op_canonical") {
    REQUIRE_THROWS_AS(fs::canonical(""), fs::filesystem_error);
    {
        std::error_code ec;
        REQUIRE(fs::canonical("", ec) == "");
        REQUIRE(ec);
    }
    REQUIRE(fs::canonical(fs::current_path()) == fs::current_path());

    REQUIRE(fs::canonical(".") == fs::current_path());
    REQUIRE(fs::canonical("..") == fs::current_path().parent_path());
    REQUIRE(fs::canonical("/") == fs::current_path().root_path());
    REQUIRE_THROWS_AS(fs::canonical("foo"), fs::filesystem_error);
    {
        std::error_code ec;
        REQUIRE_NOTHROW(fs::canonical("foo", ec));
        REQUIRE(ec);
    }
    {
        TemporaryDirectory t(TempOpt::change_path);
        auto dir = t.path() / "d0";
        fs::create_directories(dir / "d1");
        generateFile(dir / "f0");
        fs::path rel(dir.filename());
        REQUIRE(fs::canonical(dir) == dir);
        REQUIRE(fs::canonical(rel) == dir);
        REQUIRE(fs::canonical(dir / "f0") == dir / "f0");
        REQUIRE(fs::canonical(rel / "f0") == dir / "f0");
        REQUIRE(fs::canonical(rel / "./f0") == dir / "f0");
        REQUIRE(fs::canonical(rel / "d1/../f0") == dir / "f0");
    }

    if (is_symlink_creation_supported()) {
        TemporaryDirectory t(TempOpt::change_path);
        fs::create_directory(t.path() / "dir1");
        generateFile(t.path() / "dir1/test1");
        fs::create_directory(t.path() / "dir2");
        fs::create_directory_symlink(t.path() / "dir1", t.path() / "dir2/dirSym");
        REQUIRE(fs::canonical(t.path() / "dir2/dirSym/test1") ==
                t.path() / "dir1/test1");
    }
}

TEST_CASE("FilesystemDir, op_copy") {
    {
        TemporaryDirectory t(TempOpt::change_path);
        std::error_code ec;
        fs::create_directory("dir1");
        generateFile("dir1/file1");
        generateFile("dir1/file2");
        fs::create_directory("dir1/dir2");
        generateFile("dir1/dir2/file3");
        REQUIRE_NOTHROW(fs::copy("dir1", "dir3"));
        REQUIRE(fs::exists("dir3/file1"));
        REQUIRE(fs::exists("dir3/file2"));
        REQUIRE(!fs::exists("dir3/dir2"));
        REQUIRE_NOTHROW(fs::copy("dir1", "dir4", fs::copy_options::recursive, ec));
        REQUIRE(!ec);
        REQUIRE(fs::exists("dir4/file1"));
        REQUIRE(fs::exists("dir4/file2"));
        REQUIRE(fs::exists("dir4/dir2/file3"));
        fs::create_directory("dir5");
        generateFile("dir5/file1");
        REQUIRE_THROWS_AS(fs::copy("dir1/file1", "dir5/file1"), fs::filesystem_error);
        REQUIRE_NOTHROW(
                fs::copy("dir1/file1", "dir5/file1", fs::copy_options::skip_existing));
    }
    if (is_symlink_creation_supported()) {
        TemporaryDirectory t(TempOpt::change_path);
        std::error_code ec;
        fs::create_directory("dir1");
        generateFile("dir1/file1");
        generateFile("dir1/file2");
        fs::create_directory("dir1/dir2");
        generateFile("dir1/dir2/file3");
#ifdef TEST_LWG_2682_BEHAVIOUR
        REQUIRE_THROWS_AS(fs::copy("dir1", "dir3",
                                   fs::copy_options::create_symlinks |
                                   fs::copy_options::recursive),
                          fs::filesystem_error);
#else
        REQUIRE_NOTHROW(fs::copy("dir1", "dir3",
                                 fs::copy_options::create_symlinks |
                                     fs::copy_options::recursive));
        REQUIRE(!ec);
        REQUIRE(fs::exists("dir3/file1"));
        REQUIRE(fs::is_symlink("dir3/file1"));
        REQUIRE(fs::exists("dir3/file2"));
        REQUIRE(fs::is_symlink("dir3/file2"));
        REQUIRE(fs::exists("dir3/dir2/file3"));
        REQUIRE(fs::is_symlink("dir3/dir2/file3"));
#endif
    }
#ifndef TURBO_PLATFORM_WEB
    {
        TemporaryDirectory t(TempOpt::change_path);
        std::error_code ec;
        fs::create_directory("dir1");
        generateFile("dir1/file1");
        generateFile("dir1/file2");
        fs::create_directory("dir1/dir2");
        generateFile("dir1/dir2/file3");
        auto f1hl = fs::hard_link_count("dir1/file1");
        auto f2hl = fs::hard_link_count("dir1/file2");
        auto f3hl = fs::hard_link_count("dir1/dir2/file3");
        REQUIRE_NOTHROW(fs::copy(
                "dir1", "dir3",
                fs::copy_options::create_hard_links | fs::copy_options::recursive, ec));
        REQUIRE(!ec);
        REQUIRE(fs::exists("dir3/file1"));
        REQUIRE(fs::hard_link_count("dir1/file1") == f1hl + 1);
        REQUIRE(fs::exists("dir3/file2"));
        REQUIRE(fs::hard_link_count("dir1/file2") == f2hl + 1);
        REQUIRE(fs::exists("dir3/dir2/file3"));
        REQUIRE(fs::hard_link_count("dir1/dir2/file3") == f3hl + 1);
    }
#endif
}

TEST_CASE("Filesystem, copy_file") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo", 100);
    REQUIRE(!fs::exists("bar"));
    REQUIRE(fs::copy_file("foo", "bar"));
    REQUIRE(fs::exists("bar"));
    REQUIRE(fs::file_size("foo") == fs::file_size("bar"));
    REQUIRE(fs::copy_file("foo", "bar2", ec));
    REQUIRE(!ec);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    generateFile("foo2", 200);
    REQUIRE(fs::copy_file("foo2", "bar", fs::copy_options::update_existing));
    REQUIRE(fs::file_size("bar") == 200);
    REQUIRE(!fs::copy_file("foo", "bar", fs::copy_options::update_existing));
    REQUIRE(fs::file_size("bar") == 200);
    REQUIRE(
            fs::copy_file("foo", "bar", fs::copy_options::overwrite_existing));
    REQUIRE(fs::file_size("bar") == 100);
    REQUIRE_THROWS_AS(fs::copy_file("foobar", "foobar2"), fs::filesystem_error);
    REQUIRE_NOTHROW(fs::copy_file("foobar", "foobar2", ec));
    REQUIRE(ec);
    REQUIRE(!fs::exists("foobar"));
}

TEST_CASE("FilesystemDir, copy_symlink") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo");
    fs::create_directory("dir");
    if (is_symlink_creation_supported()) {
        fs::create_symlink("foo", "sfoo");
        fs::create_directory_symlink("dir", "sdir");
        REQUIRE_NOTHROW(fs::copy_symlink("sfoo", "sfooc"));
        REQUIRE(fs::exists("sfooc"));
        REQUIRE_NOTHROW(fs::copy_symlink("sfoo", "sfooc2", ec));
        REQUIRE(fs::exists("sfooc2"));
        REQUIRE(!ec);
        REQUIRE_NOTHROW(fs::copy_symlink("sdir", "sdirc"));
        REQUIRE(fs::exists("sdirc"));
        REQUIRE_NOTHROW(fs::copy_symlink("sdir", "sdirc2", ec));
        REQUIRE(fs::exists("sdirc2"));
        REQUIRE(!ec);
    }
    REQUIRE_THROWS_AS(fs::copy_symlink("bar", "barc"), fs::filesystem_error);
    REQUIRE_NOTHROW(fs::copy_symlink("bar", "barc", ec));
    REQUIRE(ec);
}

TEST_CASE("FilesystemDir, create_directories") {
    TemporaryDirectory t;
    fs::path p = t.path() / "testdir";
    fs::path p2 = p / "nested";
    REQUIRE(!fs::exists(p));
    REQUIRE(!fs::exists(p2));
    REQUIRE(fs::create_directories(p2));
    REQUIRE(fs::is_directory(p));
    REQUIRE(fs::is_directory(p2));
    REQUIRE(!fs::create_directories(p2));
#ifdef TEST_LWG_2935_BEHAVIOUR
    INFO("This test expects LWG #2935 result conformance.");
    p = t.path() / "testfile";
    generateFile(p);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    bool created = false;
    REQUIRE_NOTHROW((created = fs::create_directories(p)));
    REQUIRE(!created);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    std::error_code ec;
    REQUIRE_NOTHROW((created = fs::create_directories(p, ec)));
    REQUIRE(!created);
    REQUIRE(!ec);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    REQUIRE(!fs::create_directories(p, ec));
#else
    TLOG_INFO("This test expects conformance with P1164R1. (implemented "
              "by GCC with issue #86910.)");
    p = t.path() / "testfile";
    generateFile(p);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    REQUIRE_THROWS_AS(fs::create_directories(p), fs::filesystem_error);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    std::error_code ec;
    REQUIRE_NOTHROW(fs::create_directories(p, ec));
    REQUIRE(ec);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    REQUIRE(!fs::create_directories(p, ec));
#endif
}

TEST_CASE("FilesystemDir, create_directory") {
    TemporaryDirectory t;
    fs::path p = t.path() / "testdir";
    REQUIRE(!fs::exists(p));
    REQUIRE(fs::create_directory(p));
    REQUIRE(fs::is_directory(p));
    REQUIRE(!fs::is_regular_file(p));
    REQUIRE(fs::create_directory(p / "nested", p));
    REQUIRE(fs::is_directory(p / "nested"));
    REQUIRE(!fs::is_regular_file(p / "nested"));
#ifdef TEST_LWG_2935_BEHAVIOUR
    TURBO_LOG(INFO) << "This test expects LWG #2935 result conformance.";
    p = t.path() / "testfile";
    generateFile(p);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    bool created = false;
    REQUIRE_NOTHROW((created = fs::create_directory(p)));
    REQUIRE(!created);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    std::error_code ec;
    REQUIRE_NOTHROW((created = fs::create_directory(p, ec)));
    REQUIRE(!created);
    REQUIRE(!ec);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    REQUIRE(!fs::create_directories(p, ec));
#else
    TLOG_INFO("This test expects conformance with P1164R1. (implemented "
              "by GCC with issue #86910.)");
    p = t.path() / "testfile";
    generateFile(p);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    REQUIRE_THROWS_AS(fs::create_directory(p), fs::filesystem_error);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    std::error_code ec;
    REQUIRE_NOTHROW(fs::create_directory(p, ec));
    REQUIRE(ec);
    REQUIRE(fs::is_regular_file(p));
    REQUIRE(!fs::is_directory(p));
    REQUIRE(!fs::create_directory(p, ec));
#endif
}

TEST_CASE("FilesystemDir, create_directory_symlink") {
    if (is_symlink_creation_supported()) {
        TemporaryDirectory t;
        fs::create_directory(t.path() / "dir1");
        generateFile(t.path() / "dir1/test1");
        fs::create_directory(t.path() / "dir2");
        fs::create_directory_symlink(t.path() / "dir1", t.path() / "dir2/dirSym");
        REQUIRE(fs::exists(t.path() / "dir2/dirSym"));
        REQUIRE(fs::is_symlink(t.path() / "dir2/dirSym"));
        REQUIRE(fs::exists(t.path() / "dir2/dirSym/test1"));
        REQUIRE(fs::is_regular_file(t.path() / "dir2/dirSym/test1"));
        REQUIRE_THROWS_AS(fs::create_directory_symlink(t.path() / "dir1",
                                                       t.path() / "dir2/dirSym"),
                          fs::filesystem_error);
        std::error_code ec;
        REQUIRE_NOTHROW(fs::create_directory_symlink(t.path() / "dir1",
                                                     t.path() / "dir2/dirSym", ec));
        REQUIRE(ec);
    }
}

TEST_CASE("FilesystemDir, create_hard_link") {
#ifndef TURBO_PLATFORM_WEB
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo", 1234);
    REQUIRE_NOTHROW(fs::create_hard_link("foo", "bar"));
    REQUIRE(fs::exists("bar"));
    REQUIRE(!fs::is_symlink("bar"));
    REQUIRE_NOTHROW(fs::create_hard_link("foo", "bar2", ec));
    REQUIRE(fs::exists("bar2"));
    REQUIRE(!fs::is_symlink("bar2"));
    REQUIRE(!ec);
    REQUIRE_THROWS_AS(fs::create_hard_link("nofoo", "bar"), fs::filesystem_error);
    REQUIRE_NOTHROW(fs::create_hard_link("nofoo", "bar", ec));
    REQUIRE(ec);
#endif
}

TEST_CASE("FilesystemDir, create_symlink") {
    if (is_symlink_creation_supported()) {
        TemporaryDirectory t;
        fs::create_directory(t.path() / "dir1");
        generateFile(t.path() / "dir1/test1");
        fs::create_directory(t.path() / "dir2");
        fs::create_symlink(t.path() / "dir1/test1", t.path() / "dir2/fileSym");
        REQUIRE(fs::exists(t.path() / "dir2/fileSym"));
        REQUIRE(fs::is_symlink(t.path() / "dir2/fileSym"));
        REQUIRE(fs::exists(t.path() / "dir2/fileSym"));
        REQUIRE(fs::is_regular_file(t.path() / "dir2/fileSym"));
        REQUIRE_THROWS_AS(
                fs::create_symlink(t.path() / "dir1", t.path() / "dir2/fileSym"),
                fs::filesystem_error);
        std::error_code ec;
        REQUIRE_NOTHROW(
                fs::create_symlink(t.path() / "dir1", t.path() / "dir2/fileSym", ec));
        REQUIRE(ec);
    }
}

TEST_CASE("FilesystemDir, current_path") {
    TemporaryDirectory t;
    std::error_code ec;
    fs::path p1 = fs::current_path();
    REQUIRE_NOTHROW(fs::current_path(t.path()));
    REQUIRE(p1 != fs::current_path());
    REQUIRE_NOTHROW(fs::current_path(p1, ec));
    REQUIRE(!ec);
    REQUIRE_THROWS_AS(fs::current_path(t.path() / "foo"), fs::filesystem_error);
    REQUIRE(p1 == fs::current_path());
    REQUIRE_NOTHROW(fs::current_path(t.path() / "foo", ec));
    REQUIRE(ec);
}

TEST_CASE("FilesystemDir, equivalent") {
    TemporaryDirectory t(TempOpt::change_path);
    generateFile("foo", 1234);
    REQUIRE(fs::equivalent(t.path() / "foo", "foo"));
    if (is_symlink_creation_supported()) {
        std::error_code ec(42, std::system_category());
        fs::create_symlink("foo", "foo2");
        REQUIRE(fs::equivalent("foo", "foo2"));
        REQUIRE(fs::equivalent("foo", "foo2", ec));
        REQUIRE(!ec);
    }
#ifdef TEST_LWG_2937_BEHAVIOUR
    TLOG_INFO("This test expects LWG #2937 result conformance.");
    std::error_code ec;
    bool result = false;
    REQUIRE_THROWS_AS(fs::equivalent("foo", "foo3"), fs::filesystem_error);
    REQUIRE_NOTHROW(result = fs::equivalent("foo", "foo3", ec));
    REQUIRE(!result);
    REQUIRE(ec);
    ec.clear();
    REQUIRE_THROWS_AS(fs::equivalent("foo3", "foo"), fs::filesystem_error);
    REQUIRE_NOTHROW(result = fs::equivalent("foo3", "foo", ec));
    REQUIRE(!result);
    REQUIRE(ec);
    ec.clear();
    REQUIRE_THROWS_AS(fs::equivalent("foo3", "foo4"), fs::filesystem_error);
    REQUIRE_NOTHROW(result = fs::equivalent("foo3", "foo4", ec));
    REQUIRE(!result);
    REQUIRE(ec);
#else
    TURBO_LOG(INFO)
        << "This test expects conformance predating LWG #2937 result.";
    std::error_code ec;
    bool result = false;
    REQUIRE_NOTHROW(result = fs::equivalent("foo", "foo3"));
    REQUIRE(!result);
    REQUIRE_NOTHROW(result = fs::equivalent("foo", "foo3", ec));
    REQUIRE(!result);
    REQUIRE(!ec);
    ec.clear();
    REQUIRE_NOTHROW(result = fs::equivalent("foo3", "foo"));
    REQUIRE(!result);
    REQUIRE_NOTHROW(result = fs::equivalent("foo3", "foo", ec));
    REQUIRE(!result);
    REQUIRE(!ec);
    ec.clear();
    REQUIRE_THROWS_AS(result = fs::equivalent("foo4", "foo3"), fs::filesystem_error);
    REQUIRE(!result);
    REQUIRE_NOTHROW(result = fs::equivalent("foo4", "foo3", ec));
    REQUIRE(!result);
    REQUIRE(ec);
#endif
}

TEST_CASE("Filesystem, exists") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    REQUIRE(!fs::exists(""));
    REQUIRE(!fs::exists("foo"));
    REQUIRE(!fs::exists("foo", ec));
    REQUIRE(!ec);
    ec = std::error_code(42, std::system_category());
    REQUIRE(!fs::exists("foo", ec));
#if defined(__cpp_lib_char8_t) && !defined(TURBO_FILESYSTEM_ENFORCE_CPP17_API)
    REQUIRE(!fs::exists(u8"foo"));
#endif
    REQUIRE(!ec);
    ec.clear();
    REQUIRE(fs::exists(t.path()));
    REQUIRE(fs::exists(t.path(), ec));
    REQUIRE(!ec);
    ec = std::error_code(42, std::system_category());
    REQUIRE(fs::exists(t.path(), ec));
    REQUIRE(!ec);
#if defined(TURBO_PLATFORM_WINDOWS)
    if (::GetFileAttributesW(L"C:\\fs-test") != INVALID_FILE_ATTRIBUTES) {
      REQUIRE(fs::exists("C:\\fs-test"));
    }
#endif
}

TEST_CASE("Filesystem, file_size") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo", 0);
    generateFile("bar", 1234);
    REQUIRE(fs::file_size("foo") == 0);
    ec = std::error_code(42, std::system_category());
    REQUIRE(fs::file_size("foo", ec) == 0);
    REQUIRE(!ec);
    ec.clear();
    REQUIRE(fs::file_size("bar") == 1234);
    ec = std::error_code(42, std::system_category());
    REQUIRE(fs::file_size("bar", ec) == 1234);
    REQUIRE(!ec);
    ec.clear();
    REQUIRE_THROWS_AS(fs::file_size("foobar"), fs::filesystem_error);
    REQUIRE(fs::file_size("foobar", ec) == static_cast<uintmax_t>(-1));
    REQUIRE(ec);
    ec.clear();
}

#ifndef TURBO_PLATFORM_WINDOWS

static uintmax_t getHardlinkCount(const fs::path &p) {
    struct stat st = {};
    auto rc = ::lstat(p.c_str(), &st);
    return rc == 0 ? st.st_nlink : ~0u;
}

#endif

TEST_CASE("Filesystem, hard_link_count") {
#ifndef TURBO_PLATFORM_WEB
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
#ifdef TURBO_PLATFORM_WINDOWS
    // windows doesn't implement "."/".." as hardlinks, so it
    // starts with 1 and subdirectories don't change the count
    REQUIRE(fs::hard_link_count(t.path()) == 1);
    fs::create_directory("dir");
    REQUIRE(fs::hard_link_count(t.path()) == 1);
#else
    // unix/bsd/linux typically implements "."/".." as hardlinks
    // so an empty dir has 2 (from parent and the ".") and
    // adding a subdirectory adds one due to its ".."
    REQUIRE(fs::hard_link_count(t.path()) == getHardlinkCount(t.path()));
    fs::create_directory("dir");
    REQUIRE(fs::hard_link_count(t.path()) == getHardlinkCount(t.path()));
#endif
    generateFile("foo");
    REQUIRE(fs::hard_link_count(t.path() / "foo") == 1);
    ec = std::error_code(42, std::system_category());
    REQUIRE(fs::hard_link_count(t.path() / "foo", ec) == 1);
    REQUIRE(!ec);
    REQUIRE_THROWS_AS(fs::hard_link_count(t.path() / "bar"), fs::filesystem_error);
    REQUIRE_NOTHROW(fs::hard_link_count(t.path() / "bar", ec));
    REQUIRE(ec);
    ec.clear();
#else
    WARN("Test for unsupportet features are disabled on JS/Wasm target.");
#endif
}

class FileTypeMixFixture {
public:
    FileTypeMixFixture()
            : _t(TempOpt::change_path), _hasFifo(false), _hasSocket(false) {
        generateFile("regular");
        fs::create_directory("directory");
        if (is_symlink_creation_supported()) {
            fs::create_symlink("regular", "file_symlink");
            fs::create_directory_symlink("directory", "dir_symlink");
        }
#if !defined(TURBO_PLATFORM_WINDOWS) && !defined(TURBO_PLATFORM_WEB)
        CHECK(::mkfifo("fifo", 0644) == 0);
        _hasFifo = true;
        struct ::sockaddr_un addr;
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, "socket", sizeof(addr.sun_path));
        int fd = socket(PF_UNIX, SOCK_STREAM, 0);
        bind(fd, (struct sockaddr *) &addr, sizeof addr);
        _hasSocket = true;
#endif
    }

    ~FileTypeMixFixture() {}

    bool has_fifo() const { return _hasFifo; }

    bool has_socket() const { return _hasSocket; }

    fs::path block_path() const {
        std::error_code ec;
        if (fs::exists("/dev/sda", ec)) {
            return "/dev/sda";
        } else if (fs::exists("/dev/disk0", ec)) {
            return "/dev/disk0";
        }
        return fs::path();
    }

    fs::path character_path() const {
        std::error_code ec;
        if (fs::exists("/dev/null", ec)) {
            return "/dev/null";
        } else if (fs::exists("NUL", ec)) {
            return "NUL";
        }
        return fs::path();
    }

    fs::path temp_path() const { return _t.path(); }

private:
    TemporaryDirectory _t;
    bool _hasFifo;
    bool _hasSocket;
};


TEST_CASE_FIXTURE(FileTypeMixFixture, "is_block_file") {
    std::error_code ec;
    REQUIRE(!fs::is_block_file("directory"));
    REQUIRE(!fs::is_block_file("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(!fs::is_block_file("dir_symlink"));
        REQUIRE(!fs::is_block_file("file_symlink"));
    }
    REQUIRE((has_fifo() ? !fs::is_block_file("fifo") : true));
    REQUIRE((has_socket() ? !fs::is_block_file("socket") : true));
    REQUIRE((block_path().empty() ? true :
             fs::is_block_file(block_path())));
    REQUIRE((character_path().empty() ? true
                                      : !fs::is_block_file(character_path())));
    REQUIRE_NOTHROW(fs::is_block_file("notfound"));
    REQUIRE_NOTHROW(fs::is_block_file("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::not_found)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::regular)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::directory)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::symlink)));
    REQUIRE(fs::is_block_file(fs::file_status(fs::file_type::block)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::character)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::fifo)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::socket)));
    REQUIRE(!fs::is_block_file(fs::file_status(fs::file_type::unknown)));
}


TEST_CASE_FIXTURE(FileTypeMixFixture, "fs.op")
{
    std::error_code ec;
    REQUIRE(!fs::is_character_file("directory"));
    REQUIRE(!fs::is_character_file("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(!fs::is_character_file("dir_symlink"));
        REQUIRE(!fs::is_character_file("file_symlink"));
    }
    REQUIRE((has_fifo() ? !fs::is_character_file("fifo") : true));
    REQUIRE((has_socket() ? !fs::is_character_file("socket") : true));
    REQUIRE((block_path().empty() ? true :
             !fs::is_character_file(block_path())));
    REQUIRE((character_path().empty() ?
             true : fs::is_character_file(character_path())));
    REQUIRE_NOTHROW(fs::is_character_file("notfound"));
    REQUIRE_NOTHROW(fs::is_character_file("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::not_found)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::regular)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::directory)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::symlink)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::block)));
    REQUIRE(fs::is_character_file(fs::file_status(fs::file_type::character)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::fifo)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::socket)));
    REQUIRE(!fs::is_character_file(fs::file_status(fs::file_type::unknown)));
}

TEST_CASE_FIXTURE(FileTypeMixFixture, "fs.op.is_directory - is_directory")
{
    std::error_code ec;
    REQUIRE(fs::is_directory("directory"));
    REQUIRE(!fs::is_directory("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(fs::is_directory("dir_symlink"));
        REQUIRE(!fs::is_directory("file_symlink"));
    }
    REQUIRE((has_fifo() ? !fs::is_directory("fifo") : true));
    REQUIRE((has_socket() ? !fs::is_directory("socket") : true));
    REQUIRE((block_path().empty() ? true :
             !fs::is_directory(block_path())));
    REQUIRE((character_path().empty() ? true
                                      : !fs::is_directory(character_path())));
    REQUIRE_NOTHROW(fs::is_directory("notfound"));
    REQUIRE_NOTHROW(fs::is_directory("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::not_found)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::regular)));
    REQUIRE(fs::is_directory(fs::file_status(fs::file_type::directory)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::symlink)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::block)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::character)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::fifo)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::socket)));
    REQUIRE(!fs::is_directory(fs::file_status(fs::file_type::unknown)));
}

TEST_CASE("Filesystem, is_empty") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    REQUIRE(fs::is_empty(t.path()));
    REQUIRE(fs::is_empty(t.path(), ec));
    REQUIRE(!ec);
    generateFile("foo", 0);
    generateFile("bar", 1234);
    REQUIRE(fs::is_empty("foo"));
    REQUIRE(fs::is_empty("foo", ec));
    REQUIRE(!ec);
    REQUIRE(!fs::is_empty("bar"));
    REQUIRE(!fs::is_empty("bar", ec));
    REQUIRE(!ec);
    REQUIRE_THROWS_AS(fs::is_empty("foobar"), fs::filesystem_error);
    bool result = false;
    REQUIRE_NOTHROW(result = fs::is_empty("foobar", ec));
    REQUIRE(!result);
    REQUIRE(ec);
}


TEST_CASE_FIXTURE(FileTypeMixFixture, "fs.op.is_fifo - is_fifo")
{
    std::error_code ec;
    REQUIRE(!fs::is_fifo("directory"));
    REQUIRE(!fs::is_fifo("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(!fs::is_fifo("dir_symlink"));
        REQUIRE(!fs::is_fifo("file_symlink"));
    }
    REQUIRE((has_fifo() ? fs::is_fifo("fifo") : true));
    REQUIRE((has_socket() ? !fs::is_fifo("socket") : true));
    REQUIRE((block_path().empty() ? true : !fs::is_fifo(block_path())));
    REQUIRE((character_path().empty() ? true :
             !fs::is_fifo(character_path())));
    REQUIRE_NOTHROW(fs::is_fifo("notfound"));
    REQUIRE_NOTHROW(fs::is_fifo("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::not_found)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::regular)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::directory)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::symlink)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::block)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::character)));
    REQUIRE(fs::is_fifo(fs::file_status(fs::file_type::fifo)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::socket)));
    REQUIRE(!fs::is_fifo(fs::file_status(fs::file_type::unknown)));
}


TEST_CASE_FIXTURE(FileTypeMixFixture, "fs.op.is_other - is_other")
{
    std::error_code ec;
    REQUIRE(!fs::is_other("directory"));
    REQUIRE(!fs::is_other("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(!fs::is_other("dir_symlink"));
        REQUIRE(!fs::is_other("file_symlink"));
    }
    REQUIRE((has_fifo() ? fs::is_other("fifo") : true));
    REQUIRE((has_socket() ? fs::is_other("socket") : true));
    REQUIRE((block_path().empty() ? true : fs::is_other(block_path())));
    REQUIRE((character_path().empty() ? true :
             fs::is_other(character_path())));
    REQUIRE_NOTHROW(fs::is_other("notfound"));
    REQUIRE_NOTHROW(fs::is_other("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_other(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_other(fs::file_status(fs::file_type::not_found)));
    REQUIRE(!fs::is_other(fs::file_status(fs::file_type::regular)));
    REQUIRE(!fs::is_other(fs::file_status(fs::file_type::directory)));
    REQUIRE(!fs::is_other(fs::file_status(fs::file_type::symlink)));
    REQUIRE(fs::is_other(fs::file_status(fs::file_type::block)));
    REQUIRE(fs::is_other(fs::file_status(fs::file_type::character)));
    REQUIRE(fs::is_other(fs::file_status(fs::file_type::fifo)));
    REQUIRE(fs::is_other(fs::file_status(fs::file_type::socket)));
    REQUIRE(fs::is_other(fs::file_status(fs::file_type::unknown)));
}


TEST_CASE_FIXTURE(FileTypeMixFixture, "fs.op.is_regular_file - is_regular_file")
{
    std::error_code ec;
    REQUIRE(!fs::is_regular_file("directory"));
    REQUIRE(fs::is_regular_file("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(!fs::is_regular_file("dir_symlink"));
        REQUIRE(fs::is_regular_file("file_symlink"));
    }
    REQUIRE((has_fifo() ? !fs::is_regular_file("fifo") : true));
    REQUIRE((has_socket() ? !fs::is_regular_file("socket") : true));
    REQUIRE((block_path().empty() ? true :
             !fs::is_regular_file(block_path())));
    REQUIRE((character_path().empty() ?
             true : !fs::is_regular_file(character_path())));
    REQUIRE_NOTHROW(fs::is_regular_file("notfound"));
    REQUIRE_NOTHROW(fs::is_regular_file("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::not_found)));
    REQUIRE(fs::is_regular_file(fs::file_status(fs::file_type::regular)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::directory)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::symlink)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::block)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::character)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::fifo)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::socket)));
    REQUIRE(!fs::is_regular_file(fs::file_status(fs::file_type::unknown)));
}

TEST_CASE_FIXTURE(FileTypeMixFixture, "fs.op.is_socket - is_socket")
{
    std::error_code ec;
    REQUIRE(!fs::is_socket("directory"));
    REQUIRE(!fs::is_socket("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(!fs::is_socket("dir_symlink"));
        REQUIRE(!fs::is_socket("file_symlink"));
    }
    REQUIRE((has_fifo() ? !fs::is_socket("fifo") : true));
    REQUIRE((has_socket() ? fs::is_socket("socket") : true));
    REQUIRE((block_path().empty() ? true : !fs::is_socket(block_path())));
    REQUIRE((character_path().empty() ? true :
             !fs::is_socket(character_path())));
    REQUIRE_NOTHROW(fs::is_socket("notfound"));
    REQUIRE_NOTHROW(fs::is_socket("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::not_found)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::regular)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::directory)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::symlink)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::block)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::character)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::fifo)));
    REQUIRE(fs::is_socket(fs::file_status(fs::file_type::socket)));
    REQUIRE(!fs::is_socket(fs::file_status(fs::file_type::unknown)));
}

TEST_CASE_FIXTURE(FileTypeMixFixture, "fs.op.is_symlink - is_symlink")
{
    std::error_code ec;
    REQUIRE(!fs::is_symlink("directory"));
    REQUIRE(!fs::is_symlink("regular"));
    if (is_symlink_creation_supported()) {
        REQUIRE(fs::is_symlink("dir_symlink"));
        REQUIRE(fs::is_symlink("file_symlink"));
    }
    REQUIRE((has_fifo() ? !fs::is_symlink("fifo") : true));
    REQUIRE((has_socket() ? !fs::is_symlink("socket") : true));
    REQUIRE((block_path().empty() ? true : !fs::is_symlink(block_path())));
    REQUIRE((character_path().empty() ? true :
             !fs::is_symlink(character_path())));
    REQUIRE_NOTHROW(fs::is_symlink("notfound"));
    REQUIRE_NOTHROW(fs::is_symlink("notfound", ec));
    REQUIRE(ec);
    ec.clear();
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::none)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::not_found)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::regular)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::directory)));
    REQUIRE(fs::is_symlink(fs::file_status(fs::file_type::symlink)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::block)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::character)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::fifo)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::socket)));
    REQUIRE(!fs::is_symlink(fs::file_status(fs::file_type::unknown)));
}

#ifndef TURBO_PLATFORM_WEB

static fs::file_time_type timeFromString(const std::string &str) {
    struct ::tm tm;
    ::memset(&tm, 0, sizeof(::tm));
    std::istringstream is(str);
    is >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (is.fail()) {
        throw std::exception();
    }
    return from_time_t<fs::file_time_type>(std::mktime(&tm));
}

#endif

TEST_CASE("Filesystem, last_write_time") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    fs::file_time_type ft;
    generateFile("foo");
    auto now = fs::file_time_type::clock::now();
    REQUIRE(std::abs(std::chrono::duration_cast<std::chrono::seconds>(
            fs::last_write_time(t.path()) - now)
                             .count()) < 3);
    REQUIRE(std::abs(std::chrono::duration_cast<std::chrono::seconds>(
            fs::last_write_time("foo") - now)
                             .count()) < 3);
    REQUIRE_THROWS_AS(fs::last_write_time("bar"), fs::filesystem_error);
    REQUIRE_NOTHROW(ft = fs::last_write_time("bar", ec));
    REQUIRE(ft == fs::file_time_type::min());
    REQUIRE(ec);
    ec.clear();
    if (is_symlink_creation_supported()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        fs::create_symlink("foo", "foo2");
        ft = fs::last_write_time("foo");
        // checks that the time of the symlink is fetched
        REQUIRE(ft == fs::last_write_time("foo2"));
    }
#ifndef TURBO_PLATFORM_WEB
    auto nt = timeFromString("2015-10-21T04:30:00");
    REQUIRE_NOTHROW(fs::last_write_time(t.path() / "foo", nt));
    REQUIRE(std::abs(std::chrono::duration_cast<std::chrono::seconds>(
            fs::last_write_time("foo") - nt)
                             .count()) < 1);
    nt = timeFromString("2015-10-21T04:29:00");
    REQUIRE_NOTHROW(fs::last_write_time("foo", nt, ec));
    REQUIRE(std::abs(std::chrono::duration_cast<std::chrono::seconds>(
            fs::last_write_time("foo") - nt)
                             .count()) < 1);
    REQUIRE(!ec);
    REQUIRE_THROWS_AS(fs::last_write_time("bar", nt), fs::filesystem_error);
    REQUIRE_NOTHROW(fs::last_write_time("bar", nt, ec));
    REQUIRE(ec);
#endif
}

TEST_CASE("Filesystem, permissions") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo", 512);
    auto allWrite =
            fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write;
    REQUIRE_NOTHROW(fs::permissions("foo", allWrite, fs::perm_options::remove));
    REQUIRE((fs::status("foo").permissions() & fs::perms::owner_write) !=
            fs::perms::owner_write);
#if !defined(TURBO_PLATFORM_WINDOWS)
    if (geteuid() != 0)
#endif
    {
        REQUIRE_THROWS_AS(fs::resize_file("foo", 1024), fs::filesystem_error);
        REQUIRE(fs::file_size("foo") == 512);
    }
    REQUIRE_NOTHROW(
            fs::permissions("foo", fs::perms::owner_write, fs::perm_options::add));
    REQUIRE((fs::status("foo").permissions() & fs::perms::owner_write) ==
            fs::perms::owner_write);
    REQUIRE_NOTHROW(fs::resize_file("foo", 2048));
    REQUIRE(fs::file_size("foo") == 2048);
    REQUIRE_THROWS_AS(
            fs::permissions("bar", fs::perms::owner_write, fs::perm_options::add),
            fs::filesystem_error);
    REQUIRE_NOTHROW(fs::permissions("bar", fs::perms::owner_write,
                                    fs::perm_options::add, ec));
    REQUIRE(ec);
    REQUIRE_THROWS_AS(fs::permissions("bar", fs::perms::owner_write,
                                      static_cast<fs::perm_options>(0)),
                      fs::filesystem_error);
}

TEST_CASE("Filesystem, proximate") {
    std::error_code ec;
    REQUIRE(fs::proximate("/a/d", "/a/b/c") == "../../d");
    REQUIRE(fs::proximate("/a/d", "/a/b/c", ec) == "../../d");
    REQUIRE(!ec);
    REQUIRE(fs::proximate("/a/b/c", "/a/d") == "../b/c");
    REQUIRE(fs::proximate("/a/b/c", "/a/d", ec) == "../b/c");
    REQUIRE(!ec);
    REQUIRE(fs::proximate("a/b/c", "a") == "b/c");
    REQUIRE(fs::proximate("a/b/c", "a", ec) == "b/c");
    REQUIRE(!ec);
    REQUIRE(fs::proximate("a/b/c", "a/b/c/x/y") == "../..");
    REQUIRE(fs::proximate("a/b/c", "a/b/c/x/y", ec) == "../..");
    REQUIRE(!ec);
    REQUIRE(fs::proximate("a/b/c", "a/b/c") == ".");
    REQUIRE(fs::proximate("a/b/c", "a/b/c", ec) == ".");
    REQUIRE(!ec);
    REQUIRE(fs::proximate("a/b", "c/d") == "../../a/b");
    REQUIRE(fs::proximate("a/b", "c/d", ec) == "../../a/b");
    REQUIRE(!ec);
#ifndef TURBO_PLATFORM_WINDOWS
    if (has_host_root_name_support()) {
        REQUIRE(fs::proximate("//host1/a/d", "//host2/a/b/c") == "//host1/a/d");
        REQUIRE(fs::proximate("//host1/a/d", "//host2/a/b/c", ec) ==
                "//host1/a/d");
        REQUIRE(!ec);
    }
#endif
}

TEST_CASE("Filesystem, read_symlink") {
    if (is_symlink_creation_supported()) {
        TemporaryDirectory t(TempOpt::change_path);
        std::error_code ec;
        generateFile("foo");
        fs::create_symlink(t.path() / "foo", "bar");
        REQUIRE(fs::read_symlink("bar") == t.path() / "foo");
        REQUIRE(fs::read_symlink("bar", ec) == t.path() / "foo");
        REQUIRE(!ec);
        REQUIRE_THROWS_AS(fs::read_symlink("foobar"), fs::filesystem_error);
        REQUIRE(fs::read_symlink("foobar", ec) == fs::path());
        REQUIRE(ec);
    }
}

TEST_CASE("Filesystem, fs_op_relative") {
    REQUIRE(fs::relative("/a/d", "/a/b/c") == "../../d");
    REQUIRE(fs::relative("/a/b/c", "/a/d") == "../b/c");
    REQUIRE(fs::relative("a/b/c", "a") == "b/c");
    REQUIRE(fs::relative("a/b/c", "a/b/c/x/y") == "../..");
    REQUIRE(fs::relative("a/b/c", "a/b/c") == ".");
    REQUIRE(fs::relative("a/b", "c/d") == "../../a/b");
    std::error_code ec;
    REQUIRE(fs::relative(fs::current_path() / "foo", ec) == "foo");
    REQUIRE(!ec);
}

TEST_CASE("filesystem, remove") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo");
    REQUIRE(fs::remove("foo"));
    REQUIRE(!fs::exists("foo"));
    REQUIRE(!fs::remove("foo"));
    generateFile("foo");
    REQUIRE(fs::remove("foo", ec));
    REQUIRE(!fs::exists("foo"));
    if (is_symlink_creation_supported()) {
        generateFile("foo");
        fs::create_symlink("foo", "bar");
        REQUIRE(fs::exists(fs::symlink_status("bar")));
        REQUIRE(fs::remove("bar", ec));
        REQUIRE(fs::exists("foo"));
        REQUIRE(!fs::exists(fs::symlink_status("bar")));
    }
    REQUIRE(!fs::remove("bar"));
    REQUIRE(!fs::remove("bar", ec));
    REQUIRE(!ec);
}

TEST_CASE("Filesystem, remove_all") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo");
    REQUIRE(fs::remove_all("foo", ec) == 1);
    REQUIRE(!ec);
    ec.clear();
    REQUIRE(fs::directory_iterator(t.path()) == fs::directory_iterator());
    fs::create_directories("dir1/dir1a");
    fs::create_directories("dir1/dir1b");
    generateFile("dir1/dir1a/f1");
    generateFile("dir1/dir1b/f2");
    REQUIRE_NOTHROW(fs::remove_all("dir1/non-existing", ec));
    REQUIRE(!ec);
    REQUIRE(fs::remove_all("dir1/non-existing", ec) == 0);
    if (is_symlink_creation_supported()) {
        fs::create_directory_symlink("dir1", "dir1link");
        REQUIRE(fs::remove_all("dir1link") == 1);
    }
    REQUIRE(fs::remove_all("dir1") == 5);
    REQUIRE(fs::directory_iterator(t.path()) == fs::directory_iterator());
}

TEST_CASE("Filesystem, rename") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo", 123);
    fs::create_directory("dir1");
    REQUIRE_NOTHROW(fs::rename("foo", "bar"));
    REQUIRE(!fs::exists("foo"));
    REQUIRE(fs::exists("bar"));
    REQUIRE_NOTHROW(fs::rename("dir1", "dir2"));
    REQUIRE(fs::exists("dir2"));
    generateFile("foo2", 42);
    REQUIRE_NOTHROW(fs::rename("bar", "foo2"));
    REQUIRE(fs::exists("foo2"));
    REQUIRE(fs::file_size("foo2") == 123u);
    REQUIRE(!fs::exists("bar"));
    REQUIRE_NOTHROW(fs::rename("foo2", "foo", ec));
    REQUIRE(!ec);
    REQUIRE_THROWS_AS(fs::rename("foobar", "barfoo"), fs::filesystem_error);
    REQUIRE_NOTHROW(fs::rename("foobar", "barfoo", ec));
    REQUIRE(ec);
    REQUIRE(!fs::exists("barfoo"));
}

TEST_CASE("Filesystem, resize_file") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo", 1024);
    REQUIRE(fs::file_size("foo") == 1024);
    REQUIRE_NOTHROW(fs::resize_file("foo", 2048));
    REQUIRE(fs::file_size("foo") == 2048);
    REQUIRE_NOTHROW(fs::resize_file("foo", 1000, ec));
    REQUIRE(!ec);
    REQUIRE(fs::file_size("foo") == 1000);
    REQUIRE_THROWS_AS(fs::resize_file("bar", 2048), fs::filesystem_error);
    REQUIRE(!fs::exists("bar"));
    REQUIRE_NOTHROW(fs::resize_file("bar", 4096, ec));
    REQUIRE(ec);
    REQUIRE(!fs::exists("bar"));
}

TEST_CASE("Filesystem, fs_op_space") {
    {
        fs::space_info si;
        REQUIRE_NOTHROW(si = fs::space(fs::current_path()));
        REQUIRE(si.capacity > 1024 * 1024);
        REQUIRE(si.capacity > si.free);
        REQUIRE(si.free >= si.available);
    }
    {
        std::error_code ec;
        fs::space_info si;
        REQUIRE_NOTHROW(si = fs::space(fs::current_path(), ec));
        REQUIRE(si.capacity > 1024 * 1024);
        REQUIRE(si.capacity > si.free);
        REQUIRE(si.free >= si.available);
        REQUIRE(!ec);
    }
#ifndef TURBO_PLATFORM_WEB // statvfs under emscripten always returns a result,
    // so this tests would fail
    {
        std::error_code ec;
        fs::space_info si;
        REQUIRE_NOTHROW(si = fs::space("foobar42", ec));
        REQUIRE(si.capacity == static_cast<uintmax_t>(-1));
        REQUIRE(si.free == static_cast<uintmax_t>(-1));
        REQUIRE(si.available == static_cast<uintmax_t>(-1));
        REQUIRE(ec);
    }
    REQUIRE_THROWS_AS(fs::space("foobar42"), fs::filesystem_error);
#endif
}

TEST_CASE("Filesystem, op_and_status") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    fs::file_status fs;
    REQUIRE_NOTHROW(fs = fs::status("foo"));
    REQUIRE(fs.type() == fs::file_type::not_found);
    REQUIRE(fs.permissions() == fs::perms::unknown);
    REQUIRE_NOTHROW(fs = fs::status("bar", ec));
    REQUIRE(fs.type() == fs::file_type::not_found);
    REQUIRE(fs.permissions() == fs::perms::unknown);
    REQUIRE(ec);
    ec.clear();
    fs = fs::status(t.path());
    REQUIRE(fs.type() == fs::file_type::directory);
    REQUIRE(
            (fs.permissions() & (fs::perms::owner_read | fs::perms::owner_write)) ==
            (fs::perms::owner_read | fs::perms::owner_write));
    generateFile("foobar");
    fs = fs::status(t.path() / "foobar");
    REQUIRE(fs.type() == fs::file_type::regular);
    REQUIRE(
            (fs.permissions() & (fs::perms::owner_read | fs::perms::owner_write)) ==
            (fs::perms::owner_read | fs::perms::owner_write));
    if (is_symlink_creation_supported()) {
        fs::create_symlink(t.path() / "foobar", t.path() / "barfoo");
        fs = fs::status(t.path() / "barfoo");
        REQUIRE(fs.type() == fs::file_type::regular);
        REQUIRE(
                (fs.permissions() & (fs::perms::owner_read | fs::perms::owner_write)) ==
                (fs::perms::owner_read | fs::perms::owner_write));
    }
}

TEST_CASE("FilesystemStatus, status_known") {
    REQUIRE(!fs::status_known(fs::file_status()));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::not_found)));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::regular)));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::directory)));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::symlink)));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::character)));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::fifo)));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::socket)));
    REQUIRE(fs::status_known(fs::file_status(fs::file_type::unknown)));
}

TEST_CASE("FilesystemStatus, symlink_status") {
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    fs::file_status fs;
    REQUIRE_NOTHROW(fs = fs::symlink_status("foo"));
    REQUIRE(fs.type() == fs::file_type::not_found);
    REQUIRE(fs.permissions() == fs::perms::unknown);
    REQUIRE_NOTHROW(fs = fs::symlink_status("bar", ec));
    REQUIRE(fs.type() == fs::file_type::not_found);
    REQUIRE(fs.permissions() == fs::perms::unknown);
    REQUIRE(ec);
    ec.clear();
    fs = fs::symlink_status(t.path());
    REQUIRE(fs.type() == fs::file_type::directory);
    REQUIRE(
            (fs.permissions() & (fs::perms::owner_read | fs::perms::owner_write)) ==
            (fs::perms::owner_read | fs::perms::owner_write));
    generateFile("foobar");
    fs = fs::symlink_status(t.path() / "foobar");
    REQUIRE(fs.type() == fs::file_type::regular);
    REQUIRE(
            (fs.permissions() & (fs::perms::owner_read | fs::perms::owner_write)) ==
            (fs::perms::owner_read | fs::perms::owner_write));
    if (is_symlink_creation_supported()) {
        fs::create_symlink(t.path() / "foobar", t.path() / "barfoo");
        fs = fs::symlink_status(t.path() / "barfoo");
        REQUIRE(fs.type() == fs::file_type::symlink);
    }
}

TEST_CASE("FilesystemStatus, temp_dir_path") {
    std::error_code ec;
    REQUIRE_NOTHROW(fs::exists(fs::temp_directory_path()));
    REQUIRE_NOTHROW(fs::exists(fs::temp_directory_path(ec)));
    REQUIRE(!fs::temp_directory_path().empty());
    REQUIRE(!ec);
}

TEST_CASE("FilesystemStatus, weakly_canonical") {
    TLOG_INFO("This might fail on std::implementations that return "
              "fs::current_path() for fs::canonical(\"\")");
    REQUIRE(fs::weakly_canonical("") == ".");
    if (fs::weakly_canonical("") == ".") {
        REQUIRE(fs::weakly_canonical("foo/bar") == "foo/bar");
        REQUIRE(fs::weakly_canonical("foo/./bar") == "foo/bar");
        REQUIRE(fs::weakly_canonical("foo/../bar") == "bar");
    } else {
        REQUIRE(fs::weakly_canonical("foo/bar") ==
                fs::current_path() / "foo/bar");
        REQUIRE(fs::weakly_canonical("foo/./bar") ==
                fs::current_path() / "foo/bar");
        REQUIRE(fs::weakly_canonical("foo/../bar") ==
                fs::current_path() / "bar");
    }

    {
        TemporaryDirectory t(TempOpt::change_path);
        auto dir = t.path() / "d0";
        fs::create_directories(dir / "d1");
        generateFile(dir / "f0");
        fs::path rel(dir.filename());
        REQUIRE(fs::weakly_canonical(dir) == dir);
        REQUIRE(fs::weakly_canonical(rel) == dir);
        REQUIRE(fs::weakly_canonical(dir / "f0") == dir / "f0");
        REQUIRE(fs::weakly_canonical(dir / "f0/") == dir / "f0/");
        REQUIRE(fs::weakly_canonical(dir / "f1") == dir / "f1");
        REQUIRE(fs::weakly_canonical(rel / "f0") == dir / "f0");
        REQUIRE(fs::weakly_canonical(rel / "f0/") == dir / "f0/");
        REQUIRE(fs::weakly_canonical(rel / "f1") == dir / "f1");
        REQUIRE(fs::weakly_canonical(rel / "./f0") == dir / "f0");
        REQUIRE(fs::weakly_canonical(rel / "./f1") == dir / "f1");
        REQUIRE(fs::weakly_canonical(rel / "d1/../f0") == dir / "f0");
        REQUIRE(fs::weakly_canonical(rel / "d1/../f1") == dir / "f1");
        REQUIRE(fs::weakly_canonical(rel / "d1/../f1/../f2") == dir / "f2");
    }
}

TEST_CASE("FilesystemStatus, string_view") {

    using std::string_view;
    using std::wstring_view;
    {
        std::string p("foo/bar");
        string_view sv(p);
        REQUIRE(
                fs::path(sv, fs::path::format::generic_format).generic_string() ==
                "foo/bar");
        fs::path p2("fo");
        p2 += string_view("o");
        REQUIRE(p2 == "foo");
        REQUIRE(p2.compare(string_view("foo")) == 0);
    }
    {
        auto p = fs::path{"XYZ"};
        p /= string_view("Appendix");
        REQUIRE(p == "XYZ/Appendix");
    }
    {
        std::wstring p(L"foo/bar");
        wstring_view sv(p);
        REQUIRE(
                fs::path(sv, fs::path::format::generic_format).generic_string() ==
                "foo/bar");
        fs::path p2(L"fo");
        p2 += wstring_view(L"o");
        REQUIRE(p2 == "foo");
        REQUIRE(p2.compare(wstring_view(L"foo")) == 0);
    }
}

TEST_CASE("FilesystemStatus, win_long") {
#ifdef TURBO_PLATFORM_WINDOWS
    TemporaryDirectory t(TempOpt::change_path);
    char c = 'A';
    fs::path dir{"\\\\?\\"};
    dir += fs::current_path().u8string();
    for (; c <= 'Z'; ++c) {
      std::string part = std::string(16, c);
      dir /= part;
      REQUIRE_NOTHROW(fs::create_directory(dir));
      REQUIRE(fs::exists(dir));
      generateFile(dir / "f0");
      REQUIRE(fs::exists(dir / "f0"));
    }
    REQUIRE(c > 'Z');
    fs::remove_all(fs::current_path() / std::string(16, 'A'));
    REQUIRE(!fs::exists(fs::current_path() / std::string(16, 'A')));
    REQUIRE_NOTHROW(fs::create_directories(dir));
    REQUIRE(fs::exists(dir));
    generateFile(dir / "f0");
    REQUIRE(fs::exists(dir / "f0"));
#else
    TLOG_WARN("Windows specific tests are empty on non-Windows systems.");
#endif
}

TEST_CASE("Filesystem, win_namespaces") {
#ifdef TURBO_PLATFORM_WINDOWS
    {
      std::error_code ec;
      fs::path p(R"(\\localhost\c$\Windows)");
      auto symstat = fs::symlink_status(p, ec);
      REQUIRE(!ec);
      auto p2 = fs::canonical(p, ec);
      REQUIRE(!ec);
      REQUIRE(p2 == p);
    }

    struct TestInfo {
      std::string _path;
      std::string _string;
      std::string _rootName;
      std::string _rootPath;
      std::string _iterateResult;
    };
    std::vector<TestInfo> variants = {
        {R"(C:\Windows\notepad.exe)", R"(C:\Windows\notepad.exe)", "C:", "C:\\",
         "C:,/,Windows,notepad.exe"},
#ifdef USE_STD_FS
        {R"(\\?\C:\Windows\notepad.exe)", R"(\\?\C:\Windows\notepad.exe)",
         "\\\\?", "\\\\?\\", "//?,/,C:,Windows,notepad.exe"},
        {R"(\??\C:\Windows\notepad.exe)", R"(\??\C:\Windows\notepad.exe)", "\\??",
         "\\??\\", "/??,/,C:,Windows,notepad.exe"},
#else
        {R"(\\?\C:\Windows\notepad.exe)", R"(\\?\C:\Windows\notepad.exe)",
         "C:", "C:\\", "//?/,C:,/,Windows,notepad.exe"},
        {R"(\??\C:\Windows\notepad.exe)", R"(\??\C:\Windows\notepad.exe)",
         "C:", "C:\\", "/?\?/,C:,/,Windows,notepad.exe"},
#endif
        {R"(\\.\C:\Windows\notepad.exe)", R"(\\.\C:\Windows\notepad.exe)",
         "\\\\.", "\\\\.\\", "//.,/,C:,Windows,notepad.exe"},
        {R"(\\?\HarddiskVolume1\Windows\notepad.exe)",
         R"(\\?\HarddiskVolume1\Windows\notepad.exe)", "\\\\?", "\\\\?\\",
         "//?,/,HarddiskVolume1,Windows,notepad.exe"},
        {R"(\\?\Harddisk0Partition1\Windows\notepad.exe)",
         R"(\\?\Harddisk0Partition1\Windows\notepad.exe)", "\\\\?", "\\\\?\\",
         "//?,/,Harddisk0Partition1,Windows,notepad.exe"},
        {R"(\\.\GLOBALROOT\Device\HarddiskVolume1\Windows\notepad.exe)",
         R"(\\.\GLOBALROOT\Device\HarddiskVolume1\Windows\notepad.exe)", "\\\\.",
         "\\\\.\\",
         "//.,/,GLOBALROOT,Device,HarddiskVolume1,Windows,notepad.exe"},
        {R"(\\?\GLOBALROOT\Device\Harddisk0\Partition1\Windows\notepad.exe)",
         R"(\\?\GLOBALROOT\Device\Harddisk0\Partition1\Windows\notepad.exe)",
         "\\\\?", "\\\\?\\",
         "//?,/,GLOBALROOT,Device,Harddisk0,Partition1,Windows,notepad.exe"},
        {R"(\\?\Volume{e8a4a89d-0000-0000-0000-100000000000}\Windows\notepad.exe)",
         R"(\\?\Volume{e8a4a89d-0000-0000-0000-100000000000}\Windows\notepad.exe)",
         "\\\\?", "\\\\?\\",
         "//?,/"
         ",Volume{e8a4a89d-0000-0000-0000-100000000000},Windows,notepad.exe"},
        {R"(\\LOCALHOST\C$\Windows\notepad.exe)",
         R"(\\LOCALHOST\C$\Windows\notepad.exe)", "\\\\LOCALHOST",
         "\\\\LOCALHOST\\", "//LOCALHOST,/,C$,Windows,notepad.exe"},
        {R"(\\?\UNC\C$\Windows\notepad.exe)", R"(\\?\UNC\C$\Windows\notepad.exe)",
         "\\\\?", "\\\\?\\", "//?,/,UNC,C$,Windows,notepad.exe"},
        {R"(\\?\GLOBALROOT\Device\Mup\C$\Windows\notepad.exe)",
         R"(\\?\GLOBALROOT\Device\Mup\C$\Windows\notepad.exe)", "\\\\?",
         "\\\\?\\", "//?,/,GLOBALROOT,Device,Mup,C$,Windows,notepad.exe"},
    };

    for (auto ti : variants) {
      INFO("Used path: " + ti._path);
      auto p = fs::path(ti._path);
      REQUIRE(p.string() == ti._string);
      REQUIRE(p.is_absolute());
      REQUIRE(p.root_name().string() == ti._rootName);
      REQUIRE(p.root_path().string() == ti._rootPath);
      REQUIRE(iterateResult(p) == ti._iterateResult);
    }
#else
    TLOG_WARN("Windows specific tests are empty on non-Windows systems.");
#endif
}

TEST_CASE("Filesystem, win_mapped") {
#ifdef TURBO_PLATFORM_WINDOWS
    // this test expects a mapped volume on C:\\fs-test as is the case on the
    // development test system does nothing on other systems
    if (fs::exists("C:\\fs-test")) {
      REQUIRE(fs::canonical("C:\\fs-test\\Test.txt").string() ==
                  "C:\\fs-test\\Test.txt");
    }
#else
    TLOG_WARN("Windows specific tests are empty on non-Windows systems.");
#endif
}

TEST_CASE("Filesystem, win_remove") {
#ifdef TURBO_PLATFORM_WINDOWS
    TemporaryDirectory t(TempOpt::change_path);
    std::error_code ec;
    generateFile("foo", 512);
    auto allWrite =
        fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write;
    REQUIRE_NOTHROW(fs::permissions("foo", allWrite, fs::perm_options::remove));
    REQUIRE_NOTHROW(fs::remove("foo"));
    REQUIRE(!fs::exists("foo"));
#else
    TLOG_WARN("Windows specific tests are empty on non-Windows systems.");
#endif
}
