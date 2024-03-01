// Copyright 2023 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#pragma once

#include <chrono>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <iostream>
#include <ostream>

#include "hercules/compiler/error.h"
#include "hercules/config/config.h"
#include "hercules/parser/ast/error.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#define DBG(c, ...)                                                                    \
  fmt::print(hercules::getLogger().log, "{}" c "\n",                                      \
             std::string(size_t(2) * size_t(hercules::getLogger().level), ' '),           \
             ##__VA_ARGS__)
#define LOG(c, ...) DBG(c, ##__VA_ARGS__)
#define LOG_TIME(c, ...)                                                               \
  {                                                                                    \
    if (hercules::getLogger().flags & hercules::Logger::FLAG_TIME)                           \
      DBG(c, ##__VA_ARGS__);                                                           \
  }
#define LOG_REALIZE(c, ...)                                                            \
  {                                                                                    \
    if (hercules::getLogger().flags & hercules::Logger::FLAG_REALIZE)                        \
      DBG(c, ##__VA_ARGS__);                                                           \
  }
#define LOG_TYPECHECK(c, ...)                                                          \
  {                                                                                    \
    if (hercules::getLogger().flags & hercules::Logger::FLAG_TYPECHECK)                      \
      DBG(c, ##__VA_ARGS__);                                                           \
  }
#define LOG_IR(c, ...)                                                                 \
  {                                                                                    \
    if (hercules::getLogger().flags & hercules::Logger::FLAG_IR)                             \
      DBG(c, ##__VA_ARGS__);                                                           \
  }
#define LOG_USER(c, ...)                                                               \
  {                                                                                    \
    if (hercules::getLogger().flags & hercules::Logger::FLAG_USER)                           \
      DBG(c, ##__VA_ARGS__);                                                           \
  }

#define TIME(name) hercules::Timer __timer(name)

#ifndef NDEBUG
#define seqassertn(expr, msg, ...)                                                     \
  ((expr) ? (void)(0)                                                                  \
          : hercules::assertionFailure(#expr, __FILE__, __LINE__,                         \
                                    fmt::format(msg, ##__VA_ARGS__)))
#define seqassert(expr, msg, ...)                                                      \
  ((expr) ? (void)(0)                                                                  \
          : hercules::assertionFailure(                                                   \
                #expr, __FILE__, __LINE__,                                             \
                fmt::format(msg " [{}]", ##__VA_ARGS__, getSrcInfo())))
#else
#define seqassertn(expr, msg, ...) ;
#define seqassert(expr, msg, ...) ;
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace hercules {

void assertionFailure(const char *expr_str, const char *file, int line,
                      const std::string &msg);

struct Logger {
  static constexpr int FLAG_TIME = (1 << 0);
  static constexpr int FLAG_REALIZE = (1 << 1);
  static constexpr int FLAG_TYPECHECK = (1 << 2);
  static constexpr int FLAG_IR = (1 << 3);
  static constexpr int FLAG_USER = (1 << 4);

  int flags;
  int level;
  std::ostream &out;
  std::ostream &err;
  std::ostream &log;

  Logger() : flags(0), level(0), out(std::cout), err(std::cerr), log(std::clog) {}

  void parse(const std::string &logs);
};

Logger &getLogger();
void pushLogger();
bool popLogger();

class Timer {
private:
  using clock_type = std::chrono::high_resolution_clock;
  std::string name;
  std::chrono::time_point<clock_type> start, end;

public:
  bool logged;

public:
  void log() {
    if (!logged) {
      LOG_TIME("[T] {} = {:.3f}", name, elapsed());
      logged = true;
    }
  }

  double elapsed(std::chrono::time_point<clock_type> end = clock_type::now()) const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() /
           1000.0;
  }

  Timer(std::string name) : name(std::move(name)), start(), end(), logged(false) {
    start = clock_type::now();
  }

  ~Timer() { log(); }
};

std::ostream &operator<<(std::ostream &out, const hercules::SrcInfo &src);

struct SrcObject {
private:
  SrcInfo info;

public:
  SrcObject() : info() {}
  SrcObject(const SrcObject &s) { setSrcInfo(s.getSrcInfo()); }

  virtual ~SrcObject() = default;

  SrcInfo getSrcInfo() const { return info; }

  void setSrcInfo(SrcInfo info) { this->info = std::move(info); }
};
template <class... TA> void E(error::Error e, hercules::SrcObject *o, const TA &...args) {
  E(e, o->getSrcInfo(), args...);
}
template <class... TA>
void E(error::Error e, const hercules::SrcObject &o, const TA &...args) {
  E(e, o.getSrcInfo(), args...);
}
template <class... TA>
void E(error::Error e, const std::shared_ptr<SrcObject> &o, const TA &...args) {
  E(e, o->getSrcInfo(), args...);
}

enum MessageGroupPos {
  NONE = 0,
  HEAD,
  MID,
  LAST,
};

void compilationError(const std::string &msg, const std::string &file = "",
                      int line = 0, int col = 0, int len = 0, int errorCode = -1,
                      bool terminate = true, MessageGroupPos pos = NONE);

void compilationWarning(const std::string &msg, const std::string &file = "",
                        int line = 0, int col = 0, int len = 0, int errorCode = -1,
                        bool terminate = false, MessageGroupPos pos = NONE);

} // namespace hercules

template <> struct fmt::formatter<hercules::SrcInfo> : fmt::ostream_formatter {};
