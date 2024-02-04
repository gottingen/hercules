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

#include "hercules/cir/cir.h"
#include "hercules/cir/transform/manager.h"
#include "hercules/cir/transform/pass.h"
#include "hercules/parser/cache.h"
#include "llvm/Passes/PassBuilder.h"
#include <functional>
#include <string>
#include <vector>

namespace hercules {

/// Base class for DSL plugins. Plugins will return an instance of
/// a child of this class, which defines various characteristics of
/// the DSL, like keywords and IR passes.
class DSL {
public:
  /// General information about this plugin.
  struct Info {
    /// Extension name
    std::string name;
    /// Extension description
    std::string description;
    /// Extension version
    std::string version;
    /// Extension URL
    std::string url;
    /// Supported Hercules versions (semver range)
    std::string supported;
    /// Plugin stdlib path
    std::string stdlibPath;
    /// Plugin dynamic library path
    std::string dylibPath;
    /// Linker arguments (to replace "-l dylibPath" if present)
    std::vector<std::string> linkArgs;
  };

  using KeywordCallback =
      std::function<ast::StmtPtr(ast::SimplifyVisitor *, ast::CustomStmt *)>;

  struct ExprKeyword {
    std::string keyword;
    KeywordCallback callback;
  };

  struct BlockKeyword {
    std::string keyword;
    KeywordCallback callback;
    bool hasExpr;
  };

  virtual ~DSL() noexcept = default;

  /// Registers this DSL's IR passes with the given pass manager.
  /// @param pm the pass manager to add the passes to
  /// @param debug true if compiling in debug mode
  virtual void addIRPasses(ir::transform::PassManager *pm, bool debug) {}

  /// Registers this DSL's LLVM passes with the given pass builder.
  /// @param pb the pass builder to add the passes to
  /// @param debug true if compiling in debug mode
  virtual void addLLVMPasses(llvm::PassBuilder *pb, bool debug) {}

  /// Returns a vector of "expression keywords", defined as keywords of
  /// the form "keyword <expr>".
  /// @return this DSL's expression keywords
  virtual std::vector<ExprKeyword> getExprKeywords() { return {}; }

  /// Returns a vector of "block keywords", defined as keywords of the
  /// form "keyword <expr>: <block of code>".
  /// @return this DSL's block keywords
  virtual std::vector<BlockKeyword> getBlockKeywords() { return {}; }
};

} // namespace hercules
