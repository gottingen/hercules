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

#include <vector>

namespace hercules {
namespace ir {
namespace util {

/// Base for CIR visitor contexts.
template <typename Frame> class CIRContext {
private:
  std::vector<Frame> frames;

public:
  /// Emplaces a frame onto the stack.
  /// @param args a parameter pack of the arguments
  template <typename... Args> void emplaceFrame(Args... args) {
    frames.emplace_back(args...);
  }
  /// Replaces a frame.
  /// @param newFrame the new frame
  void replaceFrame(Frame newFrame) {
    frames.pop_back();
    frames.push_back(newFrame);
  }
  /// @return all frames
  std::vector<Frame> &getFrames() { return frames; }
  /// @return the current frame
  Frame &getFrame() { return frames.back(); }
  /// Pops a frame.
  void popFrame() { return frames.pop_back(); }
};

} // namespace util
} // namespace ir
} // namespace hercules
