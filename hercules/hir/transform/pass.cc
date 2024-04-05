// Copyright 2024 The EA Authors.
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

#include <hercules/hir/transform/pass.h>
#include <hercules/hir/transform/manager.h>

namespace hercules::ir::transform {

    analyze::Result *Pass::doGetAnalysis(const std::string &key) {
        return manager ? manager->getAnalysisResult(key) : nullptr;
    }

    void PassGroup::run(Module *module) {
        for (auto &p: passes)
            p->run(module);
    }

    void PassGroup::setManager(PassManager *mng) {
        Pass::setManager(mng);
        for (auto &p: passes)
            p->setManager(mng);
    }

}  // namespace hercules::ir::transform
