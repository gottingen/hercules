#include <algorithm>

#include "hercules/cir/cir.h"
#include "hercules/cir/util/cloning.h"
#include "gtest/gtest.h"

class CIRCoreTest : public testing::Test {
protected:
  std::unique_ptr<hercules::ir::Module> module;
  std::unique_ptr<hercules::ir::util::CloneVisitor> cv;

  void SetUp() override {
    hercules::ir::IdMixin::resetId();
    module = std::make_unique<hercules::ir::Module>("test");
    cv = std::make_unique<hercules::ir::util::CloneVisitor>(module.get());
  }
};
