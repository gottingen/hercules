// This program demonstrates how to use multi-condition task
// to jump to multiple successor tasks
//
// A ----> B
//   |
//   |---> C
//   |
//   |---> D
//
#include <collie/taskflow/taskflow.h>

int main() {

  collie::tf::Executor executor;
  collie::tf::Taskflow taskflow("Multi-Conditional Tasking Demo");

  auto A = taskflow.emplace([&]() -> collie::InlinedVector<int> {
    std::cout << "A\n";
    return {0, 2};
  }).name("A");
  auto B = taskflow.emplace([&](){ std::cout << "B\n"; }).name("B");
  auto C = taskflow.emplace([&](){ std::cout << "C\n"; }).name("C");
  auto D = taskflow.emplace([&](){ std::cout << "D\n"; }).name("D");

  A.precede(B, C, D);

  // visualizes the taskflow
  taskflow.dump(std::cout);

  // executes the taskflow
  executor.run(taskflow).wait();

  return 0;
}

