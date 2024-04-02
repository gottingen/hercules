// A simple example to capture the following task dependencies.
//
//           +---+
//     +---->| B |-----+
//     |     +---+     |
//   +---+           +-v-+
//   | A |           | D |
//   +---+           +-^-+
//     |     +---+     |
//     +---->| C |-----+
//           +---+
//
#include <collie/taskflow/taskflow.h>  // the only include you need

int main(){

  collie::tf::Executor executor;
  collie::tf::Taskflow taskflow("simple");

  auto [A, B, C, D] = taskflow.emplace(
    []() { std::cout << "TaskA\n"; },
    []() { std::cout << "TaskB\n"; },
    []() { std::cout << "TaskC\n"; },
    []() { std::cout << "TaskD\n"; }
  );

  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C

  executor.run(taskflow).wait();

  return 0;
}
