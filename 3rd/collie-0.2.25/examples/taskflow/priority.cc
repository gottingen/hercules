// This program demonstrates how to set priority to a task.
//
// Currently, Taskflow supports only three priority levels:
//   + collie::tf::TaskPriority::HIGH   (numerical value = 0)
//   + collie::tf::TaskPriority::NORMAL (numerical value = 1)
//   + collie::tf::TaskPriority::LOW    (numerical value = 2)
// 
// Priority-based execution is non-preemptive. Once a task 
// has started to execute, it will execute to completion,
// even if a higher priority task has been spawned or enqueued. 

#include <collie/taskflow/taskflow.h>

int main() {
  
  // create an executor of only one worker to enable 
  // deterministic behavior
  collie::tf::Executor executor(1);

  collie::tf::Taskflow taskflow;

  int counter {0};
  
  // Here we create five tasks and print thier execution
  // orders which should align with assigned priorities
  auto [A, B, C, D, E] = taskflow.emplace(
    [] () { },
    [&] () { 
      std::cout << "Task B: " << counter++ << '\n';  // 0
    },
    [&] () { 
      std::cout << "Task C: " << counter++ << '\n';  // 2
    },
    [&] () { 
      std::cout << "Task D: " << counter++ << '\n';  // 1
    },
    [] () { }
  );

  A.precede(B, C, D); 
  E.succeed(B, C, D);
  
  // By default, all tasks are of collie::tf::TaskPriority::HIGH
  B.priority(collie::tf::TaskPriority::HIGH);
  C.priority(collie::tf::TaskPriority::LOW);
  D.priority(collie::tf::TaskPriority::NORMAL);

  assert(B.priority() == collie::tf::TaskPriority::HIGH);
  assert(C.priority() == collie::tf::TaskPriority::LOW);
  assert(D.priority() == collie::tf::TaskPriority::NORMAL);
  
  // we should see B, D, and C in their priority order
  executor.run(taskflow).wait();
}

