// This example demonstrates how to use the corun
// method in the executor.
#include <collie/taskflow/taskflow.h>

int main(){
  
  const size_t N = 100;
  const size_t T = 1000;
  
  // create an executor and a taskflow
  collie::tf::Executor executor(2);
  collie::tf::Taskflow taskflow;

  std::array<collie::tf::Taskflow, N> taskflows;

  std::atomic<size_t> counter{0};
  
  for(size_t n=0; n<N; n++) {
    for(size_t i=0; i<T; i++) {
      taskflows[n].emplace([&](){ counter++; });
    }
    taskflow.emplace([&executor, &tf=taskflows[n]](){
      executor.corun(tf);
      //executor.run(tf).wait();  <-- can result in deadlock
    });
  }

  executor.run(taskflow).wait();

  return 0;
}
