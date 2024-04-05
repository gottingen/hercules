#include <collie/taskflow/taskflow.h>

int spawn(int n, collie::tf::Subflow& sbf) {
  if (n < 2) return n;
  int res1, res2;

  // compute f(n-1)
  sbf.emplace([&res1, n] (collie::tf::Subflow& sbf_n_1) { res1 = spawn(n - 1, sbf_n_1); } )
     .name(std::to_string(n-1));

  // compute f(n-2)
  sbf.emplace([&res2, n] (collie::tf::Subflow& sbf_n_2) { res2 = spawn(n - 2, sbf_n_2); } )
     .name(std::to_string(n-2));

  sbf.join();
  return res1 + res2;
}

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./fibonacci N\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);

  if(N < 0) {
    throw std::runtime_error("N must be non-negative");
  }

  int res;  // result

  collie::tf::Executor executor;
  collie::tf::Taskflow taskflow("fibonacci");

  taskflow.emplace([&res, N] (collie::tf::Subflow& sbf) {
    res = spawn(N, sbf);
  }).name(std::to_string(N));

  executor.run(taskflow).wait();

  //taskflow.dump(std::cout);

  std::cout << "Fib[" << N << "]: " << res << std::endl;

  return 0;
}









