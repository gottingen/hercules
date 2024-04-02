// This program demonstrate how to perform a parallel scan
// using cudaFlow.

#include <collie/taskflow/cuda/cudaflow.h>
#include <collie/taskflow/cuda/algorithm/scan.h>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_scan N\n";
    std::exit(EXIT_FAILURE);
  }

  int N = std::atoi(argv[1]);

  auto data1 = collie::tf::cuda_malloc_shared<int>(N);
  auto data2 = collie::tf::cuda_malloc_shared<int>(N);
  auto scan1 = collie::tf::cuda_malloc_shared<int>(N);
  auto scan2 = collie::tf::cuda_malloc_shared<int>(N);

  // --------------------------------------------------------------------------
  // inclusive/exclusive scan
  // --------------------------------------------------------------------------

  // initialize the data
  std::iota(data1, data1 + N, 0);
  std::iota(data2, data2 + N, 0);
  
  collie::tf::cudaStream stream;
  collie::tf::cudaDefaultExecutionPolicy policy(stream);

  // declare the buffer
  void* buff;
  cudaMalloc(&buff, policy.scan_bufsz<int>(N));
  
  // create inclusive and exclusive scan tasks
  collie::tf::cuda_inclusive_scan(policy, data1, data1+N, scan1, collie::tf::cuda_plus<int>{}, buff);
  collie::tf::cuda_exclusive_scan(policy, data2, data2+N, scan2, collie::tf::cuda_plus<int>{}, buff);

  stream.synchronize();
  
  // inspect 
  for(int i=1; i<N; i++) {
    if(scan1[i] != scan1[i-1] + data1[i]) {
      throw std::runtime_error("incorrect inclusive scan result");
    }
    if(scan2[i] != scan2[i-1] + data2[i-1]) {
      throw std::runtime_error("incorrect exclusive scan result");
    }
  }

  std::cout << "scan done\n";
  
  // --------------------------------------------------------------------------
  // transform inclusive/exclusive scan
  // --------------------------------------------------------------------------
  
  // initialize the data
  std::iota(data1, data1 + N, 0);
  std::iota(data2, data2 + N, 0);
  
  // transform inclusive scan
  collie::tf::cuda_transform_inclusive_scan(policy,
    data1, data1+N, scan1, collie::tf::cuda_plus<int>{},
    [] __device__ (int a) { return a*10; },
    buff
  );

  // transform exclusive scan
  collie::tf::cuda_transform_exclusive_scan(policy,
    data2, data2+N, scan2, collie::tf::cuda_plus<int>{},
    [] __device__ (int a) { return a*11; },
    buff
  );
  
  stream.synchronize();
  
  // inspect 
  for(int i=1; i<N; i++) {
    if(scan1[i] != scan1[i-1] + data1[i] * 10) {
      throw std::runtime_error("incorrect transform inclusive scan result");
    }
    if(scan2[i] != scan2[i-1] + data2[i-1] * 11) {
      throw std::runtime_error("incorrect transform exclusive scan result");
    }
  }

  std::cout << "transform scan done - all results are correct\n";
  
  // deallocate the data
  cudaFree(data1);
  cudaFree(data2);
  cudaFree(scan1);
  cudaFree(scan2);
  cudaFree(buff);

  return 0;
}


