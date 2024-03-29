// This program demonstrates how to performs a parallel transform
// using cudaFlow.

#include <collie/taskflow/cuda/cudaflow.h>
#include <collie/taskflow/cuda/algorithm/transform.h>

int main(int argc, char* argv[]) {

  if(argc != 2) {
    std::cerr << "usage: ./cuda_transform num_items\n";
    std::exit(EXIT_FAILURE);
  }

  size_t N = std::atoi(argv[1]);

  auto input  = collie::tf::cuda_malloc_shared<int>(N);
  auto output = collie::tf::cuda_malloc_shared<int>(N);
  
  // initialize the data
  for(size_t i=0; i<N; i++) {
    input [i] = -1;
    output[i] = 1;
  }
  
  // perform parallel transform
  collie::tf::cudaFlow cudaflow;
  collie::tf::cudaStream stream;
  
  // output[i] = input[i] + 11
  cudaflow.transform(
    input, input + N, output, [] __device__ (int a) { return a + 11; }
  );

  cudaflow.run(stream);
  stream.synchronize();

  // inspect the result
  for(size_t i=0; i<N; i++) {
    if(output[i] != 10) {
      throw std::runtime_error("incorrect result");
    }
  }

  std::cout << "correct result\n";

  return 0;
}
