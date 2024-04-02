#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <collie/testing/doctest.h>
#include <collie/taskflow/taskflow.h>
#include <collie/taskflow/cuda/cudaflow.h>
#include <collie/taskflow/cuda/algorithm/for_each.hpp>

constexpr float eps = 0.0001f;

template <typename T>
void run_and_wait(T& cf) {
  collie::tf::cudaStream stream;
  cf.run(stream);
  stream.synchronize();
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

template <typename T>
void cuda_for_each_index() {

  collie::tf::Taskflow taskflow;
  collie::tf::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){
      collie::tf::cudaStream stream;
      collie::tf::cudaDefaultExecutionPolicy policy(stream);

      auto g_data = collie::tf::cuda_malloc_shared<T>(n);
      for(int i=0; i<n; i++) {
        g_data[i] = 0;
      }

      collie::tf::cuda_for_each_index(policy,
        0, n, 1, [g_data] __device__ (int i) { g_data[i] = 12222; }
      );

      stream.synchronize();

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(g_data[i] - (T)12222) < eps);
      }

      collie::tf::cuda_free(g_data);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_for_each_index.int" * doctest::timeout(300)) {
  cuda_for_each_index<int>();
}

TEST_CASE("cuda_for_each_index.float" * doctest::timeout(300)) {
  cuda_for_each_index<float>();
}

TEST_CASE("cuda_for_each_index.double" * doctest::timeout(300)) {
  cuda_for_each_index<double>();
}

// ----------------------------------------------------------------------------
// for_each_index
// ----------------------------------------------------------------------------

template <typename T, typename F>
void cudaflow_for_each_index() {
    
  collie::tf::Taskflow taskflow;
  collie::tf::Executor executor;

  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {
    
    taskflow.emplace([n](){

      auto cpu = static_cast<T*>(std::calloc(n, sizeof(T)));
      
      T* gpu = nullptr;
      REQUIRE(cudaMalloc(&gpu, n*sizeof(T)) == cudaSuccess);

      F cf;
      auto d2h = cf.copy(cpu, gpu, n);
      auto h2d = cf.copy(gpu, cpu, n);
      auto kernel = cf.for_each_index(
        0, n, 1, [gpu] __device__ (int i) { gpu[i] = 65536; }
      );
      h2d.precede(kernel);
      d2h.succeed(kernel);

      run_and_wait(cf);

      for(int i=0; i<n; i++) {
        REQUIRE(std::fabs(cpu[i] - (T)65536) < eps);
      }
      
      // update
      cf.for_each_index(kernel,
        0, n, 1, [gpu] __device__ (int i) { gpu[i] = (T)100; }
      );

      run_and_wait(cf);
      
      for(int j=0; j<n; j++) {
        REQUIRE(std::fabs(cpu[j] - (T)100) < eps);
      }

      std::free(cpu);
      REQUIRE(cudaFree(gpu) == cudaSuccess); 
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cudaFlow.for_each_index.int" * doctest::timeout(300)) {
  cudaflow_for_each_index<int, collie::tf::cudaFlow>();
}

TEST_CASE("cudaFlow.for_each_index.float" * doctest::timeout(300)) {
  cudaflow_for_each_index<float, collie::tf::cudaFlow>();
}

TEST_CASE("cudaFlow.for_each_index.double" * doctest::timeout(300)) {
  cudaflow_for_each_index<double, collie::tf::cudaFlow>();
}

TEST_CASE("cudaFlowCapturer.for_each_index.int" * doctest::timeout(300)) {
  cudaflow_for_each_index<int, collie::tf::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.for_each_index.float" * doctest::timeout(300)) {
  cudaflow_for_each_index<float, collie::tf::cudaFlowCapturer>();
}

TEST_CASE("cudaFlowCapturer.for_each_index.double" * doctest::timeout(300)) {
  cudaflow_for_each_index<double, collie::tf::cudaFlowCapturer>();
}

