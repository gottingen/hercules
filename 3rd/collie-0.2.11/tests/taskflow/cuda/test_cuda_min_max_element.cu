#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <collie/testing/doctest.h>
#include <collie/taskflow/taskflow.h>
#include <collie/taskflow/cuda/cudaflow.h>
#include <collie/taskflow/cuda/algorithm/find.hpp>

// ----------------------------------------------------------------------------
// cuda_min_max_element
// ----------------------------------------------------------------------------

template <typename T>
void cuda_min_max_element() {

  collie::tf::Taskflow taskflow;
  collie::tf::Executor executor;
  
  for(int n=1; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      collie::tf::cudaStream stream;
      collie::tf::cudaDefaultExecutionPolicy policy(stream);
  
      // gpu data
      auto gdata = collie::tf::cuda_malloc_shared<T>(n);
      auto min_i = collie::tf::cuda_malloc_shared<unsigned>(1);
      auto max_i = collie::tf::cuda_malloc_shared<unsigned>(1);

      // buffer
      void* buff;
      cudaMalloc(&buff, policy.min_element_bufsz<T>(n));

      for(int i=0; i<n; i++) {
        gdata[i] = rand() % 1000 - 500;
      }

      // --------------------------------------------------------------------------
      // GPU find
      // --------------------------------------------------------------------------
      collie::tf::cudaStream s;
      collie::tf::cudaDefaultExecutionPolicy p(s);

      collie::tf::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return a < b; }, buff
      );
      
      collie::tf::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return a < b; }, buff
      );
      s.synchronize();
      
      auto min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return a < b; });
      auto max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return a < b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);
      
      // change the comparator
      collie::tf::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return a > b; }, buff
      );
      
      collie::tf::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return a > b; }, buff
      );
      s.synchronize();
      
      min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return a > b; });
      max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return a > b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);

      // change the comparator
      collie::tf::cuda_min_element(
        p, gdata, gdata+n, min_i, []__device__(T a, T b) { return -a > -b; }, buff
      );
      
      collie::tf::cuda_max_element(
        p, gdata, gdata+n, max_i, []__device__(T a, T b) { return -a > -b; }, buff
      );
      s.synchronize();
      
      min_v = *std::min_element(gdata, gdata+n, [](T a, T b) { return -a > -b; });
      max_v = *std::max_element(gdata, gdata+n, [](T a, T b) { return -a > -b; });

      REQUIRE(gdata[*min_i] == min_v);
      REQUIRE(gdata[*max_i] == max_v);
      
      // change the comparator
      collie::tf::cuda_min_element(
        p, gdata, gdata+n, min_i, 
        []__device__(T a, T b) { return std::abs(a) < std::abs(b); }, 
        buff
      );
      
      collie::tf::cuda_max_element(
        p, gdata, gdata+n, max_i, 
        []__device__(T a, T b) { return std::abs(a) < std::abs(b); }, 
        buff
      );
      s.synchronize();
      
      min_v = *std::min_element(
        gdata, gdata+n, [](T a, T b) { return std::abs(a) < std::abs(b); }
      );

      max_v = *std::max_element(
        gdata, gdata+n, [](T a, T b) { return std::abs(a) < std::abs(b); }
      );

      REQUIRE(std::abs(gdata[*min_i]) == std::abs(min_v));
      REQUIRE(std::abs(gdata[*max_i]) == std::abs(max_v));
      
      // deallocate the memory
      REQUIRE(cudaFree(gdata) == cudaSuccess);
      REQUIRE(cudaFree(min_i) == cudaSuccess);
      REQUIRE(cudaFree(max_i) == cudaSuccess);
      REQUIRE(cudaFree(buff)  == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_min_max_element.int" * doctest::timeout(300)) {
  cuda_min_max_element<int>();
}

TEST_CASE("cuda_min_max_element.float" * doctest::timeout(300)) {
  cuda_min_max_element<float>();
}

