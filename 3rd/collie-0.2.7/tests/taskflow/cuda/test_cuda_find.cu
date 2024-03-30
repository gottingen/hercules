#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <collie/testing/doctest.h>
#include <collie/taskflow/taskflow.h>
#include <collie/taskflow/cuda/cudaflow.h>
#include <collie/taskflow/cuda/algorithm/find.hpp>

// ----------------------------------------------------------------------------
// cuda_find_if
// ----------------------------------------------------------------------------

template <typename T>
void cuda_find_if() {

  collie::tf::Taskflow taskflow;
  collie::tf::Executor executor;
  
  for(int n=0; n<=1234567; n = (n<=100) ? n+1 : n*2 + 1) {

    taskflow.emplace([n](){

      collie::tf::cudaStream stream;
      collie::tf::cudaDefaultExecutionPolicy policy(stream);
  
      // gpu data
      auto gdata = collie::tf::cuda_malloc_shared<T>(n);
      auto gfind = collie::tf::cuda_malloc_shared<unsigned>(1);

      // cpu data
      auto hdata = std::vector<T>(n);

      // initialize the data
      for(int i=0; i<n; i++) {
        T k = rand()% 100;
        gdata[i] = k;
        hdata[i] = k;
      }

      // --------------------------------------------------------------------------
      // GPU find
      // --------------------------------------------------------------------------
      collie::tf::cudaStream s;
      collie::tf::cudaDefaultExecutionPolicy p(s);
      collie::tf::cuda_find_if(
        p, gdata, gdata+n, gfind, []__device__(T v) { return v == (T)50; }
      );
      s.synchronize();
      
      // --------------------------------------------------------------------------
      // CPU find
      // --------------------------------------------------------------------------
      auto hiter = std::find_if(
        hdata.begin(), hdata.end(), [=](T v) { return v == (T)50; }
      );
      
      // --------------------------------------------------------------------------
      // verify the result
      // --------------------------------------------------------------------------
      unsigned hfind = std::distance(hdata.begin(), hiter);
      REQUIRE(*gfind == hfind);

      REQUIRE(cudaFree(gdata) == cudaSuccess);
      REQUIRE(cudaFree(gfind) == cudaSuccess);
    });
  }

  executor.run(taskflow).wait();
}

TEST_CASE("cuda_find_if.int" * doctest::timeout(300)) {
  cuda_find_if<int>();
}

TEST_CASE("cuda_find_if.float" * doctest::timeout(300)) {
  cuda_find_if<float>();
}
