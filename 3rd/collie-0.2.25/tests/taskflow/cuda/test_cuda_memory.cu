#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <collie/testing/doctest.h>
#include <collie/taskflow/taskflow.h>
#include <collie/taskflow/cuda/cudaflow.h>
#include <collie/taskflow/cuda/algorithm/for_each.hpp>

// ----------------------------------------------------------------------------
// USM Allocator
// ----------------------------------------------------------------------------

TEST_CASE("cudaUSMAllocator" * doctest::timeout(300)) {

  collie::tf::cudaStream stream;

  std::vector<int, collie::tf::cudaUSMAllocator<int>> vec;
  std::vector<int, collie::tf::cudaUSMAllocator<int>> rhs;

  REQUIRE(vec.size() == 0);

  vec.resize(100, 10);
  REQUIRE(vec.size() == 100);
  for(auto c : vec) {
    REQUIRE(c == 10);
  }

  rhs = std::move(vec);

  REQUIRE(vec.size() == 0);
  REQUIRE(rhs.size() == 100);
  for(auto c : rhs) {
    REQUIRE(c == 10);
  }

  for(int i=0; i<65536; i++) {
    vec.push_back(-i);
  }
  for(int i=0; i<65536; i++) {
    REQUIRE(vec[i] == -i);
  }

  rhs = vec;
  
  for(int i=0; i<65536; i++) {
    REQUIRE(vec[i] == rhs[i]);
  }

  collie::tf::cudaDefaultExecutionPolicy p(stream);
  
  collie::tf::cuda_for_each(p, vec.data(), vec.data() + vec.size(), [] __device__ (int& v) {
    v = -177;
  });
  stream.synchronize();

  rhs = vec;
  for(size_t i=0; i<vec.size(); i++) {
    REQUIRE(vec[i] == -177);
    REQUIRE(rhs[i] == vec[i]);
  }

  vec.clear();
  REQUIRE(vec.size() == 0);
}

// ----------------------------------------------------------------------------
// Device Allocator
// ----------------------------------------------------------------------------

TEST_CASE("cudaDeviceAllocator" * doctest::timeout(300)) {


  size_t N = 10000;
  
  std::vector<collie::tf::NoInit<int>, collie::tf::cudaDeviceAllocator<collie::tf::NoInit<int>>> vec;
  std::vector<collie::tf::NoInit<int>, collie::tf::cudaDeviceAllocator<collie::tf::NoInit<int>>> rhs(N);

  REQUIRE(vec.size() == 0);
  REQUIRE(rhs.size() == 10000);
  
  //collie::tf::cudaStream stream;
  //collie::tf::cudaDefaultExecutionPolicy policy(stream);
  //
  //collie::tf::cuda_for_each(policy, rhs.data(), rhs.data() + N, [] __device__ (collie::tf::NoInit<int>& v) {
  //  v = -177;
  //});
  //stream.synchronize();
}












