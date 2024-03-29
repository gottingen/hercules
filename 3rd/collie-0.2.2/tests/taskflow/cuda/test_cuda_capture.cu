#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <collie/testing/doctest.h>
#include <collie/taskflow/taskflow.h>
#include <collie/taskflow/cuda/cudaflow.h>

void __global__ testKernel() {}

TEST_CASE("cudaFlowCapturer.noEventError") {
  collie::tf::cudaFlow f;
  f.capture([](collie::tf::cudaFlowCapturer& cpt) {
    cpt.on([] (cudaStream_t stream) {
      testKernel<<<256,256,0,stream>>>();
    });
    REQUIRE((cudaGetLastError() == cudaSuccess));
  });
  REQUIRE((cudaGetLastError() == cudaSuccess));
}
