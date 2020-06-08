#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */

#ifndef SRC_TEST_NET_TestCudaMemory_H_
#define SRC_TEST_NET_TestCudaMemory_H_

#include "../../src/memory/CudaMemory.h"
#include <gtest/gtest.h>


using namespace rdma;

class TestCudaMemory : public testing::Test {
protected:

  void SetUp() override;
  
  struct testMsg {
  int id;
  char a;
  testMsg(int n, char t)
      : id(n),
        a(t)  // Create an object of type _book.
  {
  };
};

};

#endif
#endif /* CUDA support */
