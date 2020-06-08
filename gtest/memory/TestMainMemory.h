#ifndef SRC_TEST_NET_TestMainMemory_H_
#define SRC_TEST_NET_TestMainMemory_H_

#include "../../src/memory/MainMemory.h"
#include <gtest/gtest.h>


using namespace rdma;

class TestMainMemory : public testing::Test {
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
