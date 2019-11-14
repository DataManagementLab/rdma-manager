/**
 * @file TestRDMAServerMCast.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestRDMAServerMCast_H_
#define SRC_TEST_NET_TestRDMAServerMCast_H_

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"
#include <gtest/gtest.h>

using namespace rdma;

class TestRDMAServerMCast : public testing::Test {

 protected:
  void SetUp() override;
  void TearDown() override;

  std::unique_ptr<RDMAServer<UnreliableRDMA>> m_rdmaServer;
  std::unique_ptr<RDMAClient<UnreliableRDMA>> m_rdmaClient;
  std::unique_ptr<NodeIDSequencer> m_nodeIDSequencer;
  string m_mCastAddr;
  NodeID m_addrClient;
  NodeID m_addrServer;

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
