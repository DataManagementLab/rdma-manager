/**
 * @file TestRDMAServerMultClients.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */


#pragma once

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"
#include <gtest/gtest.h>


using namespace rdma;

class TestRDMAServerMultClients : public testing::Test {

protected:
  void SetUp() override;

  std::unique_ptr<RDMAServer<ReliableRDMA>> m_rdmaServer;
  std::unique_ptr<NodeIDSequencer> m_nodeIDSequencer;
  std::unique_ptr<RDMAClient<ReliableRDMA>> m_rdmaClient_0;
  std::unique_ptr<RDMAClient<ReliableRDMA>> m_rdmaClient_1;

  string m_connection;
  NodeID m_nodeId = 0;

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
