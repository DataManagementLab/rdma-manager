/**
 * @file TestRDMAServerSRQ.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestRDMAServerSRQ_H_
#define SRC_TEST_NET_TestRDMAServerSRQ_H_

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"
#include <gtest/gtest.h>


using namespace rdma;

class TestRDMAServerSRQ : public testing::Test {
protected:
  void SetUp() override;

  std::unique_ptr<RDMAServer<ReliableRDMA>> m_rdmaServer;
  std::unique_ptr<NodeIDSequencer> m_nodeIDSequencer;
  std::unique_ptr<RDMAClient<ReliableRDMA>> m_rdmaClient_0;
  std::unique_ptr<RDMAClient<ReliableRDMA>> m_rdmaClient_1;

  NodeID m_nodeId = 0;

  string m_connection;
  size_t m_srq_id;
  
  struct testMsg {
  int id;
  char a;
  testMsg() = default;
  testMsg(int n, char t)
      : id(n),
        a(t)  // Create an object of type _book.
  {
  };
};
};

#endif /* SRC_TEST_NET_TESTDATANODE_H_ */
