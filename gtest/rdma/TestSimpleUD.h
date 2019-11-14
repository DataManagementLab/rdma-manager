/**
 * @file TestSimpleUD.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_NET_TestSimpleUD_H_
#define SRC_TEST_NET_TestSimpleUD_H_

#include "../../src/utils/Config.h"
#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"
#include <gtest/gtest.h>


using namespace rdma;

class TestSimpleUD : public testing::Test
{
 protected:
    void SetUp();

    std::unique_ptr<NodeIDSequencer> m_nodeIDSequencer;

    std::unique_ptr<RDMAServer<UnreliableRDMA>> m_rdmaServer1;
    std::unique_ptr<RDMAServer<UnreliableRDMA>> m_rdmaServer2;

    std::unique_ptr<RDMAClient<UnreliableRDMA>> m_rdmaClient1;
    std::unique_ptr<RDMAClient<UnreliableRDMA>> m_rdmaClient2;

    string m_connection1;
    string m_connection2;

    NodeID m_s1_nodeId = 0;
    NodeID m_s2_nodeId = 1;

    struct testMsg
    {
        int id;
        char a;
        testMsg(int n, char t)
            : id(n),
              a(t) // Create an object of type _book.
              {};
    };
};

#endif
