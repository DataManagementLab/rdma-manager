//
// Created by Tilo Gaulke on 2019-06-05.
//

#ifndef RDMA_TESTRPC_H
#define RDMA_TESTRPC_H



#include "../../src/rdma/RDMAServer.h"
#include "../../src/RPC/RPCHandlerThread.h"
#include "../../src/rdma/RDMAClient.h"

using namespace rdma;




struct testMsg {
    int id;
    char a;

    testMsg(int n, char t)
            : id(n),
              a(t)  // Create an object of type _book.
    {
    };
};

const static size_t QUEUE_MSG_SIZE = sizeof(testMsg);


class TestRPCHandlerThread: public RPCHandlerThread<testMsg> {
public:
    TestRPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                       size_t maxNumberMsgs,char* rpcbuffer):RPCHandlerThread<testMsg>
                                                 (rdmaServer,  srqID,
                                                  maxNumberMsgs,rpcbuffer)
    { }

    TestRPCHandlerThread(RDMAServer *rdmaServer, size_t srqID,
                         size_t maxNumberMsgs):RPCHandlerThread<testMsg>
                                                                       (rdmaServer, srqID,
                                                                        maxNumberMsgs)
    { }

    TestRPCHandlerThread(RDMAServer *rdmaServer,
                         size_t maxNumberMsgs):RPCHandlerThread<testMsg>
                                                       (rdmaServer,
                                                        maxNumberMsgs)
    { }



    void handleRDMARPC(testMsg* msg, ib_addr_t &returnAdd){

        m_intermediateRspBuffer->id= msg->id+1;
        m_intermediateRspBuffer->a = msg->a - 1;

        m_rdmaServer->send(returnAdd, (void *)m_intermediateRspBuffer,
                            sizeof(testMsg), true);

    };

};

class TestRPC : public CppUnit::TestFixture {
DPI_UNIT_TEST_SUITE (TestRPC);
        DPI_UNIT_TEST_RC(testRPC);DPI_UNIT_TEST_UD (testSendRecieve);DPI_UNIT_TEST_SUITE_END ()
    ;

public:
    void setUp();
    void tearDown();

    void testRPC();

private:

    TestRPCHandlerThread *the;

    RDMAServer* m_rdmaServer;
    RDMAClient* m_rdmaClient_0;
    RDMAClient* m_rdmaClient_1;
    RDMAClient* m_rdmaClient_2;
    RDMAClient* m_rdmaClient_3;

    NodeID m_nodeId = 0;

    string m_connection;

    char* rpc_buffer;

    size_t m_srq_id;

    testMsg* localresp;
    testMsg* localstruct;



};



#endif //RDMA_TESTRPC_H
