//
// Created by Tilo Gaulke on 2019-06-05.
//

#include "TestRPC.h"

#include "../../src/rdma/RDMAServer.h"
#include "../../src/rdma/RDMAClient.h"
#include "../../src/RPC/RPCMemory.h"
#include "../../src/utils/Config.h"




using namespace rdma;


void TestRPC::setUp() {






    const static size_t MAX_NUM_RPC_MSG = 32;  // Number of RDMA send MSGs RPCHandler can recv



    m_rdmaServer = new RDMAServer();
    CPPUNIT_ASSERT(m_rdmaServer->startServer());


    //Create SRQ
    CPPUNIT_ASSERT(m_rdmaServer->createSRQ(m_srq_id));

    m_rdmaServer->activateSRQ(m_srq_id);

    m_connection = "127.0.0.1:" + to_string(Config::RDMA_PORT);
    m_rdmaClient_0 = new RDMAClient();
    CPPUNIT_ASSERT(m_rdmaClient_0->connect(m_connection, m_nodeId));






    the = new TestRPCHandlerThread(m_rdmaServer, m_srq_id,
                                              MAX_NUM_RPC_MSG);
    localresp = (testMsg*) m_rdmaClient_0->localAlloc(sizeof(testMsg));
    localstruct = (testMsg*) m_rdmaClient_0->localAlloc(sizeof(testMsg));

    the->startServer();
}

void TestRPC::tearDown() {
    the->stopServer();

    delete the;

    m_rdmaServer->localFree(localresp);
    m_rdmaServer->localFree(localstruct);

    if (m_rdmaServer != nullptr) {
        m_rdmaServer->stopServer();
        delete m_rdmaServer;
        m_rdmaServer = nullptr;
    }
    if (m_rdmaClient_0 != nullptr) {
        delete m_rdmaClient_0;
        m_rdmaClient_0 = nullptr;
    }

    if (m_rdmaClient_1 != nullptr) {
        delete m_rdmaClient_1;
        m_rdmaClient_1 = nullptr;
    }

}

void TestRPC::testRPC() {


    for (int i = 0; i < 100; ++i) {
        localstruct->id = i;
        localstruct->a = 'B';
        m_rdmaClient_0->receive(m_nodeId, (void*) localresp, sizeof(testMsg));
        m_rdmaClient_0->send(m_nodeId, (void*) localstruct, sizeof(testMsg), true);


        //while(!( m_rdmaClient_0->pollReceive(m_nodeId)));

        auto ret = m_rdmaClient_0->pollReceive(m_nodeId);
        if(ret){

        }

        CPPUNIT_ASSERT_EQUAL(i+1,localresp->id);
        CPPUNIT_ASSERT_EQUAL(localstruct->id+1,localresp->id );

        CPPUNIT_ASSERT_EQUAL( 'A',localresp->a);
        //is this ok?
        CPPUNIT_ASSERT_EQUAL( (char)(localstruct->a -1),localresp->a);
    }






}
