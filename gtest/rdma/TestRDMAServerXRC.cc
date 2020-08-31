
#include "TestRDMAServerXRC.h"

void TestRDMAServerXRC::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  m_nodeId = 0;
  Config::SEQUENCER_IP = rdma::Config::getIP(rdma::Config::RDMA_INTERFACE);
  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();
  // ASSERT_TRUE(m_nodeIDSequencer->startServer());
  m_rdmaServer = std::make_unique<RDMAServer<ExReliableRDMA>>();

  // creating XRC and hack it
  //m_rdmaServer->createSharedReceiveQueue(m_srq_id);
  //m_rdmaServer->activateSRQ(m_srq_id);

  // for testing purposes use ID 0 as SRQ for now
  m_srq_id = 0;


  ASSERT_TRUE(m_rdmaServer->startServer());
  m_connection = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT);
  m_rdmaClient_0 = std::make_unique<RDMAClient<ExReliableRDMA>>();
  ASSERT_TRUE(m_rdmaClient_0->connect(m_connection, m_nodeId));
  m_rdmaClient_1 = std::make_unique<RDMAClient<ExReliableRDMA>>();
  ASSERT_TRUE(m_rdmaClient_1->connect(m_connection, m_nodeId));

}


//can fail sometimes because  ordering is not always preserved
TEST_F(TestRDMAServerXRC, testSendReceiveOld) {

  Logging::debug("TestRDMAServerXRC started", __LINE__, __FILE__);

  testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
      sizeof(testMsg));
  localstruct1->a = 'a';
  localstruct1->id = 1;

  testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
      sizeof(testMsg));
  localstruct2->a = 'b';
  localstruct2->id = 2;


  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
  testMsg* remotestruct2 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

  ASSERT_TRUE(remotestruct != nullptr);
  ASSERT_TRUE(remotestruct2 != nullptr);

  ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, (void* ) remotestruct, sizeof(testMsg)));
  ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, (void* ) remotestruct2, sizeof(testMsg)));
  

  ASSERT_NO_THROW(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), false));
  ASSERT_NO_THROW(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), false));

  NodeID retNodeID = 0;
  bool poll= true;
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID, poll));


  ASSERT_EQ(localstruct1->id, remotestruct->id);
  ASSERT_EQ(localstruct1->a, remotestruct->a);


  ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID, poll));


  ASSERT_EQ(localstruct2->id, remotestruct2->id);
  ASSERT_EQ(localstruct2->a, remotestruct2->a);
}

TEST_F(TestRDMAServerXRC, testWriteImmReceiveOld) {

    Logging::debug("TestRDMAServerXRC started", __LINE__, __FILE__);

    testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
            sizeof(testMsg));
    localstruct1->a = 'a';
    localstruct1->id = 1;

    testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
            sizeof(testMsg));
    localstruct2->a = 'b';
    localstruct2->id = 2;


    testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
    testMsg* remotestruct2 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

    ASSERT_TRUE(remotestruct != nullptr);
    ASSERT_TRUE(remotestruct2 != nullptr);

    //receive does not need mem address
    ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, nullptr, 0));
    ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, nullptr, 0));

    uint32_t imm1 = 12345678;
    uint32_t imm2 = 87654321;


    ASSERT_NO_THROW( m_rdmaClient_0->writeImm(m_nodeId,m_rdmaServer->convertPointerToOffset((void*)remotestruct),(void*)localstruct1, sizeof(testMsg),imm1,false));
    ASSERT_NO_THROW( m_rdmaClient_1->writeImm(m_nodeId,m_rdmaServer->convertPointerToOffset((void*)remotestruct2),(void*)localstruct2, sizeof(testMsg),imm2,false));

    NodeID retNodeID = 0;

    uint32_t imm1Re =0;
    uint32_t imm2Re =0;
    atomic<bool> poll = true;
    ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID,&imm1Re, poll));

    //if m_rdmaClient_0 was not first swap input
    if(retNodeID !=m_rdmaClient_0->getOwnNodeID()){
      std::swap(imm1,imm2);
    }

    ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID,&imm2Re, poll));

    ASSERT_EQ(imm1, imm1Re);
    ASSERT_EQ(imm2, imm2Re);


    ASSERT_EQ(localstruct1->id, remotestruct->id);
    ASSERT_EQ(localstruct1->a, remotestruct->a);
    ASSERT_EQ(localstruct2->id, remotestruct2->id);
    ASSERT_EQ(localstruct2->a, remotestruct2->a);


}


TEST_F(TestRDMAServerXRC, testSendReceive) {

    Logging::debug("TestRDMAServerXRC started", __LINE__, __FILE__);

    testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
            sizeof(testMsg));
    localstruct1->a = 'a';
    localstruct1->id = 1;

    testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
            sizeof(testMsg));
    localstruct2->a = 'b';
    localstruct2->id = 2;


    testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
    testMsg* remotestruct2 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

    ASSERT_TRUE(remotestruct != nullptr);
    ASSERT_TRUE(remotestruct2 != nullptr);

    ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id,0, (void* ) remotestruct, sizeof(testMsg)));
    ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id,1, (void* ) remotestruct2, sizeof(testMsg)));


    ASSERT_NO_THROW(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), false));
    ASSERT_NO_THROW(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), false));

    NodeID retNodeID = 0;
    atomic<bool> poll= true;
    std::size_t memIndex0 = 101010;
    ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID,memIndex0, poll));


    ASSERT_TRUE(memIndex0 ==0 || memIndex0==1);
    if(memIndex0 == 0){
        ASSERT_EQ(localstruct1->id, remotestruct->id);
        ASSERT_EQ(localstruct1->a, remotestruct->a);

    }else if (memIndex0 == 1){
        ASSERT_EQ(localstruct2->id, remotestruct2->id);
        ASSERT_EQ(localstruct2->a, remotestruct2->a);
    }else{
        FAIL();
    }


    std::size_t memIndex1 = 555555;
    ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID,memIndex1, poll));

    ASSERT_TRUE(memIndex1 == 0 || memIndex1 == 1);
    if(memIndex1 == 0){
        ASSERT_EQ(localstruct1->id, remotestruct->id);
        ASSERT_EQ(localstruct1->a, remotestruct->a);

    }else if (memIndex1 == 1){
        ASSERT_EQ(localstruct2->id, remotestruct2->id);
        ASSERT_EQ(localstruct2->a, remotestruct2->a);
    }else{
        FAIL();
    }


}

TEST_F(TestRDMAServerXRC, testWriteImmReceive) {

    Logging::debug("TestRDMAServerXRC started", __LINE__, __FILE__);

    testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
            sizeof(testMsg));
    localstruct1->a = 'a';
    localstruct1->id = 1;

    testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
            sizeof(testMsg));
    localstruct2->a = 'b';
    localstruct2->id = 2;


    testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
    testMsg* remotestruct2 = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));

    ASSERT_TRUE(remotestruct != nullptr);
    ASSERT_TRUE(remotestruct2 != nullptr);

    //receive does not need mem address
    ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, 0,nullptr, 0));
    ASSERT_NO_THROW(m_rdmaServer->receiveSRQ(m_srq_id, 1,nullptr, 0));

    uint32_t imm1 = 12345678;
    uint32_t imm2 = 87654321;


    ASSERT_NO_THROW( m_rdmaClient_0->writeImm(m_nodeId,m_rdmaServer->convertPointerToOffset((void*)remotestruct),(void*)localstruct1, sizeof(testMsg),imm1,false));
    ASSERT_NO_THROW( m_rdmaClient_1->writeImm(m_nodeId,m_rdmaServer->convertPointerToOffset((void*)remotestruct2),(void*)localstruct2, sizeof(testMsg),imm2,false));

    NodeID retNodeID = 0;

    uint32_t imm1Re =0;
    uint32_t imm2Re =0;
    atomic<bool> poll = true;
    std::size_t memIndex = 454545;
    ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id,retNodeID, memIndex,&imm1Re, poll));

    ASSERT_TRUE(memIndex == 0 || memIndex == 1);
    if(memIndex == 0){
        ASSERT_EQ(localstruct1->id, remotestruct->id);
        ASSERT_EQ(localstruct1->a, remotestruct->a);

    }else if(memIndex == 1){
        ASSERT_EQ(localstruct2->id, remotestruct2->id);
        ASSERT_EQ(localstruct2->a, remotestruct2->a);

    }else{
        cout << memIndex << endl;
        FAIL();
    }

    ASSERT_TRUE(retNodeID == m_rdmaClient_0->getOwnNodeID() || retNodeID == m_rdmaClient_1->getOwnNodeID());
    if(retNodeID == m_rdmaClient_0->getOwnNodeID()){
        ASSERT_EQ(imm1, imm1Re);
    }else if(retNodeID == m_rdmaClient_1->getOwnNodeID()){
        ASSERT_EQ(imm2, imm1Re);
    }else{
        FAIL();
    }

    memIndex = 746388448;
    ASSERT_NO_THROW(m_rdmaServer->pollReceiveSRQ(m_srq_id, retNodeID,memIndex,&imm2Re, poll));

    ASSERT_TRUE(memIndex == 0 || memIndex == 1);

    if(memIndex == 0){
        ASSERT_EQ(localstruct1->id, remotestruct->id);
        ASSERT_EQ(localstruct1->a, remotestruct->a);
    }else if(memIndex == 1){
        ASSERT_EQ(localstruct2->id, remotestruct2->id);
        ASSERT_EQ(localstruct2->a, remotestruct2->a);
    }else{
        FAIL();
    }

    ASSERT_TRUE(retNodeID == m_rdmaClient_0->getOwnNodeID()|| retNodeID == m_rdmaClient_1->getOwnNodeID());
    if(retNodeID == m_rdmaClient_0->getOwnNodeID()){
        ASSERT_EQ(imm1, imm2Re);
    }else if(retNodeID == m_rdmaClient_1->getOwnNodeID()){
        ASSERT_EQ(imm2, imm2Re);
    }else{
        FAIL();
    }







}


TEST_F(TestRDMAServerXRC, DISABLED_testPollReceiveBatch) {

  Logging::debug("TestRDMAServerXRC started", __LINE__, __FILE__);

  testMsg* localstruct1 = (testMsg*) m_rdmaClient_0->localAlloc(
      sizeof(testMsg));
  localstruct1->a = 'a';
  localstruct1->id = 1;

  testMsg* localstruct2 = (testMsg*) m_rdmaClient_1->localAlloc(
      sizeof(testMsg));
  localstruct2->a = 'a';
  localstruct2->id = 1;

  const size_t receives = 4;

  testMsg *remotestructs[receives];
  for (size_t i = 0; i < receives; i++)
  {
    remotestructs[i] = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
    ASSERT_TRUE(remotestructs[i] != nullptr);
  }

  for (size_t i = 0; i < receives; i++)
  {
    ASSERT_NO_THROW(m_rdmaServer->receive(m_srq_id, (void* ) remotestructs[i], sizeof(testMsg)));
  }

  size_t num_received_1 = 0;
  size_t num_received_2 = 0;

  ASSERT_NO_THROW(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), true));
  ASSERT_NO_THROW(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), true));
  bool poll = true;
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveBatch(m_srq_id, num_received_1, poll));

  ASSERT_NO_THROW(m_rdmaClient_0->send(m_nodeId, (void*) localstruct1, sizeof(testMsg), true));
  ASSERT_NO_THROW(m_rdmaClient_1->send(m_nodeId, (void*) localstruct2, sizeof(testMsg), true));
  ASSERT_NO_THROW(m_rdmaServer->pollReceiveBatch(m_srq_id, num_received_2, poll));

  ASSERT_EQ(num_received_1 + num_received_2, receives);

  for (size_t i = 0; i < receives; i++) {
    ASSERT_EQ(localstruct1->id, remotestructs[i]->id);
    ASSERT_EQ(localstruct1->a, remotestructs[i]->a);
  }
}
