
#include "TestRDMAServer.h"

#include <numa.h>
#include <numaif.h>

void TestRDMAServer::SetUp() {
  Config::RDMA_MEMSIZE = 1024 * 1024;
  Config::SEQUENCER_IP = rdma::Config::getIP(rdma::Config::RDMA_INTERFACE);

  m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();

  m_rdmaServer = std::make_unique<RDMAServer<ReliableRDMA>>();
  m_rdmaServer->startServer();
  m_connection = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT);
  m_rdmaClient = std::make_unique<RDMAClient<ReliableRDMA>>();

  ASSERT_TRUE(m_rdmaClient->connect(m_connection, m_nodeId));
}


TEST_F(TestRDMAServer, testWrite) {
  size_t remoteOffset = 0;
  size_t memSize = sizeof(int) * 2;

  //allocate local array
  int* localValues = (int*) m_rdmaClient->localAlloc(memSize);
  ASSERT_TRUE(localValues!=nullptr);

  //remote allocate array
  ASSERT_TRUE(
      m_rdmaClient->remoteAlloc(m_connection, memSize, remoteOffset));

  //write to remote machine
  localValues[0] = 1;
  localValues[1] = 2;
  m_rdmaClient->write(m_nodeId, remoteOffset, localValues, memSize, true);

  //read from remote machine
  int* remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);
  ASSERT_EQ(remoteVals[0], localValues[0]);
  ASSERT_EQ(remoteVals[1], localValues[1]);

  //remote free
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, remoteOffset));
}

TEST_F(TestRDMAServer, testWriteImm) {
    size_t remoteOffset = 0;
    size_t memSize = sizeof(int) * 2;

    //allocate local array
    int* localValues = (int*) m_rdmaClient->localAlloc(memSize);
    ASSERT_TRUE(localValues!=nullptr);

    //remote allocate array
    ASSERT_TRUE(
            m_rdmaClient->remoteAlloc(m_connection, memSize, remoteOffset));

    //write to remote machine
    localValues[0] = 1;
    localValues[1] = 2;
    uint32_t immData = 123456789;

    testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
    ASSERT_TRUE(remotestruct!=nullptr);


    //writeImm consumes RR
    //remotestruct only needed for valid mem adress
    m_rdmaServer->receive(m_rdmaClient->getOwnNodeID(), (void*)remotestruct, 0);

    m_rdmaClient->writeImm(m_nodeId, remoteOffset, localValues, memSize,immData, false);

    //poll
    //maybe consider htonl and ntohl because immediate value is in network order
    uint32_t immRvc;
    int ne = m_rdmaServer->pollReceive(m_rdmaClient->getOwnNodeID(),true,&immRvc);
    ASSERT_TRUE(ne == 1);
    ASSERT_EQ(immData,immRvc);

    //read from remote machine
    int* remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);
    ASSERT_EQ(remoteVals[0], localValues[0]);
    ASSERT_EQ(remoteVals[1], localValues[1]);



    immData = 96347165;
    //writeImm consumes RR
    //remotestruct only needed for valid mem adress
    m_rdmaServer->receive(m_rdmaClient->getOwnNodeID(), (void*)remotestruct, 0);

    m_rdmaClient->writeImm(m_nodeId, remoteOffset, localValues, memSize,immData, false);

    //poll
    //maybe consider htonl and ntohl because immediate value is in network order

    ne = m_rdmaServer->pollReceive(m_rdmaClient->getOwnNodeID(),true,&immRvc);
    ASSERT_TRUE(ne == 1);
    ASSERT_EQ(immData,immRvc);

    //read from remote machine
    remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);
    ASSERT_EQ(remoteVals[0], localValues[0]);
    ASSERT_EQ(remoteVals[1], localValues[1]);

    //remote free
    ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, remoteOffset));
}

TEST_F(TestRDMAServer, restRemoteAlloc) {
  size_t memSize = 10;
  size_t offset = 0;  // arbitrary value greater memSize
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize / 2, offset));
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize / 2, offset));
}

TEST_F(TestRDMAServer, testRemoteFree) {
  size_t memSize = 15;
  size_t offset = 0;  // arbitrary value greater memSize
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize, offset));
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, offset));
  ASSERT_TRUE(m_rdmaClient->remoteAlloc(m_connection, memSize, offset));
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, offset));
}

TEST_F(TestRDMAServer, testSendRecieve) {
  testMsg* localstruct = (testMsg*) m_rdmaClient->localAlloc(sizeof(testMsg));
  ASSERT_TRUE(localstruct!=nullptr);
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
  ASSERT_TRUE(remotestruct!=nullptr);


  ASSERT_NO_THROW(
      m_rdmaServer->receive(m_rdmaClient->getOwnNodeID(), (void* )remotestruct, sizeof(testMsg)));
  ASSERT_NO_THROW(
      m_rdmaClient->send(m_nodeId,(void*)localstruct,sizeof(testMsg),true));
  bool poll = true;
  ASSERT_NO_THROW(m_rdmaServer->pollReceive(m_rdmaClient->getOwnNodeID(), poll));

  ASSERT_EQ(localstruct->id, remotestruct->id);
  ASSERT_EQ(localstruct->a, remotestruct->a);
}

TEST_F(TestRDMAServer, testAtomics) {
  size_t remoteOffset = 0;
  size_t memSize = sizeof(int64_t);

  //allocate local array
  int64_t* localValues = (int64_t*) m_rdmaClient->localAlloc(memSize);
  ASSERT_TRUE(localValues!=nullptr);

  //remote allocate array
  ASSERT_TRUE(
      m_rdmaClient->remoteAlloc(m_connection, memSize, remoteOffset));

  //write to remote machine
  localValues[0] = 1;
  ASSERT_NO_THROW(m_rdmaClient->fetchAndAdd(m_nodeId,remoteOffset,localValues, sizeof(uint64_t), true));
  ASSERT_NO_THROW(m_rdmaClient->fetchAndAdd(m_nodeId,remoteOffset,localValues, sizeof(uint64_t), true));


  int* remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);

  ASSERT_EQ(remoteVals[0], 2);
  
  // Compare and swap to zero
  ASSERT_NO_THROW(m_rdmaClient->compareAndSwap(m_nodeId,remoteOffset,localValues,2,0, sizeof(uint64_t), true));
  ASSERT_EQ(remoteVals[0], 0);

  ASSERT_NO_THROW(m_rdmaClient->fetchAndAdd(m_nodeId,remoteOffset,localValues,10,sizeof(uint64_t), true));
  ASSERT_EQ(remoteVals[0], 10);

  //remote free
  ASSERT_TRUE(m_rdmaClient->remoteFree(m_connection, memSize, remoteOffset));
} 


TEST_F(TestRDMAServer, serverToServerCommunication) {

  auto m_rdmaServer2 = std::make_unique<RDMAServer<ReliableRDMA>>("RDMAServer2", Config::RDMA_PORT +1);
  NodeID retServerNodeID;
  string m_connection2 = Config::getIP(Config::RDMA_INTERFACE) + ":" + to_string(Config::RDMA_PORT + 1);
  ASSERT_TRUE(m_rdmaServer2->connect(m_connection, retServerNodeID));

  // WRITE
  size_t remoteOffset = 0;
  size_t memSize = sizeof(int) * 2;

  //allocate local array
  int* localValues = (int*) m_rdmaServer2->localAlloc(memSize);
  ASSERT_TRUE(localValues!=nullptr);

  //remote allocate array
  ASSERT_TRUE(
      m_rdmaServer2->remoteAlloc(m_connection, memSize, remoteOffset));

  //write to remote machine
  localValues[0] = 1;
  localValues[1] = 2;
  m_rdmaServer2->write(m_nodeId, remoteOffset, localValues, memSize, true);

  //read from remote machine
  int* remoteVals = (int*) m_rdmaServer->getBuffer(remoteOffset);
  ASSERT_EQ(remoteVals[0], localValues[0]);
  ASSERT_EQ(remoteVals[1], localValues[1]);


  // SEND AND RECEIVE

  testMsg* localstruct = (testMsg*) m_rdmaServer2->localAlloc(sizeof(testMsg));
  ASSERT_TRUE(localstruct!=nullptr);
  localstruct->a = 'a';
  localstruct->id = 1;
  testMsg* remotestruct = (testMsg*) m_rdmaServer->localAlloc(sizeof(testMsg));
  ASSERT_TRUE(remotestruct!=nullptr);


  ASSERT_NO_THROW(
      m_rdmaServer->receive(m_rdmaServer2->getOwnNodeID(), (void* )remotestruct, sizeof(testMsg)));
  ASSERT_NO_THROW(
      m_rdmaServer2->send(m_nodeId,(void*)localstruct,sizeof(testMsg),true));
  bool poll = true;
  ASSERT_NO_THROW(m_rdmaServer->pollReceive(m_rdmaServer2->getOwnNodeID(), poll));

  ASSERT_EQ(localstruct->id, remotestruct->id);
  ASSERT_EQ(localstruct->a, remotestruct->a);


}


TEST_F(TestRDMAServer, testNumaRegion) {
  
  Config::RDMA_NUMAREGION = numa_max_node();

  std::cout << "Allocating on NUMA node: " << Config::RDMA_NUMAREGION << std::endl;
  auto server = new RDMAServer<ReliableRDMA>("server", Config::RDMA_PORT+1, 1024);

  //allocate local array
  int* buffer = (int*) server->getBuffer(0);

  int numa_node = -1;
    if (get_mempolicy(&numa_node, NULL, 0, buffer, MPOL_F_NODE | MPOL_F_ADDR) < 0)
        std::cout << "WARNING: get_mempolicy failed" << std::endl;

  ASSERT_EQ(numa_node, Config::RDMA_NUMAREGION);
  delete server;
}