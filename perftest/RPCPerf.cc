//
// Created by Tilo Gaulke on 15.10.19.
//

#include "RPCPerf.h"


mutex rdma::RPCPerf::waitLock;
condition_variable rdma::RPCPerf::waitCv;
bool rdma::RPCPerf::signaled;

rdma::RPCPerfThread::RPCPerfThread(vector<string>& conns,
                                                     size_t size, size_t iter) {
    m_size = size;
    m_iter = iter;
    m_conns = conns;
    //m_remOffsets = new size_t[m_conns.size()];

    for (size_t i = 0; i < m_conns.size(); ++i) {
        NodeID  nodeId = 0;
        //ib_addr_t ibAddr;
        string conn = m_conns[i];
        if (!m_client.connect(conn, nodeId)) {
            throw invalid_argument(
                    "RPCPerf connection failed");
        }
        std::cout << "Connected to server Node ID " << nodeId; 
        m_addr.push_back(nodeId);
        //m_client.remoteAlloc(conn, m_size, m_remOffsets[i]);
    }

    localresp = (testPage*) m_client.localAlloc(sizeof(testPage));
    localsend = (testMsg*) m_client.localAlloc(sizeof(testMsg));
    localresp->id = -1;
    returnOffset = m_client.convertPointerToOffset((void*)localresp);

    //m_data = m_client.localAlloc(m_size);
    //memset(m_data, 1, m_size);
}

rdma::RPCPerfThread::~RPCPerfThread() {
    //delete m_remOffsets;
    //m_client.localFree(m_data);
    m_client.localFree(localresp);
    m_client.localFree(localsend);
    /*
    for (size_t i = 0; i < m_conns.size(); ++i) {
        string conn = m_conns[i];
        m_client.remoteFree(conn, m_remOffsets[i], m_size);
    }
     */
}

void rdma::RPCPerfThread::run() {
    unique_lock < mutex > lck(RPCPerf::waitLock);
    if (!RPCPerf::signaled) {
        m_ready = true;
        RPCPerf::waitCv.wait(lck);
    }
    lck.unlock();
    startTimer();
    for (size_t i = 0; i < m_iter; ++i) {
        // size_t connIdx = i % m_conns.size();
        // bool signaled = (i == (m_iter - 1));
        localsend->id = i;
        localsend->offset = returnOffset;
        // m_client.receive(m_addr[connIdx], (void*) localresp, sizeof(testMsg));
        m_client.send(m_addr[0], (void*) localsend, sizeof(testMsg), false);
    
#ifdef SENDReturn
        bool poll = true;
        m_client.pollReceive(m_addr[0], poll);
#else
       while(localresp->id == -1){
        //    std::cout << "Waiting for receive" << i << std::endl;
        __asm__("pause");
       }
       localresp->id = -1;
#endif



        //todo assert id in loclresp



    }
    endTimer();
}

//todo remove size
rdma::RPCPerf::RPCPerf(config_t config, bool isClient) :
        RPCPerf(config.server, config.port, config.data, config.iter,
                         config.threads) {
    this->isClient(isClient);

    //check parameters
    if (isClient && config.server.length() > 0) {
        this->isRunnable(true);
    } else if (!isClient) {
        this->isRunnable(true);
    }
}

//todo remove size
rdma::RPCPerf::RPCPerf(string& conns, size_t serverPort,
                                         size_t, size_t iter, size_t threads) {
    m_conns = StringHelper::split(conns);
    m_serverPort = serverPort;
    //m_size = size;
    m_iter = iter;
    m_numThreads = threads;
    RPCPerf::signaled = false;
}

rdma::RPCPerf::~RPCPerf() {
    if (this->isClient()) {
        for (size_t i = 0; i < m_threads.size(); i++) {
            delete m_threads[i];
        }
        m_threads.clear();
    } else {
        if (m_dServer != nullptr) {
            if(the != nullptr){
                the->stopHandler();
                delete the;
            }
            m_dServer->stopServer();
            delete m_dServer;
        }
        m_dServer = nullptr;
    }
}

void rdma::RPCPerf::runServer() {
    m_nodeIDSequencer = new NodeIDSequencer();
    size_t MAX_NUM_RPC_MSG = 4096;
    std::cout << "server port " << m_serverPort << std::endl;
    m_dServer = new RDMAServer<ReliableRDMA>("test", m_serverPort);
    size_t srqID = 0;
    m_dServer->createSharedReceiveQueue(srqID);
    m_dServer->activateSRQ(srqID);

    the = new TestRPCHandlerThread(m_dServer,srqID,MAX_NUM_RPC_MSG);
    the->startHandler();

    while (m_dServer->isRunning()) {
        usleep(Config::RDMA_SLEEP_INTERVAL);
    }
}

void rdma::RPCPerf::runClient() {
    //start all client threads
    for (size_t i = 0; i < m_numThreads; i++) {
        std::cout << "Started thread " << i << std::endl;
        RPCPerfThread* perfThread = new RPCPerfThread(m_conns,
                                                                        m_size, m_iter);
        perfThread->start();
        if (!perfThread->ready()) {
            usleep(Config::RDMA_SLEEP_INTERVAL);
        }
        m_threads.push_back(perfThread);
    }

    //wait for user input
    // waitForUser();

    //send signal to run benchmark
    RPCPerf::signaled = false;
    unique_lock < mutex > lck(RPCPerf::waitLock);
    RPCPerf::waitCv.notify_all();
    RPCPerf::signaled = true;
    lck.unlock();
    for (size_t i = 0; i < m_threads.size(); i++) {
        m_threads[i]->join();
    }
}

double rdma::RPCPerf::time() {
    uint128_t totalTime = 0;
    for (size_t i = 0; i < m_threads.size(); i++) {
        totalTime += m_threads[i]->time();
    }
    return ((double) totalTime) / m_threads.size();
}


