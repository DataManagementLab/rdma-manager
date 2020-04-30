//
// Created by Tilo Gaulke on 15.10.19.
//

#include "RPCPerf.h"
#include "memory"

mutex rdma::RPCPerf::waitLock;
condition_variable rdma::RPCPerf::waitCv;
bool rdma::RPCPerf::signaled;

rdma::RPCPerfThread::RPCPerfThread(vector<string>& conns,
                                                     size_t size, size_t iter) {
    m_size = size;
    m_iter = iter;
    m_conns = conns;
    //m_remOffsets = new size_t[m_conns.size()];

    for (const auto& conn : m_conns) {
        NodeID  nodeId = 0;
        //ib_addr_t ibAddr;
        if (!m_client.connect(conn, nodeId)) {
            throw invalid_argument(
                    "RPCPerf connection failed");
        }
        std::cout << "Connected to server Node ID " << nodeId <<endl;
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
    //todo make many server work again
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

        //todo make sendreturn work
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
                         config.threads,config.returnMethod,config.old) {
    this->isClient(isClient);

    //check parameters
    if (isClient && config.server.length() > 0) {
        this->isRunnable(true);
    } else if (!isClient) {
        this->isRunnable(true);
    }
}

//todo remove size
rdma::RPCPerf::RPCPerf(string& conns, size_t serverPort, size_t, size_t iter, size_t threads,std::size_t returnMethod,bool old) {
    m_conns = StringHelper::split(conns);
    m_serverPort = serverPort;
    //m_size = size;
    m_iter = iter;
    m_numThreads = threads;
    m_old = old;
    m_returnMethod = returnMethod;
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
    Config::SEQUENCER_IP = "localhost";
    m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();
    size_t MAX_NUM_RPC_MSG = 4096;
    cout << "Server Ip Adress: " << rdma::Config::getIP(Config::RDMA_INTERFACE) << endl;
    std::cout << "server port " << m_serverPort << std::endl;
    std::cout << "sizeof(testpage) = "<<sizeof(testPage) << endl;
    std::cout << "old = "<< m_old << endl;
    std::cout << "returnMethod = "<< m_returnMethod << endl;
    m_dServer = new RDMAServer<ReliableRDMA>("test", m_serverPort);
    size_t srqID = 0;
    m_dServer->createSharedReceiveQueue(srqID);
    m_dServer->activateSRQ(srqID);


    if(m_old){
        the = new TestRPCHandlerThreadOld(m_dServer,srqID,MAX_NUM_RPC_MSG);
    }else{
        the = new TestRPCHandlerThread(m_dServer,srqID,MAX_NUM_RPC_MSG);
    }


    m_dServer->startServer();


    the->startHandler();

    while (m_dServer->isRunning()) {
        usleep(Config::RDMA_SLEEP_INTERVAL);
    }
}

void rdma::RPCPerf::runClient() {
    //todo make many server work again
    //make this better
    string str = m_conns[0];
    //sequencer is the same IP as the rdma server here
    Config::SEQUENCER_IP =  str.substr(0,str.find(':') );;
    //start all client threads
    for (size_t i = 0; i < m_numThreads; i++) {
        std::cout << "Started thread " << i << std::endl;
        auto* perfThread = new RPCPerfThread(m_conns,m_size, m_iter);
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
    for (auto & m_thread : m_threads) {
        m_thread->join();
    }
}

double rdma::RPCPerf::time() {
    uint128_t totalTime = 0;
    for (auto & m_thread : m_threads) {
        totalTime += m_thread->time();
    }
    return ((double) totalTime) / m_threads.size();
}


