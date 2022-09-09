//
// Created by Tilo Gaulke on 15.10.19.
//

#include "RPCPerf.h"
#include "memory"

mutex rdma::RPCPerf::waitLock;
condition_variable rdma::RPCPerf::waitCv;
bool rdma::RPCPerf::signaled;


rdma::RPCPerfThread::RPCPerfThread(vector<string>& conns,
                                   size_t size, size_t iter, std::size_t returnMethod,bool signaled) {
    m_size = size;
    m_iter = iter;
    m_conns = conns;
    m_returnMethod = returnMethod;
    m_signaled = signaled;
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
    }

    localresp = (char*) m_client.localAlloc(m_size);
    localsend = (testMsg*) m_client.localAlloc(sizeof(testMsg));



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
    switch (m_returnMethod) {
        case 0:
            poll();
            break;
        case 1:
            imm();
            break;
        case 2:
            send();
            break;
        default:
            poll();

            //todo assert id in loclresp

    }
    endTimer();
}

void rdma::RPCPerfThread::poll()  {
    for (size_t i = 0; i < this->m_iter; ++i) {
        // size_t connIdx = i % m_conns.size();
        // bool signaled = (i == (m_iter - 1));
        this->localsend->id = i;
        this->localsend->offset = this->returnOffset;
        auto ft = (rdma::testFooter*) (this->localresp + this->m_size - sizeof(rdma::testFooter));
        ft->ret = -1;
        this->m_client.send(this->m_addr[0], (void*) this->localsend, sizeof(rdma::testMsg), m_signaled);

        while(ft->ret == -1){
            //std::cout << "Waiting for receive" << i << std::endl;
#ifdef __x86_64__
            __asm__("pause");
#else
            usleep(0);
#endif
        }
        //ft->ret = -1;

}
}

void rdma::RPCPerfThread::imm() {
    for (size_t i = 0; i < this->m_iter; ++i) {
        // size_t connIdx = i % m_conns.size();
        // bool signaled = (i == (m_iter - 1));
        this->localsend->id = i;
        this->localsend->offset = this->returnOffset;

        m_client.receive(m_addr[0], (void*) localresp, 0);

        this->m_client.send(this->m_addr[0], (void*) this->localsend, sizeof(rdma::testMsg), m_signaled);


        bool poll = true;
        uint32_t immVal;
        m_client.pollReceive(m_addr[0], poll,&immVal);
        //check immVal



    }
}

void rdma::RPCPerfThread::send() {
    for (size_t i = 0; i < this->m_iter; ++i) {
        // size_t connIdx = i % m_conns.size();
        // bool signaled = (i == (m_iter - 1));
        this->localsend->id = i;
        this->localsend->offset = this->returnOffset;

        m_client.receive(m_addr[0], (void*) localresp, m_size);

        this->m_client.send(this->m_addr[0], (void*) this->localsend, sizeof(rdma::testMsg), m_signaled);

        bool poll = true;
        m_client.pollReceive(m_addr[0], poll);
        //check immVal

    }
}

rdma::RPCPerf::RPCPerf(config_t config, bool isClient) :
        RPCPerf(config.server, config.port, config.data, config.iter,
                         config.threads,config.returnMethod,config.old,config.signaled) {
    this->isClient(isClient);

    //check parameters
    if (isClient && config.server.length() > 0) {
        this->isRunnable(true);
    } else if (!isClient) {
        this->isRunnable(true);
    }
}

rdma::RPCPerf::RPCPerf(string& conns, size_t serverPort, size_t size, size_t iter, size_t threads,std::size_t returnMethod,
        bool old,bool signaledCall) {
    m_conns = StringHelper::split(conns);
    m_serverPort = serverPort;
    m_size = size;
    m_iter = iter;
    m_numThreads = threads;
    m_old = old;
    m_returnMethod = returnMethod;
    m_signaled = signaledCall;
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
    std::cout << "old = "<< m_old << endl;
    std::cout << "returnMethod = "<< m_returnMethod << endl;
    std::cout << "size = " << m_size << endl;
    std::cout << "m_signaled = " << m_signaled << endl;
    if(m_size< sizeof(testFooter)){
        cout << "Error size (-d) must be bigger then " << sizeof(testFooter) << endl;
        exit(1);
    }
    m_dServer = new RDMAServer<ReliableRDMA>("test", m_serverPort);
    size_t srqID = 0;
    m_dServer->createSharedReceiveQueue(srqID);
    m_dServer->activateSRQ(srqID);


    if(m_old){
        the = new TestRPCHandlerThreadOld(m_dServer,srqID,MAX_NUM_RPC_MSG,m_size,m_returnMethod);
    }else{
        switch (m_returnMethod) {
            case 0:
                the = new TestRPCHandlerThreadPoll(m_dServer,srqID,MAX_NUM_RPC_MSG,m_size,m_signaled);
                break;
            case 1:
                the = new TestRPCHandlerThreadImm(m_dServer,srqID,MAX_NUM_RPC_MSG,m_size,m_signaled);
                break;
            case 2:
                the = new TestRPCHandlerThreadSend(m_dServer,srqID,MAX_NUM_RPC_MSG,m_size,m_signaled);
                break;
            default:
                the = new TestRPCHandlerThreadPoll(m_dServer,srqID,MAX_NUM_RPC_MSG,m_size,m_signaled);

        }

    }


    m_dServer->startServer();


    the->startHandler();

    while (m_dServer->isRunning()) {
        usleep(Config::RDMA_SLEEP_INTERVAL);
    }
}

void rdma::RPCPerf::runClient() {
    std::cout << "returnMethod = "<< m_returnMethod << endl;
    std::cout << "size = " << m_size << endl;
    std::cout << "m_signaled = " << m_signaled << endl;

    if(m_size< sizeof(testFooter)){
        cout << "Error size (-d) must be bigger then " << sizeof(testFooter) << endl;
        exit(1);
    }

    //todo make many server work again
    //make this better
    string str = m_conns[0];
    //sequencer is the same IP as the rdma server here
    Config::SEQUENCER_IP =  str.substr(0,str.find(':') );
    //start all client threads
    for (size_t i = 0; i < m_numThreads; i++) {
        std::cout << "Started thread " << i << std::endl;
        auto* perfThread = new RPCPerfThread(m_conns,m_size, m_iter,m_returnMethod,m_signaled);
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


