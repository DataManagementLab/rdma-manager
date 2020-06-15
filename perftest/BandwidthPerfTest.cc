#include "BandwidthPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"


mutex rdma::BandwidthPerfTest::waitLock;
condition_variable rdma::BandwidthPerfTest::waitCv;
bool rdma::BandwidthPerfTest::signaled;

rdma::BandwidthPerfThread::BandwidthPerfThread(std::vector<std::string>& conns, size_t memory_size_per_thread, size_t iterations) {
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_iterations = iterations;
	this->m_is_main_memory = m_is_main_memory;
	m_conns = conns;
	m_remOffsets = new size_t[m_conns.size()];

	for (size_t i = 0; i < m_conns.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_conns[i];
		if (!m_client.connect(conn, nodeId)) {
			throw invalid_argument("BandwidthPerfThread connection failed");
		}
		m_addr.push_back(nodeId);
		m_client.remoteAlloc(conn, m_size, m_remOffsets[i]);
	}

	m_data = m_client.localAlloc(m_size);
	memset(m_data, 1, m_size);
}

rdma::BandwidthPerfThread::~BandwidthPerfThread() {

	m_client.localFree(m_data);

	for (size_t i = 0; i < m_conns.size(); ++i) {
		string conn = m_conns[i];
		m_client.remoteFree(conn, m_remOffsets[i], m_size);
	}
    delete m_remOffsets;

}

void rdma::BandwidthPerfThread::run() {
	unique_lock < mutex > lck(BandwidthPerfTest::waitLock);
	if (!BandwidthPerfTest::signaled) {
		m_ready = true;
		BandwidthPerfTest::waitCv.wait(lck);
	}
	lck.unlock();

	{
		std::cout << "Perf Measurement enabled" << std::endl;
		//PerfEventBlock pb(m_iter);
		//startTimer();
		for (size_t i = 0; i < m_iter; ++i) {
			size_t connIdx = i % m_conns.size();
			bool signaled = (i == (m_iter - 1));
			m_client.write(m_addr[connIdx],m_remOffsets[connIdx],m_data,m_size,signaled);


		}
		//endTimer();
	}	

}


rdma::BandwidthPerfTest::BandwidthPerfTest(bool is_server, std::string nodeIdSequencerAddr, int rdma_port, int gpu_index, int thread_count, uint64_t memory_per_thread, uint64_t iterations) : PerfTest(){
	this->m_is_server = is_server;
	this->m_nodeIdSequencerAddr = nodeIdSequencerAddr;
	this->m_rdma_port = rdma_port;
	this->m_gpu_index = gpu_index;
	this->m_thread_count = thread_count;
	this->m_memory_per_thread = memory_per_thread;
	this->m_memory_size = thread_count * memory_per_thread;
	this->m_iterations = iterations;
}
rdma::BandwidthPerfTest::~BandwidthPerfTest(){
	for (size_t i = 0; i < m_threads.size(); i++) {
		delete m_threads[i];
	}
	m_threads.clear();
	delete m_memory;
	if(m_is_server)
		delete m_server;
	else
		delete m_client;
}

std::string rdma::BandwidthPerfTest::getTestParameters(){
	std::ostringstream oss;
	if(m_is_server){
		oss << "Server | memory=";
	} else {
		oss << "Client | threads=" << m_thread_count << " | memory=";
	}
	oss << memory_size << " (" << m_thread_count << "x " << m_memory_per_thread << ") [";
	if(m_gpu_index < 0){
		oss << "MAIN";
	} else {
		oss << "GPU." << m_gpu_index; 
	}
	oss << "]";
	if(!m_is_server){
		oss << " | iterations=" << m_iterations;
	}
	return oss.str();
}

void rdma::BandwidthPerfTest::setupTest(){
	m_memory = (m_gpu_index<0 ? (rdma::BaseMemory*)new rdma::MainMemory(m_memory_size) : (rdma::BaseMemory*)new rdma::CudaMemory(m_memory_size, m_gpu_index));
}

void rdma::BandwidthPerfTest::runTest(){
	if(m_is_server){
		// Server
		m_server = new RDMAServer<ReliableRDMA>("BandwidthTestRDMAServer", m_rdma_port, m_memory);
		std::cout << "Server listening on " << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << " . . ." << std::endl;
		m_server->startServer();
		while (m_server->isRunning()) {
            usleep(Config::RDMA_SLEEP_INTERVAL);
        }
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client
		m_client = new RDMAClient<ReliableRDMA>(m_memory, "BandwidthTestRDMAClient");

		for (size_t i = 0; i < thread_count; i++) {
            BandwidthPerfThread* perfThread = new BandwidthPerfThread(m_conns, m_memory_per_thread, m_iterations);
            perfThread->start();
            if (!perfThread->ready()) {
                usleep(Config::RDMA_SLEEP_INTERVAL);
            }
            m_threads.push_back(perfThread);
        }

        //send signal to run benchmark
        BandwidthPerfThread::signaled = false;
        unique_lock<mutex> lck(BandwidthPerfThread::waitLock);
        BandwidthPerfThread::waitCv.notify_all();
        BandwidthPerfThread::signaled = true;
        lck.unlock();
        for (size_t i = 0; i < m_threads.size(); i++) {
            m_threads[i]->join();
        }
	}
}


std::string rdma::BandwidthPerfTest::getTestResults(){
	if(is_server){
		return "only client";
	} else {

		// TODO
		return "TODO";

	}
}