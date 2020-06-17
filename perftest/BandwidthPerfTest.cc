#include "BandwidthPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"
#include "../src/utils/StringHelper.h"

mutex rdma::BandwidthPerfTest::waitLock;
condition_variable rdma::BandwidthPerfTest::waitCv;
bool rdma::BandwidthPerfTest::signaled;
char rdma::BandwidthPerfTest::testMode;

rdma::BandwidthPerfThread::BandwidthPerfThread(std::vector<std::string>& rdma_addresses, size_t memory_size_per_thread, size_t iterations) {
	this->m_rdma_addresses = rdma_addresses;
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_iterations = iterations;
	this->m_is_main_memory = m_is_main_memory;
	m_remOffsets = new size_t[m_rdma_addresses.size()];

	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_rdma_addresses[i];
		//std::cout << "Thread trying to connect to '" << conn << "' . . ." << std::endl; // TODO REMOVE
		if(!m_client.connect(conn, nodeId)) {
			std::cerr << "BandwidthPerfThread::BandwidthPerfThread(): Could not connect to '" << conn << "'" << std::endl;
			throw invalid_argument("BandwidthPerfThread connection failed");
		}
		//std::cout << "Thread connected to '" << conn << "'" << std::endl; // TODO REMOVE
		m_addr.push_back(nodeId);
		m_client.remoteAlloc(conn, m_memory_size_per_thread, m_remOffsets[i]);
	}

	m_memory = m_client.localMalloc(m_memory_size_per_thread);
	m_memory->setMemory(1);
}

rdma::BandwidthPerfThread::~BandwidthPerfThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client.remoteFree(addr, m_remOffsets[i], m_memory_size_per_thread);
	}
    delete m_remOffsets;
	delete m_memory;
}

void rdma::BandwidthPerfThread::run() {
	unique_lock<mutex> lck(BandwidthPerfTest::waitLock);
	if (!BandwidthPerfTest::signaled) {
		m_ready = true;
		BandwidthPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	auto start = rdma::PerfTest::startTimer();
	switch(BandwidthPerfTest::testMode){
		case 0x00: // Write
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1));
				m_client.write(m_addr[connIdx], m_remOffsets[connIdx], m_memory->pointer(), m_memory_size_per_thread, signaled);
			}
			m_elapsedWriteMs = rdma::PerfTest::stopTimer(start);
			break;
		case 0x01: // Read
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1));
				m_client.read(m_addr[connIdx], m_remOffsets[connIdx], m_memory->pointer(), m_memory_size_per_thread, signaled);
			}
			m_elapsedReadMs = rdma::PerfTest::stopTimer(start);
			break;
		case 0x02: // Send & Receive
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1));
				m_client.send(m_addr[connIdx], m_memory->pointer(), m_memory_size_per_thread, signaled);
			}
			m_elapsedSendMs = rdma::PerfTest::stopTimer(start);
			break;
		case 0x03: // Fetch & Add
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1));
				m_client.fetchAndAdd(m_addr[connIdx], m_remOffsets[connIdx], m_memory->pointer(), 1, m_memory_size_per_thread, signaled);
			}
			m_elapsedFetchAddMs = rdma::PerfTest::stopTimer(start);
			break;
		case 0x04: // Compare & Swap
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1));
				m_client.compareAndSwap(m_addr[connIdx], m_remOffsets[connIdx], m_memory->pointer(), 2, 3, m_memory_size_per_thread, signaled);
			}
			m_elapsedCompareSwapMs = rdma::PerfTest::stopTimer(start);
			break;
		default: throw invalid_argument("BandwidthPerfThread unknown test mode");
	}
}


rdma::BandwidthPerfTest::BandwidthPerfTest(bool is_server, std::string rdma_addresses, int rdma_port, int gpu_index, int thread_count, uint64_t memory_per_thread, uint64_t iterations) : PerfTest(){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_gpu_index = gpu_index;
	this->m_thread_count = thread_count;
	this->m_memory_per_thread = memory_per_thread;
	this->m_memory_size = thread_count * memory_per_thread;
	this->m_iterations = iterations;
	this->m_rdma_addresses = StringHelper::split(rdma_addresses);
	for (auto &addr : this->m_rdma_addresses){
		addr += ":" + to_string(rdma_port);
	}
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
	oss << m_memory_size << " (" << m_thread_count << "x " << m_memory_per_thread << ") [";
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

void rdma::BandwidthPerfTest::makeThreadsReady(char testMode){
	BandwidthPerfTest::testMode = testMode;
	for(BandwidthPerfThread* perfThread : m_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
}

void rdma::BandwidthPerfTest::runThreads(){
	BandwidthPerfTest::signaled = false;
	unique_lock<mutex> lck(BandwidthPerfTest::waitLock);
	BandwidthPerfTest::waitCv.notify_all();
	BandwidthPerfTest::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_threads.size(); i++) {
		m_threads[i]->join();
	}
}

void rdma::BandwidthPerfTest::setupTest(){
	m_elapsedWriteMs = -1;
	m_elapsedReadMs = -1;
	m_elapsedSendMs = -1;
	m_elapsedFetchAddMs = -1;
	m_elapsedCompareSwapMs = -1;
	m_memory = (m_gpu_index<0 ? (rdma::BaseMemory*)new rdma::MainMemory(m_memory_size) : (rdma::BaseMemory*)new rdma::CudaMemory(m_memory_size, m_gpu_index));

	if(m_is_server){
		// Server
		if (rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) == rdma::Network::getAddressOfConnection(m_rdma_addresses[0])){
			std::cout << "Starting NodeIDSequencer on " << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << rdma::Config::SEQUENCER_PORT << std::endl;
			m_nodeIDSequencer = new NodeIDSequencer();
		}
		m_server = new RDMAServer<ReliableRDMA>("BandwidthTestRDMAServer", m_rdma_port, m_memory);

	} else {
		// Client

		m_client = new RDMAClient<ReliableRDMA>(m_memory, "BandwidthTestRDMAClient");
		for (int i = 0; i < m_thread_count; i++) {
            BandwidthPerfThread* perfThread = new BandwidthPerfThread(m_rdma_addresses, m_memory_per_thread, m_iterations);
            m_threads.push_back(perfThread);
        }
	}
}

void rdma::BandwidthPerfTest::runTest(){
	if(m_is_server){
		// Server
		std::cout << "Starting server on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "' . . ." << std::endl;
		if(!m_server->startServer()){
			std::cerr << "BandwidthPerfTest::runTest(): Could not start server" << std::endl;
			throw invalid_argument("BandwidthPerfTest server startup failed");
		} else {
			std::cout << "Server running on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "'" << std::endl; // TODO REMOVE
		}
		while (m_server->isRunning()) {
            usleep(Config::RDMA_SLEEP_INTERVAL);
        }
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client

        // Measure bandwidth for writing
		makeThreadsReady(0x00); // write
		auto startWrite = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedWriteMs = rdma::PerfTest::stopTimer(startWrite);

		// Measure bandwidth for reading
		makeThreadsReady(0x01); // read
		auto startRead = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedReadMs = rdma::PerfTest::stopTimer(startRead);

		// Measure bandwidth for sending
		/* TODO
		makeThreadsReady(0x02); // send
		auto startSend = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedSendMs = rdma::PerfTest::stopTimer(startSend); */

		// Measure bandwidth for fetching & adding
		/* TODO 
		makeThreadsReady(0x03); // fetch & add
		auto startFetchAdd = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedFetchAddMs = rdma::PerfTest::stopTimer(startFetchAdd); */

		// Measure bandwidth for comparing & swaping
		makeThreadsReady(0x04); // compare & swap
		auto startCompareSwap = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedCompareSwapMs = rdma::PerfTest::stopTimer(startCompareSwap);
	}
}


std::string rdma::BandwidthPerfTest::getTestResults(){
	if(m_is_server){
		return "only client";
	} else {
		
		uint64_t transferedBytes = m_thread_count * m_iterations * m_memory_per_thread;

		std::ostringstream oss;
		oss << "transfered = " << rdma::PerfTest::convertByteSize(transferedBytes) << std::endl;
		oss << " - Write:         time = " << m_elapsedWriteMs << "ms | bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*1000/m_elapsedWriteMs) << std::endl;
		oss << " - Read:          time = " << m_elapsedReadMs << "ms | bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*1000/m_elapsedReadMs) << std::endl;
		oss << " - Send:          time = " << m_elapsedSendMs << "ms | bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*1000/m_elapsedSendMs) << std::endl;
		oss << " - Fetch&Add:     time = " << m_elapsedFetchAddMs << "ms | bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*1000/m_elapsedFetchAddMs) << std::endl;
		oss << " - Compare&Swap:  time = " << m_elapsedCompareSwapMs << "ms | bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*1000/m_elapsedCompareSwapMs) << std::endl;
		return oss.str();

	}
	return NULL;
}