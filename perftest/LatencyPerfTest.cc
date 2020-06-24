#include "LatencyPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"
#include "../src/utils/StringHelper.h"

mutex rdma::LatencyPerfTest::waitLock;
condition_variable rdma::LatencyPerfTest::waitCv;
bool rdma::LatencyPerfTest::signaled;
rdma::TestMode rdma::LatencyPerfTest::testMode;



rdma::LatencyPerfClientThread::LatencyPerfClientThread(RDMAClient<ReliableRDMA> *client, std::vector<std::string>& rdma_addresses, size_t memory_size_per_thread, size_t iterations) {
	this->m_client = client;
	this->m_rdma_addresses = rdma_addresses;
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_iterations = iterations;
	this->m_remOffsets = new size_t[m_rdma_addresses.size()];

	this->m_arrWriteMs = new int64_t[iterations];
	this->m_arrReadMs = new int64_t[iterations];
	this->m_arrSendMs = new int64_t[iterations];
	this->m_arrFetchAddMs = new int64_t[iterations];
	this->m_arrCompareSwapMs = new int64_t[iterations];

	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_rdma_addresses[i];
		//std::cout << "Thread trying to connect to '" << conn << "' . . ." << std::endl; // TODO REMOVE
		if(!m_client->connect(conn, nodeId)) {
			std::cerr << "LatencyPerfThread::LatencyPerfThread(): Could not connect to '" << conn << "'" << std::endl;
			throw invalid_argument("LatencyPerfThread connection failed");
		}
		//std::cout << "Thread connected to '" << conn << "'" << std::endl; // TODO REMOVE
		m_addr.push_back(nodeId);
		m_client->remoteAlloc(conn, m_memory_size_per_thread, m_remOffsets[i]);
	}

	m_local_memory = m_client->localMalloc(m_memory_size_per_thread);
	m_local_memory->setMemory(1);
}

rdma::LatencyPerfClientThread::~LatencyPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_remOffsets[i], m_memory_size_per_thread);
	}
    delete m_remOffsets;
	delete m_local_memory; // implicitly deletes local allocs in RDMAClient

	delete m_arrWriteMs;
	delete m_arrReadMs;
	delete m_arrSendMs;
	delete m_arrFetchAddMs;
	delete m_arrCompareSwapMs;
}

void rdma::LatencyPerfClientThread::run() {
	unique_lock<mutex> lck(LatencyPerfTest::waitLock);
	if (!LatencyPerfTest::signaled) {
		m_ready = true;
		LatencyPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	switch(LatencyPerfTest::testMode){
		case TEST_WRITE: // Write
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				auto start = rdma::PerfTest::startTimer();
				// because signaled=true the operation blocks until the write is fully executed
				m_client->write(m_addr[connIdx], m_remOffsets[connIdx], m_local_memory->pointer(), m_memory_size_per_thread, true); // true=signaled
				int64_t time = rdma::PerfTest::stopTimer(start);
				m_sumWriteMs += time;
				if(m_minWriteMs > time) m_minWriteMs = time;
				if(m_maxWriteMs < time) m_maxWriteMs = time;
				m_arrWriteMs[i] = time;
			}
			break;
		case TEST_READ: // Read
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				auto start = rdma::PerfTest::startTimer();
				m_client->read(m_addr[connIdx], m_remOffsets[connIdx], m_local_memory->pointer(), m_memory_size_per_thread, true); // true=signaled
				int64_t time = rdma::PerfTest::stopTimer(start);
				m_sumReadMs += time;
				if(m_minReadMs > time) m_minReadMs = time;
				if(m_maxReadMs < time) m_maxReadMs = time;
				m_arrReadMs[i] = time;
			}
			break;
		case TEST_SEND_AND_RECEIVE: // Send & Receive
			for(size_t i = 0; i < m_rdma_addresses.size(); i++){
				// TODO REMOVE std::cout << "Receive: " << 0 << "." << j << std::endl; // TODO REMOVE
				m_client->receive(m_addr[i], m_local_memory->pointer(), m_memory_size_per_thread);
			}
			for(size_t i = 0; i < m_iterations; i++){
				size_t clientId = m_addr[i % m_rdma_addresses.size()];
				auto start = rdma::PerfTest::startTimer();
				// TODO REMOVE std::cout << "Send: " << (i+j) << std::endl; // TODO REMOVE
				m_client->send(clientId, m_local_memory->pointer(), m_memory_size_per_thread, true); // true=signaled
				// TODO REMOVE std::cout << "PollReceive: " << i << "." << j << std::endl; // TODO REMOVE
				m_client->pollReceive(clientId, true); // true=poll
				// TODO REMOVE std::cout << "Receive: " << 0 << "." << j << std::endl; // TODO REMOVE
				m_client->receive(clientId, m_local_memory->pointer(), m_memory_size_per_thread);
				int64_t time = rdma::PerfTest::stopTimer(start) / 2; // half round trip time
				m_sumSendMs += time;
				if(m_minSendMs > time) m_minSendMs = time;
				if(m_maxSendMs < time) m_maxSendMs = time;
				m_arrSendMs[i] = time;
			}
			break;
		case TEST_FETCH_AND_ADD: // Fetch & Add
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				auto start = rdma::PerfTest::startTimer();
				m_client->fetchAndAdd(m_addr[connIdx], m_remOffsets[connIdx], m_local_memory->pointer(), m_memory_size_per_thread, true); // true=signaled
				int64_t time = rdma::PerfTest::stopTimer(start);
				m_sumFetchAddMs += time;
				if(m_minFetchAddMs > time) m_minFetchAddMs = time;
				if(m_maxFetchAddMs < time) m_maxFetchAddMs = time;
				m_arrFetchAddMs[i] = time;
			}
			break;
		case TEST_COMPARE_AND_SWAP: // Compare & Swap
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				auto start = rdma::PerfTest::startTimer();
				m_client->compareAndSwap(m_addr[connIdx], m_remOffsets[connIdx], m_local_memory->pointer(), 2, 3, m_memory_size_per_thread, true); // true=signaled
				int64_t time = rdma::PerfTest::stopTimer(start);
				m_sumCompareSwapMs += time;
				if(m_minCompareSwapMs > time) m_minCompareSwapMs = time;
				if(m_maxCompareSwapMs < time) m_maxCompareSwapMs = time;
				m_arrCompareSwapMs[i] = time;
			}
			break;
		default: throw invalid_argument("LatencyPerfClientThread unknown test mode");
	}
}



rdma::LatencyPerfServerThread::LatencyPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t memory_size_per_thread, size_t iterations) {
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_iterations = iterations;
	this->m_server = server;
	this->m_local_memory = server->localMalloc(memory_size_per_thread);
}

rdma::LatencyPerfServerThread::~LatencyPerfServerThread() {
	delete m_local_memory;  // implicitly deletes local allocs in RDMAServer
}

void rdma::LatencyPerfServerThread::run() {
	unique_lock<mutex> lck(LatencyPerfTest::waitLock);
	if (!LatencyPerfTest::signaled) {
		m_ready = true;
		LatencyPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	const std::vector<size_t> clientIds = m_server->getConnectedConnIDs();
	
	for(size_t i = 0; i < clientIds.size(); i++){
		// TODO REMOVE std::cout << "Receive: " << (i+j) << std::endl; // TODO REMOVE
		m_server->receive(clientIds[i], m_local_memory->pointer(), m_memory_size_per_thread);
	}

	// Measure Latency for receiving
	for(size_t i = 0; i < m_iterations; i++){
		size_t clientId = clientIds[i % clientIds.size()];
		// TODO REMOVE std::cout << "PollReceive: " << (i+j) << std::endl; // TODO REMOVE
		m_server->pollReceive(clientId, true); // true=poll
		// TODO REMOVE std::cout << "Receive: " << (i+j) << std::endl; // TODO REMOVE
		m_server->receive(clientId, m_local_memory->pointer(), m_memory_size_per_thread);
		// TODO REMOVE std::cout << "Send: " << i << "." << j << std::endl; // TODO REMOVE
		m_server->send(clientId, m_local_memory->pointer(), m_memory_size_per_thread, true); // true=signaled
	}
}




rdma::LatencyPerfTest::LatencyPerfTest(bool is_server, std::string rdma_addresses, int rdma_port, int gpu_index, int thread_count, uint64_t memory_size_per_thread, uint64_t iterations) : PerfTest(){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_gpu_index = gpu_index;
	this->m_thread_count = thread_count;
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_memory_size = 2 * thread_count * memory_size_per_thread; // 2x because for send & receive separat
	this->m_iterations = iterations;
	this->m_rdma_addresses = StringHelper::split(rdma_addresses);
	for (auto &addr : this->m_rdma_addresses){
		addr += ":" + to_string(rdma_port);
	}
}
rdma::LatencyPerfTest::~LatencyPerfTest(){
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		delete m_client_threads[i];
	}
	m_client_threads.clear();
	for (size_t i = 0; i < m_server_threads.size(); i++) {
		delete m_server_threads[i];
	}
	m_server_threads.clear();
	if(m_is_server)
		delete m_server;
	else
		delete m_client;
	delete m_memory;
}

std::string rdma::LatencyPerfTest::getTestParameters(){
	std::ostringstream oss;
	if(m_is_server){
		oss << "Server | memory=";
	} else {
		oss << "Client | threads=" << m_thread_count << " | memory=";
	}
	oss << m_memory_size << " (2x " << m_thread_count << "x " << m_memory_size_per_thread << ") [";
	if(m_gpu_index < 0){
		oss << "MAIN";
	} else {
		oss << "GPU." << m_gpu_index; 
	}
	oss << " mem]";
	if(!m_is_server){
		oss << " | iterations=" << m_iterations;
	}
	return oss.str();
}

void rdma::LatencyPerfTest::makeThreadsReady(TestMode testMode){
	LatencyPerfTest::testMode = testMode;
	for(LatencyPerfServerThread* perfThread : m_server_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
	for(LatencyPerfClientThread* perfThread : m_client_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
}

void rdma::LatencyPerfTest::runThreads(){
	LatencyPerfTest::signaled = false;
	unique_lock<mutex> lck(LatencyPerfTest::waitLock);
	LatencyPerfTest::waitCv.notify_all();
	LatencyPerfTest::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_server_threads.size(); i++) {
		m_server_threads[i]->join();
	}
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		m_client_threads[i]->join();
	}
}

void rdma::LatencyPerfTest::setupTest(){
	m_memory = (m_gpu_index<0 ? (rdma::BaseMemory*)new rdma::MainMemory(m_memory_size) : (rdma::BaseMemory*)new rdma::CudaMemory(m_memory_size, m_gpu_index));

	if(m_is_server){
		// NodeIDSequencer (Server)
		if (rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) == rdma::Network::getAddressOfConnection(m_rdma_addresses[0])){
			std::cout << "Starting NodeIDSequencer on " << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << rdma::Config::SEQUENCER_PORT << std::endl;
			m_nodeIDSequencer = new NodeIDSequencer();
		}
		// Server
		m_server = new RDMAServer<ReliableRDMA>("LatencyTestRDMAServer", m_rdma_port, m_memory);
		for (int i = 0; i < m_thread_count; i++) {
			LatencyPerfServerThread* perfThread = new LatencyPerfServerThread(m_server, m_memory_size_per_thread, m_iterations);
			m_server_threads.push_back(perfThread);
		}

	} else {
		// Client
		m_client = new RDMAClient<ReliableRDMA>(m_memory, "LatencyTestRDMAClient");
		for (int i = 0; i < m_thread_count; i++) {
			LatencyPerfClientThread* perfThread = new LatencyPerfClientThread(m_client, m_rdma_addresses, m_memory_size_per_thread, m_iterations);
			m_client_threads.push_back(perfThread);
		}
	}
}

void rdma::LatencyPerfTest::runTest(){
	if(m_is_server){
		// Server
		std::cout << "Starting server on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "' . . ." << std::endl;
		if(!m_server->startServer()){
			std::cerr << "LatencyPerfTest::runTest(): Could not start server" << std::endl;
			throw invalid_argument("LatencyPerfTest server startup failed");
		} else {
			std::cout << "Server running on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "'" << std::endl; // TODO REMOVE
		}

		// waiting until clients have connected
		while(m_server->getConnectedConnIDs().size() < (size_t)m_thread_count) usleep(Config::RDMA_SLEEP_INTERVAL);

		makeThreadsReady(TEST_SEND_AND_RECEIVE); // receive
        runThreads();

		// wait until server is done
		while (m_server->isRunning() && m_server->getConnectedConnIDs().size() > 0) {
			std::cout << "Server waiting until clients disconnect: " << m_server->getConnectedConnIDs().size() << std::endl; // TODO REMOVE
			 usleep(Config::RDMA_SLEEP_INTERVAL);
        }
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client

        // Measure Latency for writing
		makeThreadsReady(TEST_WRITE); // write
        runThreads();

		// Measure Latency for reading
		makeThreadsReady(TEST_READ); // read
        runThreads();

		// Measure Latency for sending
		makeThreadsReady(TEST_SEND_AND_RECEIVE); // send
        runThreads();

		// Measure Latency for fetching & adding
		makeThreadsReady(TEST_FETCH_AND_ADD); // fetch & add
        runThreads();

		// Measure Latency for comparing & swaping
		makeThreadsReady(TEST_COMPARE_AND_SWAP); // compare & swap
        runThreads();
	}
}


std::string rdma::LatencyPerfTest::getTestResults(){
	if(m_is_server){
		return "only client";
	} else {

		int64_t minWriteMs=std::numeric_limits<int64_t>::max(), maxWriteMs=-1, medianWriteMs=-1;
		int64_t minReadMs=std::numeric_limits<int64_t>::max(), maxReadMs=-1, medianReadMs=-1;
		int64_t minSendMs=std::numeric_limits<int64_t>::max(), maxSendMs=-1, medianSendMs=-1;
		int64_t minFetchAddMs=std::numeric_limits<int64_t>::max(), maxFetchAddMs=-1, medianFetchAddMs=-1;
		int64_t minCompareSwapMs=std::numeric_limits<int64_t>::max(), maxCompareSwapMs=-1, medianCompareSwapMs=-1;
		long double avgWriteMs=0, avgReadMs=0, avgSendMs=0, avgFetchAddMs=0, avgCompareSwapMs=0;
		int64_t tmpMs[m_iterations];

		uint64_t index = 0;
		for(size_t i=0; i<m_client_threads.size(); i++){
			long double itr = (long double)m_iterations;
			LatencyPerfClientThread *thr = m_client_threads[i];
			if(minWriteMs > thr->m_minWriteMs) minWriteMs = thr->m_minWriteMs;
			if(maxWriteMs < thr->m_maxWriteMs) maxWriteMs = thr->m_maxWriteMs;
			avgWriteMs += thr->m_sumWriteMs / itr;
			if(minReadMs > thr->m_minReadMs) minReadMs = thr->m_minReadMs;
			if(maxReadMs < thr->m_maxReadMs) maxReadMs = thr->m_maxReadMs;
			avgReadMs += thr->m_sumReadMs / itr;
			if(minSendMs > thr->m_minSendMs) minSendMs = thr->m_minSendMs;
			if(maxSendMs < thr->m_maxSendMs) maxSendMs = thr->m_maxSendMs;
			avgSendMs += thr->m_sumSendMs / itr;
			if(minFetchAddMs > thr->m_minFetchAddMs) minFetchAddMs = thr->m_minFetchAddMs;
			if(maxFetchAddMs < thr->m_maxFetchAddMs) maxFetchAddMs = thr->m_maxFetchAddMs;
			avgFetchAddMs += thr->m_sumFetchAddMs / itr;
			if(minCompareSwapMs > thr->m_minCompareSwapMs) minCompareSwapMs = thr->m_minCompareSwapMs;
			if(maxCompareSwapMs < thr->m_maxCompareSwapMs) maxCompareSwapMs = thr->m_maxCompareSwapMs;
			avgCompareSwapMs += thr->m_sumCompareSwapMs / itr;

			for(uint64_t j = 0; j < m_iterations; j++){ tmpMs[index++] = thr->m_arrWriteMs[j]; }
			std::sort(tmpMs, tmpMs + m_iterations);
			medianWriteMs = tmpMs[(int)(m_iterations/2)];
		}

		index = 0;
		for(size_t i=0; i<m_client_threads.size(); i++){
			LatencyPerfClientThread *thr = m_client_threads[i];
			for(uint64_t j = 0; j < m_iterations; j++){ tmpMs[index++] = thr->m_arrReadMs[j]; }
			std::sort(tmpMs, tmpMs + m_iterations);
			medianReadMs = tmpMs[(int)(m_iterations/2)];
		}

		index = 0;
		for(size_t i=0; i<m_client_threads.size(); i++){
			LatencyPerfClientThread *thr = m_client_threads[i];
			for(uint64_t j = 0; j < m_iterations; j++){ tmpMs[index++] = thr->m_arrSendMs[j]; }
			std::sort(tmpMs, tmpMs + m_iterations);
			medianSendMs = tmpMs[(int)(m_iterations/2)];
		}

		index = 0;
		for(size_t i=0; i<m_client_threads.size(); i++){
			LatencyPerfClientThread *thr = m_client_threads[i];
			for(uint64_t j = 0; j < m_iterations; j++){ tmpMs[index++] = thr->m_arrFetchAddMs[j]; }
			std::sort(tmpMs, tmpMs + m_iterations);
			medianFetchAddMs = tmpMs[(int)(m_iterations/2)];
		}

		index = 0;
		for(size_t i=0; i<m_client_threads.size(); i++){
			LatencyPerfClientThread *thr = m_client_threads[i];
			for(uint64_t j = 0; j < m_iterations; j++){ tmpMs[index++] = thr->m_arrCompareSwapMs[j]; }
			std::sort(tmpMs, tmpMs + m_iterations);
			medianCompareSwapMs = tmpMs[(int)(m_iterations/2)];
		}

		std::ostringstream oss;
		std::cout << "Measured as 'one trip time' latencies:" << std::endl;
		std::cout << " - Write:           average = " << rdma::PerfTest::convertTime(avgWriteMs) << "    median = " << rdma::PerfTest::convertTime(medianWriteMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minWriteMs) << " - " << rdma::PerfTest::convertTime(maxWriteMs) << std::endl;
		std::cout << " - Read:            average = " << rdma::PerfTest::convertTime(avgReadMs) << "    median = " << rdma::PerfTest::convertTime(medianReadMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minReadMs) << " - " << rdma::PerfTest::convertTime(maxReadMs) << std::endl;
		std::cout << " - Send:            average = " << rdma::PerfTest::convertTime(avgSendMs) << "    median = " << rdma::PerfTest::convertTime(medianSendMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minSendMs) << " - " << rdma::PerfTest::convertTime(maxSendMs) << std::endl;
		std::cout << " - Fetch&Add:       average = " << rdma::PerfTest::convertTime(avgFetchAddMs) << "    median = " << rdma::PerfTest::convertTime(medianFetchAddMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minFetchAddMs) << " - " << rdma::PerfTest::convertTime(maxFetchAddMs) << std::endl;
		std::cout << " - Compare&Swap:    average = " << rdma::PerfTest::convertTime(avgCompareSwapMs) << "   median = " << rdma::PerfTest::convertTime(medianCompareSwapMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minCompareSwapMs) << " - " << rdma::PerfTest::convertTime(maxCompareSwapMs) << std::endl;
		return oss.str();

	}
	return NULL;
}
