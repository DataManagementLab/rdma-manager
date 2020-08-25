#include "AtomicsLatencyPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"

mutex rdma::AtomicsLatencyPerfTest::waitLock;
condition_variable rdma::AtomicsLatencyPerfTest::waitCv;
bool rdma::AtomicsLatencyPerfTest::signaled;
rdma::TestMode rdma::AtomicsLatencyPerfTest::testMode;



rdma::AtomicsLatencyPerfClientThread::AtomicsLatencyPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, int buffer_slots, size_t iterations_per_thread) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "AtomicsLatencyPerfTestClient", ownIpPort, sequencerIpPort);
	this->m_rdma_addresses = rdma_addresses;
	this->m_memory_per_thread = buffer_slots * rdma::ATOMICS_SIZE;
	this->m_buffer_slots = buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_remOffsets = new size_t[m_rdma_addresses.size()];
	this->m_arrFetchAddMs = new int64_t[iterations_per_thread];
	this->m_arrCompareSwapMs = new int64_t[iterations_per_thread];

	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_rdma_addresses[i];
		//std::cout << "Thread trying to connect to '" << conn << "' . . ." << std::endl; // TODO REMOVE
		if(!m_client->connect(conn, nodeId)) {
			std::cerr << "AtomicsLatencyPerfThread::LatencyPerfThread(): Could not connect to '" << conn << "'" << std::endl;
			throw invalid_argument("AtomicsLatencyPerfThread connection failed");
		}
		//std::cout << "Thread connected to '" << conn << "'" << std::endl; // TODO REMOVE
		m_addr.push_back(nodeId);
		m_client->remoteAlloc(conn, m_memory_per_thread, m_remOffsets[i]);
	}

	m_local_memory = m_client->localMalloc(m_memory_per_thread);
	m_local_memory->openContext();
	m_local_memory->setMemory(0);
}

rdma::AtomicsLatencyPerfClientThread::~AtomicsLatencyPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_memory_per_thread, m_remOffsets[i]);
	}
    delete m_remOffsets;
	delete m_local_memory; // implicitly deletes local allocs in RDMAClient
	delete m_client;

	delete m_arrFetchAddMs;
	delete m_arrCompareSwapMs;
}

void rdma::AtomicsLatencyPerfClientThread::run() {
	m_local_memory->setMemory(0);
	unique_lock<mutex> lck(AtomicsLatencyPerfTest::waitLock);
	if (!AtomicsLatencyPerfTest::signaled) {
		m_ready = true;
		AtomicsLatencyPerfTest::waitCv.wait(lck);
	}
	lck.unlock();

	switch(AtomicsLatencyPerfTest::testMode){
		case TEST_FETCH_AND_ADD: // Fetch & Add
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				int offset = (i % m_buffer_slots) * rdma::ATOMICS_SIZE;
				auto start = rdma::PerfTest::startTimer();
				m_client->fetchAndAdd(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), 2, rdma::ATOMICS_SIZE, true); // true=signaled
				int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
				m_sumFetchAddMs += time;
				if(m_minFetchAddMs > time) m_minFetchAddMs = time;
				if(m_maxFetchAddMs < time) m_maxFetchAddMs = time;
				m_arrFetchAddMs[i] = time;
			}
			break;
		case TEST_COMPARE_AND_SWAP: // Compare & Swap
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				int offset = (i % m_buffer_slots) * rdma::ATOMICS_SIZE;
				auto start = rdma::PerfTest::startTimer();
				m_client->compareAndSwap(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), i, i+1, rdma::ATOMICS_SIZE, true); // true=signaled
				int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
				m_sumCompareSwapMs += time;
				if(m_minCompareSwapMs > time) m_minCompareSwapMs = time;
				if(m_maxCompareSwapMs < time) m_maxCompareSwapMs = time;
				m_arrCompareSwapMs[i] = time;
			}
			break;
		default: throw invalid_argument("LatencyPerfClientThread unknown test mode");
	}
}


rdma::AtomicsLatencyPerfTest::AtomicsLatencyPerfTest(bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int thread_count, int buffer_slots, uint64_t iterations_per_thread) : PerfTest(){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_ownIpPort = ownIpPort;
	this->m_sequencerIpPort = sequencerIpPort;
	this->m_local_gpu_index = local_gpu_index;
	this->m_remote_gpu_index = remote_gpu_index;
	this->m_thread_count = thread_count;
	this->m_memory_size = thread_count * rdma::ATOMICS_SIZE * buffer_slots;
	this->m_buffer_slots = buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_rdma_addresses = rdma_addresses;
}
rdma::AtomicsLatencyPerfTest::~AtomicsLatencyPerfTest(){
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		delete m_client_threads[i];
	}
	m_client_threads.clear();
	if(m_is_server)
		delete m_server;
	delete m_memory;
}

std::string rdma::AtomicsLatencyPerfTest::getTestParameters(bool forCSV){
	std::ostringstream oss;
	oss << (m_is_server ? "Server" : "Client") << ", threads=" << m_thread_count << ", bufferslots=" << m_buffer_slots << ", packetsize=" << rdma::ATOMICS_SIZE << ", memory=";
	oss << m_memory_size << " (" << m_thread_count << "x " << m_buffer_slots << "x " << rdma::ATOMICS_SIZE << ")";
	oss << ", memory_type=" << getMemoryName(m_local_gpu_index) << (m_remote_gpu_index!=-404 ? "->"+getMemoryName(m_remote_gpu_index) : "");
	if(!forCSV){ oss << ", iterations=" << (m_iterations_per_thread*m_thread_count); }
	return oss.str();
}
std::string rdma::AtomicsLatencyPerfTest::getTestParameters(){
	return getTestParameters(false);
}

void rdma::AtomicsLatencyPerfTest::makeThreadsReady(TestMode testMode){
	AtomicsLatencyPerfTest::testMode = testMode;
	AtomicsLatencyPerfTest::signaled = false;
	for(AtomicsLatencyPerfClientThread* perfThread : m_client_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
}

void rdma::AtomicsLatencyPerfTest::runThreads(){
	AtomicsLatencyPerfTest::signaled = false;
	unique_lock<mutex> lck(AtomicsLatencyPerfTest::waitLock);
	AtomicsLatencyPerfTest::waitCv.notify_all();
	AtomicsLatencyPerfTest::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		m_client_threads[i]->join();
	}
}

void rdma::AtomicsLatencyPerfTest::setupTest(){
	#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
		m_memory = (m_local_gpu_index<=-3 ? (rdma::BaseMemory*)new rdma::MainMemory(m_memory_size) : (rdma::BaseMemory*)new rdma::CudaMemory(m_memory_size, m_local_gpu_index));
	#else
		m_memory = (rdma::BaseMemory*)new MainMemory(m_memory_size);
	#endif

	if(m_is_server){
		// Server
		m_server = new RDMAServer<ReliableRDMA>("LatencyTestRDMAServer", m_rdma_port, Network::getAddressOfConnection(m_ownIpPort), m_memory, m_sequencerIpPort);

	} else {
		// Client
		for (int i = 0; i < m_thread_count; i++) {
			AtomicsLatencyPerfClientThread* perfThread = new AtomicsLatencyPerfClientThread(m_memory, m_rdma_addresses, m_ownIpPort, m_sequencerIpPort, m_buffer_slots, m_iterations_per_thread);
			m_client_threads.push_back(perfThread);
		}
	}
}

void rdma::AtomicsLatencyPerfTest::runTest(){
	if(m_is_server){
		// Server
		std::cout << "Starting server on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "' . . ." << std::endl;
		if(!m_server->startServer()){
			std::cerr << "AtomicsLatencyPerfTest::runTest(): Could not start server" << std::endl;
			throw invalid_argument("AtomicsLatencyPerfTest server startup failed");
		} else {
			std::cout << "Server running on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "'" << std::endl; // TODO REMOVE
		}

		// waiting until clients have connected
		while(m_server->getConnectedConnIDs().size() < (size_t)m_thread_count) usleep(Config::RDMA_SLEEP_INTERVAL);

		// wait until clients have finished
		while (m_server->isRunning() && m_server->getConnectedConnIDs().size() > 0) usleep(Config::RDMA_SLEEP_INTERVAL);
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client

		// Measure Latency for fetching & adding
		makeThreadsReady(TEST_FETCH_AND_ADD); // fetch & add
        runThreads();

		// Measure Latency for comparing & swaping
		makeThreadsReady(TEST_COMPARE_AND_SWAP); // compare & swap
        runThreads();
	}
}


std::string rdma::AtomicsLatencyPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
	if(m_is_server){
		return "only client";
	} else {

		/*	There are  n  threads
			Each thread computes  iterations_per_thread = total_iterations / n
			Each thread takes  n  times more time compared to a single thread
			  Latency 	= elapsedTime / iterations
						= (elpasedTime_per_thread / n) / (n * iterations_per_thread)
		*/

		int64_t minFetchAddMs=std::numeric_limits<int64_t>::max(), maxFetchAddMs=-1, medianFetchAddMs=-1;
		int64_t minCompareSwapMs=std::numeric_limits<int64_t>::max(), maxCompareSwapMs=-1, medianCompareSwapMs=-1;
		long double avgFetchAddMs=0, avgCompareSwapMs=0;
		int64_t mediansFetchAddNs[m_thread_count], mediansCompareSwapNs[m_thread_count];
        const long double divAvg = m_client_threads.size() * m_iterations_per_thread; // for calculating average
		for(size_t i=0; i<m_client_threads.size(); i++){
			AtomicsLatencyPerfClientThread *thr = m_client_threads[i];
			if(minFetchAddMs > thr->m_minFetchAddMs) minFetchAddMs = thr->m_minFetchAddMs;
			if(maxFetchAddMs < thr->m_maxFetchAddMs) maxFetchAddMs = thr->m_maxFetchAddMs;
			avgFetchAddMs += thr->m_sumFetchAddMs / divAvg;
			if(minCompareSwapMs > thr->m_minCompareSwapMs) minCompareSwapMs = thr->m_minCompareSwapMs;
			if(maxCompareSwapMs < thr->m_maxCompareSwapMs) maxCompareSwapMs = thr->m_maxCompareSwapMs;
			avgCompareSwapMs += thr->m_sumCompareSwapMs / divAvg;
			
			std::sort(thr->m_arrFetchAddMs, thr->m_arrFetchAddMs + m_iterations_per_thread);
			std::sort(thr->m_arrCompareSwapMs, thr->m_arrCompareSwapMs + m_iterations_per_thread);
			mediansFetchAddNs[i] = thr->m_arrFetchAddMs[(int)(m_iterations_per_thread/2)];
			mediansCompareSwapNs[i] = thr->m_arrCompareSwapMs[(int)(m_iterations_per_thread/2)];
		}

		std::sort(mediansFetchAddNs, mediansFetchAddNs + m_thread_count);
		std::sort(mediansCompareSwapNs, mediansCompareSwapNs + m_thread_count);
		medianFetchAddMs = mediansFetchAddNs[(int)(m_thread_count/2)];
		medianCompareSwapMs = mediansCompareSwapNs[(int)(m_thread_count/2)];

		// write results into CSV file
		if(!csvFileName.empty()){
			const uint64_t ustu = 1000; // nanosec to microsec
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			ofs << rdma::CSV_PRINT_NOTATION << rdma::CSV_PRINT_PRECISION;
			if(csvAddHeader){
				ofs << std::endl << "ATOMICS LATENCY, " << getTestParameters(true) << std::endl;
				ofs << "Iterations, Avg Fetch&Add [usec], Avg Comp&Swap [usec], Median Fetch&Add [usec], Median Comp&Swap [usec], ";
				ofs << "Min Fetch&Add [usec], Min Comp&Swap [usec], Max Fetch&Add [usec], Max Comp&Swap [usec]" << std::endl;
			}
			ofs << (m_iterations_per_thread * m_thread_count) << ", ";
			ofs << (round(avgFetchAddMs/ustu * 10)/10.0) << ", " << (round(avgCompareSwapMs/ustu * 10)/10.0) << ", "; // avg fetch&add, comp&swap us
			ofs << (round(medianFetchAddMs/ustu * 10)/10.0) << ", " << (round(medianCompareSwapMs/ustu * 10)/10.0) << ", "; // median fetch&add, comp&swap us
			ofs << (round(minFetchAddMs/ustu * 10)/10.0) << ", " << (round(minCompareSwapMs/ustu * 10)/10.0) << ", "; // min fetch&add, comp&swap us
			ofs << (round(maxFetchAddMs/ustu * 10)/10.0) << ", " << (round(maxCompareSwapMs/ustu * 10)/10.0) << std::endl; // max fetch&add, comp&swap us
			ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		oss << rdma::CONSOLE_PRINT_NOTATION << rdma::CONSOLE_PRINT_PRECISION;
		oss << " Measured as 'one trip time' latencies:" << std::endl;
		oss << " - Fetch&Add:       average = " << rdma::PerfTest::convertTime(avgFetchAddMs) << "    median = " << rdma::PerfTest::convertTime(medianFetchAddMs);
		oss << "    range = " <<  rdma::PerfTest::convertTime(minFetchAddMs) << " - " << rdma::PerfTest::convertTime(maxFetchAddMs) << std::endl;
		oss << " - Compare&Swap:    average = " << rdma::PerfTest::convertTime(avgCompareSwapMs) << "   median = " << rdma::PerfTest::convertTime(medianCompareSwapMs);
		oss << "    range = " <<  rdma::PerfTest::convertTime(minCompareSwapMs) << " - " << rdma::PerfTest::convertTime(maxCompareSwapMs) << std::endl;
		return oss.str();

	}
	return NULL;
}
