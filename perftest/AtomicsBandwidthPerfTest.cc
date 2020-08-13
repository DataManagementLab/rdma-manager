#include "AtomicsBandwidthPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"

#include <limits>
#include <algorithm>

mutex rdma::AtomicsBandwidthPerfTest::waitLock;
condition_variable rdma::AtomicsBandwidthPerfTest::waitCv;
bool rdma::AtomicsBandwidthPerfTest::signaled;
rdma::TestMode rdma::AtomicsBandwidthPerfTest::testMode;

rdma::AtomicsBandwidthPerfClientThread::AtomicsBandwidthPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, int buffer_slots, size_t iterations) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "AtomicsBandwidthPerfTestClient", ownIpPort, sequencerIpPort);
	this->m_rdma_addresses = rdma_addresses;
	this->m_memory_per_thread = buffer_slots * rdma::ATOMICS_SIZE;
	this->m_buffer_slots = buffer_slots;
	this->m_iterations = iterations;
	m_remOffsets = new size_t[m_rdma_addresses.size()];

	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_rdma_addresses[i];
		//std::cout << "Thread trying to connect to '" << conn << "' . . ." << std::endl; // TODO REMOVE
		if(!m_client->connect(conn, nodeId)) {
			std::cerr << "AtomicsBandwidthPerfThread::BandwidthPerfThread(): Could not connect to '" << conn << "'" << std::endl;
			throw invalid_argument("AtomicsBandwidthPerfThread connection failed");
		}
		//std::cout << "Thread connected to '" << conn << "'" << std::endl; // TODO REMOVE
		m_addr.push_back(nodeId);
		m_client->remoteAlloc(conn, m_memory_per_thread, m_remOffsets[i]);
	}

	m_local_memory = m_client->localMalloc(m_memory_per_thread);
	m_local_memory->openContext();
	m_local_memory->setMemory(1);
}

rdma::AtomicsBandwidthPerfClientThread::~AtomicsBandwidthPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_memory_per_thread, m_remOffsets[i]);
	}
    delete m_remOffsets;
	delete m_local_memory; // implicitly deletes local allocs in RDMAClient
	delete m_client;
}

void rdma::AtomicsBandwidthPerfClientThread::run() {
	unique_lock<mutex> lck(AtomicsBandwidthPerfTest::waitLock);
	if (!AtomicsBandwidthPerfTest::signaled) {
		m_ready = true;
		AtomicsBandwidthPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	auto start = rdma::PerfTest::startTimer();
	switch(AtomicsBandwidthPerfTest::testMode){
		case TEST_FETCH_AND_ADD: // Fetch & Add
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1) || (i+1)%Config::RDMA_MAX_WR==0);
				int offset = (i % m_buffer_slots) * rdma::ATOMICS_SIZE;
				m_client->fetchAndAdd(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), 1, rdma::ATOMICS_SIZE, signaled); // true=signaled
			}
			m_elapsedFetchAddMs = rdma::PerfTest::stopTimer(start);
			break;
		case TEST_COMPARE_AND_SWAP: // Compare & Swap
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1) || (i+1)%Config::RDMA_MAX_WR==0);
				int offset = (i % m_buffer_slots) * rdma::ATOMICS_SIZE;
				m_client->compareAndSwap(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), 2, 3, rdma::ATOMICS_SIZE, signaled); // true=signaled
			}
			m_elapsedCompareSwapMs = rdma::PerfTest::stopTimer(start);
			break;
		default: throw invalid_argument("BandwidthPerfClientThread unknown test mode");
	}
}




rdma::AtomicsBandwidthPerfTest::AtomicsBandwidthPerfTest(bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int thread_count, int buffer_slots, uint64_t iterations) : PerfTest(){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_ownIpPort = ownIpPort;
	this->m_sequencerIpPort = sequencerIpPort;
	this->m_local_gpu_index = local_gpu_index;
	this->m_remote_gpu_index = remote_gpu_index;
	this->m_thread_count = thread_count;
	this->m_memory_size = thread_count * rdma::ATOMICS_SIZE * buffer_slots;
	this->m_buffer_slots = buffer_slots;
	this->m_iterations = iterations;
	this->m_rdma_addresses = rdma_addresses;
}
rdma::AtomicsBandwidthPerfTest::~AtomicsBandwidthPerfTest(){
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		delete m_client_threads[i];
	}
	m_client_threads.clear();
	if(m_is_server)
		delete m_server;
	delete m_memory;
}

std::string rdma::AtomicsBandwidthPerfTest::getTestParameters(){
	std::ostringstream oss;
	oss << (m_is_server ? "Server" : "Client") << ", threads=" << m_thread_count << ", bufferslots=" << m_buffer_slots << ", packetsize=" << rdma::ATOMICS_SIZE << ", memory=";
	oss << m_memory_size << " (" << m_thread_count << "x " << m_buffer_slots << "x " << rdma::ATOMICS_SIZE << ")";
	oss << ", memory_type=" << getMemoryName(m_local_gpu_index) << (m_remote_gpu_index!=-404 ? "->"+getMemoryName(m_remote_gpu_index) : "");
	if(!m_is_server){
		oss << ", iterations=" << (m_iterations*m_thread_count);
	}
	return oss.str();
}

void rdma::AtomicsBandwidthPerfTest::makeThreadsReady(TestMode testMode){
	AtomicsBandwidthPerfTest::testMode = testMode;
	AtomicsBandwidthPerfTest::signaled = false;
	for(AtomicsBandwidthPerfClientThread* perfThread : m_client_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
}

void rdma::AtomicsBandwidthPerfTest::runThreads(){
	AtomicsBandwidthPerfTest::signaled = false;
	unique_lock<mutex> lck(AtomicsBandwidthPerfTest::waitLock);
	AtomicsBandwidthPerfTest::waitCv.notify_all();
	AtomicsBandwidthPerfTest::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		m_client_threads[i]->join();
	}
}

void rdma::AtomicsBandwidthPerfTest::setupTest(){
	m_elapsedFetchAddMs = -1;
	m_elapsedCompareSwapMs = -1;
	#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
		m_memory = (m_local_gpu_index<=-3 ? (rdma::BaseMemory*)new rdma::MainMemory(m_memory_size) : (rdma::BaseMemory*)new rdma::CudaMemory(m_memory_size, m_local_gpu_index));
	#else
		m_memory = (rdma::BaseMemory*)new MainMemory(m_memory_size);
	#endif

	if(m_is_server){
		// Server
		m_server = new RDMAServer<ReliableRDMA>("AtomicsBandwidthTestRDMAServer", m_rdma_port, Network::getAddressOfConnection(m_ownIpPort), m_memory, m_sequencerIpPort);

	} else {
		// Client
		for (int i = 0; i < m_thread_count; i++) {
			AtomicsBandwidthPerfClientThread* perfThread = new AtomicsBandwidthPerfClientThread(m_memory, m_rdma_addresses, m_ownIpPort, m_sequencerIpPort, m_buffer_slots, m_iterations);
			m_client_threads.push_back(perfThread);
		}
	}
}

void rdma::AtomicsBandwidthPerfTest::runTest(){
	if(m_is_server){
		// Server
		std::cout << "Starting server on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "' . . ." << std::endl;
		if(!m_server->startServer()){
			std::cerr << "AtomicsBandwidthPerfTest::runTest(): Could not start server" << std::endl;
			throw invalid_argument("AtomicsBandwidthPerfTest server startup failed");
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

		// Measure bandwidth for fetching & adding
		makeThreadsReady(TEST_FETCH_AND_ADD); // fetch & add
		auto startFetchAdd = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedFetchAddMs = rdma::PerfTest::stopTimer(startFetchAdd); 

		// Measure bandwidth for comparing & swaping
		makeThreadsReady(TEST_COMPARE_AND_SWAP); // compare & swap
		auto startCompareSwap = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedCompareSwapMs = rdma::PerfTest::stopTimer(startCompareSwap);
	}
}


std::string rdma::AtomicsBandwidthPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
	if(m_is_server){
		return "only client";
	} else {

		const long double tu = (long double)NANO_SEC; // 1sec (nano to seconds as time unit)
		
		uint64_t transferedBytesFetchAdd = m_thread_count * m_iterations * rdma::ATOMICS_SIZE * 2; // 8 bytes send + 8 bytes receive
		uint64_t transferedBytesCompSwap = m_thread_count * m_iterations * rdma::ATOMICS_SIZE * 3; // 16 bytes send + 8 bytes receive
		int64_t maxFetchAddMs=-1, minFetchAddMs=std::numeric_limits<int64_t>::max();
		int64_t maxCompareSwapMs=-1, minCompareSwapMs=std::numeric_limits<int64_t>::max();
		int64_t arrFetchAddMs[m_thread_count];
		int64_t arrCompareSwapMs[m_thread_count];
		long double avgFetchAddMs=0, medianFetchAddMs, avgCompareSwapMs=0, medianCompareSwapMs;

		for(size_t i=0; i<m_client_threads.size(); i++){
			AtomicsBandwidthPerfClientThread *thr = m_client_threads[i];
			if(thr->m_elapsedFetchAddMs < minFetchAddMs) minFetchAddMs = thr->m_elapsedFetchAddMs;
			if(thr->m_elapsedFetchAddMs > maxFetchAddMs) maxFetchAddMs = thr->m_elapsedFetchAddMs;
			avgFetchAddMs += (long double) thr->m_elapsedFetchAddMs;
			arrFetchAddMs[i] = thr->m_elapsedFetchAddMs;
			if(thr->m_elapsedCompareSwapMs < minCompareSwapMs) minCompareSwapMs = thr->m_elapsedCompareSwapMs;
			if(thr->m_elapsedCompareSwapMs > maxCompareSwapMs) maxCompareSwapMs = thr->m_elapsedCompareSwapMs;
			avgCompareSwapMs += (long double) thr->m_elapsedCompareSwapMs;
			arrCompareSwapMs[i] = thr->m_elapsedCompareSwapMs;
		}
		avgFetchAddMs /= (long double) m_thread_count;
		avgCompareSwapMs /= (long double) m_thread_count;
		std::sort(arrFetchAddMs, arrFetchAddMs + m_thread_count);
		std::sort(arrCompareSwapMs, arrCompareSwapMs + m_thread_count);
		medianFetchAddMs = arrFetchAddMs[(int)(m_thread_count/2)];
		medianCompareSwapMs = arrCompareSwapMs[(int)(m_thread_count/2)];

		// write results into CSV file
		if(!csvFileName.empty()){
			const uint64_t su = 1000*1000; // size unit (bytes to MegaBytes) | use 1024*1024 for MebiBytes
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			ofs << rdma::CSV_PRINT_NOTATION << rdma::CSV_PRINT_PRECISION;
			if(csvAddHeader){
				ofs << std::endl << "ATOMICS BANDWIDTH, " << getTestParameters() << std::endl;
				ofs << "Iterations, Fetch&Add [MB/s], Comp&Swap [MB/s], Min Fetch&Add [MB/s], Min Comp&Swap [MB/s], ";
				ofs << "Max Fetch&Add [MB/s], Max Comp&Swap [MB/s], Avg Fetch&Add [MB/s], Avg Comp&Swap [MB/s], ";
				ofs << "Median Fetch&Add [MB/s], Median Comp&Swap [MB/s], Fetch&Add [Sec], Comp&Swap [Sec], ";
				ofs << "Min Fetch&Add [Sec], Min Comp&Swap [Sec], Max Fetch&Add [Sec], Max Comp&Swap [Sec], ";
				ofs << "Avg Fetch&Add [Sec], Avg Comp&Swap [Sec], Median Fetch&Add [Sec], Median Comp&Swap [Sec]" << std::endl;
			}
			ofs << m_iterations << ", ";
			ofs << (round(transferedBytesFetchAdd*tu/su/m_elapsedFetchAddMs * 100000)/100000.0) << ", "; // fetch&add MB/s
			ofs << (round(transferedBytesCompSwap*tu/su/m_elapsedCompareSwapMs * 100000)/100000.0) << ", "; // comp&swap MB/s
			ofs << (round(transferedBytesFetchAdd*tu/su/maxFetchAddMs * 100000)/100000.0) << ", "; // min fetch&add MB/s
			ofs << (round(transferedBytesCompSwap*tu/su/maxCompareSwapMs * 100000)/100000.0) << ", "; // min comp&swap MB/s
			ofs << (round(transferedBytesFetchAdd*tu/su/minFetchAddMs * 100000)/100000.0) << ", "; // max fetch&add MB/s
			ofs << (round(transferedBytesCompSwap*tu/su/minCompareSwapMs * 100000)/100000.0) << ", "; // max comp&swap MB/s
			ofs << (round(transferedBytesFetchAdd*tu/su/avgFetchAddMs * 100000)/100000.0) << ", "; // avg fetch&add MB/s
			ofs << (round(transferedBytesCompSwap*tu/su/avgCompareSwapMs * 100000)/100000.0) << ", "; // avg comp&swap MB/s
			ofs << (round(transferedBytesFetchAdd*tu/su/medianFetchAddMs * 100000)/100000.0) << ", "; // median fetch&add MB/s
			ofs << (round(transferedBytesCompSwap*tu/su/medianCompareSwapMs * 100000)/100000.0) << ", "; // median comp&swap MB/s
			ofs << (round(m_elapsedFetchAddMs/tu * 100000)/100000.0) << ", " << (round(m_elapsedCompareSwapMs/tu * 100000)/100000.0) << ", "; // fetch&add, comp&swap Sec
			ofs << (round(minFetchAddMs/tu * 100000)/100000.0) << ", " << (round(minCompareSwapMs/tu * 100000)/100000.0) << ", "; // min fetch&add, comp&swap Sec
			ofs << (round(maxFetchAddMs/tu * 100000)/100000.0) << ", " << (round(maxCompareSwapMs/tu * 100000)/100000.0) << ", "; // max fetch&add, comp&swap Sec
			ofs << (round(avgFetchAddMs/tu * 100000)/100000.0) << ", " << (round(avgCompareSwapMs/tu * 100000)/100000.0) << ", "; // avg fetch&add comp&swap Sec
			ofs << (round(medianFetchAddMs/tu * 100000)/100000.0) << ", " << (round(medianCompareSwapMs/tu * 100000)/100000.0) << std::endl; // median fetch&add, comp&swap Sec
			ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		oss << rdma::CONSOLE_PRINT_NOTATION << rdma::CONSOLE_PRINT_PRECISION;
		oss << std::endl << " - Fetch&Add:     bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytesFetchAdd*tu/m_elapsedFetchAddMs);
		oss << "  (range = " << rdma::PerfTest::convertBandwidth(transferedBytesFetchAdd*tu/maxFetchAddMs) << " - ";
		oss << rdma::PerfTest::convertBandwidth(transferedBytesFetchAdd*tu/minFetchAddMs);
		oss << " ; avg=" << rdma::PerfTest::convertBandwidth(transferedBytesFetchAdd*tu/avgFetchAddMs) << " ; median=";
		oss << rdma::PerfTest::convertBandwidth(transferedBytesFetchAdd*tu/minFetchAddMs) << ")";
		oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedFetchAddMs) << "  (range=";
		oss << rdma::PerfTest::convertTime(minFetchAddMs) << "-" << rdma::PerfTest::convertTime(maxFetchAddMs);
		oss << " ; avg=" << rdma::PerfTest::convertTime(avgFetchAddMs) << " ; median=" << rdma::PerfTest::convertTime(medianFetchAddMs) << ")" << std::endl;
		oss << " - Compare&Swap:  bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytesCompSwap*tu/m_elapsedCompareSwapMs);
		oss << "  (range = " << rdma::PerfTest::convertBandwidth(transferedBytesCompSwap*tu/maxCompareSwapMs) << " - ";
		oss << rdma::PerfTest::convertBandwidth(transferedBytesCompSwap*tu/minCompareSwapMs);
		oss << " ; avg=" << rdma::PerfTest::convertBandwidth(transferedBytesCompSwap*tu/avgCompareSwapMs) << " ; median=";
		oss << rdma::PerfTest::convertBandwidth(transferedBytesCompSwap*tu/minCompareSwapMs) << ")";
		oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedCompareSwapMs) << "  (range=";
		oss << rdma::PerfTest::convertTime(minCompareSwapMs) << "-" << rdma::PerfTest::convertTime(maxCompareSwapMs);
		oss << " ; avg=" << rdma::PerfTest::convertTime(avgCompareSwapMs) << " ; median=" << rdma::PerfTest::convertTime(medianCompareSwapMs) << ")" << std::endl;
		return oss.str();

	}
	return NULL;
}