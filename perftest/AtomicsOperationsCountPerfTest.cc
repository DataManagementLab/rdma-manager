#include "AtomicsOperationsCountPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"

#include <limits>
#include <algorithm>

mutex rdma::AtomicsOperationsCountPerfTest::waitLock;
condition_variable rdma::AtomicsOperationsCountPerfTest::waitCv;
bool rdma::AtomicsOperationsCountPerfTest::signaled;
rdma::TestOperation rdma::AtomicsOperationsCountPerfTest::testOperation;

rdma::AtomicsOperationsCountPerfClientThread::AtomicsOperationsCountPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, int buffer_slots, size_t iterations_per_thread) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "AtomicsOperationsCountPerfTestClient", ownIpPort, sequencerIpPort);
	this->m_rdma_addresses = rdma_addresses;
	this->m_memory_per_thread = buffer_slots * rdma::ATOMICS_SIZE;
	this->m_buffer_slots = buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	m_remOffsets = new size_t[m_rdma_addresses.size()];

	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_rdma_addresses[i];
		//std::cout << "Thread trying to connect to '" << conn << "' . . ." << std::endl; // TODO REMOVE
		if(!m_client->connect(conn, nodeId)) {
			std::cerr << "AtomicsOperationsCountPerfClientThread::BandwidthPerfThread(): Could not connect to '" << conn << "'" << std::endl;
			throw invalid_argument("AtomicsOperationsCountPerfClientThread connection failed");
		}
		//std::cout << "Thread connected to '" << conn << "'" << std::endl; // TODO REMOVE
		m_addr.push_back(nodeId);
		m_client->remoteAlloc(conn, m_memory_per_thread, m_remOffsets[i]);
	}

	m_local_memory = m_client->localMalloc(m_memory_per_thread);
	m_local_memory->openContext();
	m_local_memory->setMemory(1);
}

rdma::AtomicsOperationsCountPerfClientThread::~AtomicsOperationsCountPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_memory_per_thread, m_remOffsets[i]);
	}
    delete m_remOffsets;
	delete m_local_memory; // implicitly deletes local allocs in RDMAClient
	delete m_client;
}

void rdma::AtomicsOperationsCountPerfClientThread::run() {
	unique_lock<mutex> lck(AtomicsOperationsCountPerfTest::waitLock);
	if (!AtomicsOperationsCountPerfTest::signaled) {
		m_ready = true;
		AtomicsOperationsCountPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	m_ready = false;

	auto start = rdma::PerfTest::startTimer();
	switch(AtomicsOperationsCountPerfTest::testOperation){
		case FETCH_ADD_OPERATION: // Fetch & Add
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
					bool signaled = (i == (m_iterations_per_thread - 1) || (i+1)%Config::RDMA_MAX_WR==0);
					int offset = (i % m_buffer_slots) * rdma::ATOMICS_SIZE;
					m_client->fetchAndAdd(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), 1, rdma::ATOMICS_SIZE, signaled); // true=signaled
				}
			}
			m_elapsedFetchAdd = rdma::PerfTest::stopTimer(start);
			break;
		case COMPARE_SWAP_OPERATION: // Compare & Swap
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
					bool signaled = (i == (m_iterations_per_thread - 1) || (i+1)%Config::RDMA_MAX_WR==0);
					int offset = (i % m_buffer_slots) * rdma::ATOMICS_SIZE;
					m_client->compareAndSwap(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), 2, 3, rdma::ATOMICS_SIZE, signaled); // true=signaled
				}
			}
			m_elapsedCompareSwap = rdma::PerfTest::stopTimer(start);
			break;
		default: throw invalid_argument("BandwidthPerfClientThread unknown test mode");
	}
}




rdma::AtomicsOperationsCountPerfTest::AtomicsOperationsCountPerfTest(int testOperations, bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int thread_count, int buffer_slots, uint64_t iterations_per_thread) : PerfTest(testOperations){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_ownIpPort = ownIpPort;
	this->m_sequencerIpPort = sequencerIpPort;
	this->m_local_gpu_index = local_gpu_index;
	this->m_actual_gpu_index = -1;
	this->m_remote_gpu_index = remote_gpu_index;
	this->m_thread_count = thread_count;
	this->m_memory_size = thread_count * rdma::ATOMICS_SIZE * buffer_slots;
	this->m_buffer_slots = buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_rdma_addresses = rdma_addresses;
}
rdma::AtomicsOperationsCountPerfTest::~AtomicsOperationsCountPerfTest(){
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		delete m_client_threads[i];
	}
	m_client_threads.clear();
	if(m_is_server)
		delete m_server;
	delete m_memory;
}

std::string rdma::AtomicsOperationsCountPerfTest::getTestParameters(bool forCSV){
	std::ostringstream oss;
	oss << (m_is_server ? "Server" : "Client") << ", threads=" << m_thread_count << ", bufferslots=" << m_buffer_slots << ", packetsize=" << rdma::ATOMICS_SIZE << ", memory=";
	oss << m_memory_size << " (" << m_thread_count << "x " << m_buffer_slots << "x " << rdma::ATOMICS_SIZE << ")";
	oss << ", memory_type=" << getMemoryName(m_local_gpu_index, m_actual_gpu_index) << (m_remote_gpu_index!=-404 ? "->"+getMemoryName(m_remote_gpu_index) : "");
	if(!forCSV){ oss << ", iterations=" << (m_iterations_per_thread*m_thread_count); }
	return oss.str();
}
std::string rdma::AtomicsOperationsCountPerfTest::getTestParameters(){
	return getTestParameters(false);
}

void rdma::AtomicsOperationsCountPerfTest::makeThreadsReady(TestOperation testOperation){
	AtomicsOperationsCountPerfTest::testOperation = testOperation;
	AtomicsOperationsCountPerfTest::signaled = false;
	for(AtomicsOperationsCountPerfClientThread* perfThread : m_client_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
}

void rdma::AtomicsOperationsCountPerfTest::runThreads(){
	AtomicsOperationsCountPerfTest::signaled = false;
	unique_lock<mutex> lck(AtomicsOperationsCountPerfTest::waitLock);
	AtomicsOperationsCountPerfTest::waitCv.notify_all();
	AtomicsOperationsCountPerfTest::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		m_client_threads[i]->join();
	}
}

void rdma::AtomicsOperationsCountPerfTest::setupTest(){
	m_elapsedFetchAdd = -1;
	m_elapsedCompareSwap = -1;
	m_actual_gpu_index = -1;
	#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
		if(m_local_gpu_index <= -3){
			m_memory = new rdma::MainMemory(m_memory_size);
		} else {
			rdma::CudaMemory *mem = new rdma::CudaMemory(m_memory_size, m_local_gpu_index);
			m_memory = mem;
			m_actual_gpu_index = mem->getDeviceIndex();
		}
	#else
		m_memory = (rdma::BaseMemory*)new MainMemory(m_memory_size);
	#endif

	if(m_is_server){
		// Server
		m_server = new RDMAServer<ReliableRDMA>("AtomicsOperationsCountTestRDMAServer", m_rdma_port, Network::getAddressOfConnection(m_ownIpPort), m_memory, m_sequencerIpPort);

	} else {
		// Client
		for (int i = 0; i < m_thread_count; i++) {
			AtomicsOperationsCountPerfClientThread* perfThread = new AtomicsOperationsCountPerfClientThread(m_memory, m_rdma_addresses, m_ownIpPort, m_sequencerIpPort, m_buffer_slots, m_iterations_per_thread);
			m_client_threads.push_back(perfThread);
		}
	}
}

void rdma::AtomicsOperationsCountPerfTest::runTest(){
	if(m_is_server){
		// Server
		std::cout << "Starting server on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "' . . ." << std::endl;
		if(!m_server->startServer()){
			std::cerr << "AtomicsOperationsCountPerfTest::runTest(): Could not start server" << std::endl;
			throw invalid_argument("AtomicsOperationsCountPerfTest server startup failed");
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

		// Measure operations/s for fetching & adding
		if(hasTestOperation(FETCH_ADD_OPERATION)){
			makeThreadsReady(FETCH_ADD_OPERATION); // fetch & add
			auto startFetchAdd = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedFetchAdd = rdma::PerfTest::stopTimer(startFetchAdd); 
		}

		// Measure operations/s for comparing & swaping
		if(hasTestOperation(COMPARE_SWAP_OPERATION)){
			makeThreadsReady(COMPARE_SWAP_OPERATION); // compare & swap
			auto startCompareSwap = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedCompareSwap = rdma::PerfTest::stopTimer(startCompareSwap);
		}
	}
}


std::string rdma::AtomicsOperationsCountPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
	if(m_is_server){
		return "only client";
	} else {
		
		/*	There are  n  threads
			Each thread computes  iterations_per_thread = total_iterations / n
			Each thread takes  n  times more time compared to a single thread
			Operations/sec	= iterations / elapsedTime
							= (n * iterations_per_thread) / (elpasedTime_per_thread)
		*/

		const long double tu = (long double)NANO_SEC; // 1sec (nano to seconds as time unit)
        const uint64_t iters = m_iterations_per_thread * m_thread_count;

		int64_t maxFetchAdd=-1, minFetchAdd=std::numeric_limits<int64_t>::max();
		int64_t maxCompareSwap=-1, minCompareSwap=std::numeric_limits<int64_t>::max();
		int64_t arrFetchAdd[m_thread_count];
		int64_t arrCompareSwap[m_thread_count];
		long double avgFetchAdd=0, medianFetchAdd, avgCompareSwap=0, medianCompareSwap;
		const long double div = 1; // TODO not sure why additional   m_thread_count  is too much
		const long double divAvg = m_client_threads.size() * div;
		for(size_t i=0; i<m_client_threads.size(); i++){
			AtomicsOperationsCountPerfClientThread *thr = m_client_threads[i];
			if(thr->m_elapsedFetchAdd < minFetchAdd) minFetchAdd = thr->m_elapsedFetchAdd;
			if(thr->m_elapsedFetchAdd > maxFetchAdd) maxFetchAdd = thr->m_elapsedFetchAdd;
			avgFetchAdd += (long double) thr->m_elapsedFetchAdd / divAvg;
			arrFetchAdd[i] = thr->m_elapsedFetchAdd;
			if(thr->m_elapsedCompareSwap < minCompareSwap) minCompareSwap = thr->m_elapsedCompareSwap;
			if(thr->m_elapsedCompareSwap > maxCompareSwap) maxCompareSwap = thr->m_elapsedCompareSwap;
			avgCompareSwap += (long double) thr->m_elapsedCompareSwap / divAvg;
			arrCompareSwap[i] = thr->m_elapsedCompareSwap;
		}
		minFetchAdd /= div; maxFetchAdd /= div;
		minCompareSwap /= div; maxCompareSwap /= div;

		std::sort(arrFetchAdd, arrFetchAdd + m_thread_count);
		std::sort(arrCompareSwap, arrCompareSwap + m_thread_count);
		medianFetchAdd = arrFetchAdd[(int)(m_thread_count/2)] / div;
		medianCompareSwap = arrCompareSwap[(int)(m_thread_count/2)] / div;

		// write results into CSV file
		if(!csvFileName.empty()){
			const long double su = 1000*1000; // size unit (bytes to MegaBytes) | use 1024*1024 for MebiBytes
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			ofs << rdma::CSV_PRINT_NOTATION << rdma::CSV_PRINT_PRECISION;
			if(csvAddHeader){
				ofs << std::endl << "ATOMICS OPERATIONS COUNT, " << getTestParameters(true) << std::endl;
				ofs << "Iterations";
				if(hasTestOperation(FETCH_ADD_OPERATION)){
					ofs << ", Fetch&Add [megaOp/s], Avg Fetch&Add [megaOp/s], Median Fetch&Add [megaOp/s], Min Fetch&Add [megaOp/s], Max Fetch&Add [megaOp/s], ";
					ofs << "Fetch&Add [Sec], Avg Fetch&Add [Sec], Median Fetch&Add [Sec], Min Fetch&Add [Sec], Max Fetch&Add [Sec]";
				}
				if(hasTestOperation(COMPARE_SWAP_OPERATION)){
					ofs << ", Comp&Swap [megaOp/s], Avg Comp&Swap [megaOp/s], Median Comp&Swap [megaOp/s], Min Comp&Swap [megaOp/s], Max Comp&Swap [megaOp/s], ";
					ofs << "Comp&Swap [Sec], Avg Comp&Swap [Sec], Median Comp&Swap [Sec], Min Comp&Swap [Sec], Max Comp&Swap [Sec]";
				}
				ofs << std::endl;
			}
			ofs << iters;
			if(hasTestOperation(FETCH_ADD_OPERATION)){
				ofs << ", " << (round(iters*tu/su/m_elapsedFetchAdd * 100000)/100000.0) << ", "; // fetch&add Op/s
				ofs << (round(iters*tu/su/maxFetchAdd * 100000)/100000.0) << ", "; // min fetch&add Op/s
				ofs << (round(iters*tu/su/minFetchAdd * 100000)/100000.0) << ", "; // max fetch&add Op/s
				ofs << (round(iters*tu/su/avgFetchAdd * 100000)/100000.0) << ", "; // avg fetch&add Op/s
				ofs << (round(iters*tu/su/medianFetchAdd * 100000)/100000.0) << ", "; // median fetch&add Op/s
				ofs << (round(m_elapsedFetchAdd/tu * 100000)/100000.0) << ", "; // fetch&add Op/s
				ofs << (round(minFetchAdd/tu * 100000)/100000.0) << ", "; // min fetch&add Op/s
				ofs << (round(maxFetchAdd/tu * 100000)/100000.0) << ", "; // max fetch&add Op/s
				ofs << (round(avgFetchAdd/tu * 100000)/100000.0) << ", "; // avg fetch&add Op/s
				ofs << (round(medianFetchAdd/tu * 100000)/100000.0); // median fetch&add Op/s
			}
			if(hasTestOperation(COMPARE_SWAP_OPERATION)){
				ofs << ", " << (round(iters*tu/su/m_elapsedCompareSwap * 100000)/100000.0) << ", "; // comp&swap Op/s
				ofs << (round(iters*tu/su/avgCompareSwap * 100000)/100000.0) << ", "; // avg comp&swap Op/s
				ofs << (round(iters*tu/su/medianCompareSwap * 100000)/100000.0) << ", "; // median comp&swap Op/s
				ofs << (round(iters*tu/su/maxCompareSwap * 100000)/100000.0) << ", "; // min comp&swap Op/s
				ofs << (round(iters*tu/su/minCompareSwap * 100000)/100000.0) << ", "; // max comp&swap Op/s
				ofs << (round(m_elapsedCompareSwap/tu * 100000)/100000.0) << ", "; // comp&swap Sec
				ofs << (round(avgCompareSwap/tu * 100000)/100000.0) << ", "; // avg comp&swap Sec
				ofs << (round(medianCompareSwap/tu * 100000)/100000.0) << ", "; // median fetch&add, comp&swap Sec
				ofs << (round(minCompareSwap/tu * 100000)/100000.0) << ", "; // min comp&swap Sec
				ofs << (round(maxCompareSwap/tu * 100000)/100000.0); // max comp&swap Sec
			}
			ofs << std::endl; ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		oss << rdma::CONSOLE_PRINT_NOTATION << rdma::CONSOLE_PRINT_PRECISION;
		oss << " measurements are executed as bursts with " << Config::RDMA_MAX_WR << " operations per burst" << std::endl;
		oss << std::endl << " - Fetch&Add:     operations = " << rdma::PerfTest::convertCountPerSec(iters*tu/m_elapsedFetchAdd);
		oss << "  (range = " << rdma::PerfTest::convertCountPerSec(iters*tu/maxFetchAdd) << " - ";
		oss << rdma::PerfTest::convertCountPerSec(iters*tu/minFetchAdd);
		oss << " ; avg=" << rdma::PerfTest::convertCountPerSec(iters*tu/avgFetchAdd) << " ; median=";
		oss << rdma::PerfTest::convertCountPerSec(iters*tu/minFetchAdd) << ")";
		oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedFetchAdd) << "  (range=";
		oss << rdma::PerfTest::convertTime(minFetchAdd) << "-" << rdma::PerfTest::convertTime(maxFetchAdd);
		oss << " ; avg=" << rdma::PerfTest::convertTime(avgFetchAdd) << " ; median=" << rdma::PerfTest::convertTime(medianFetchAdd) << ")" << std::endl;
		oss << " - Compare&Swap:  operations = " << rdma::PerfTest::convertCountPerSec(iters*tu/m_elapsedCompareSwap);
		oss << "  (range = " << rdma::PerfTest::convertCountPerSec(iters*tu/maxCompareSwap) << " - ";
		oss << rdma::PerfTest::convertCountPerSec(iters*tu/minCompareSwap);
		oss << " ; avg=" << rdma::PerfTest::convertCountPerSec(iters*tu/avgCompareSwap) << " ; median=";
		oss << rdma::PerfTest::convertCountPerSec(iters*tu/minCompareSwap) << ")";
		oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedCompareSwap) << "  (range=";
		oss << rdma::PerfTest::convertTime(minCompareSwap) << "-" << rdma::PerfTest::convertTime(maxCompareSwap);
		oss << " ; avg=" << rdma::PerfTest::convertTime(avgCompareSwap) << " ; median=" << rdma::PerfTest::convertTime(medianCompareSwap) << ")" << std::endl;
		return oss.str();

	}
	return NULL;
}