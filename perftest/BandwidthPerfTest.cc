#include "BandwidthPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"

#include <limits>
#include <algorithm>

mutex rdma::BandwidthPerfTest::waitLock;
condition_variable rdma::BandwidthPerfTest::waitCv;
bool rdma::BandwidthPerfTest::signaled;
rdma::TestOperation rdma::BandwidthPerfTest::testOperation;

rdma::BandwidthPerfClientThread::BandwidthPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, size_t packet_size, int buffer_slots, size_t iterations_per_thread, size_t max_rdma_wr_per_thread, WriteMode write_mode) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "BandwidthPerfTestClient", ownIpPort, sequencerIpPort);
	this->m_rdma_addresses = rdma_addresses;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size_per_thread = packet_size * buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_max_rdma_wr_per_thread = max_rdma_wr_per_thread;
	this->m_write_mode = write_mode;
	m_remOffsets = new size_t[m_rdma_addresses.size()];

	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_rdma_addresses[i];
		//std::cout << "Thread trying to connect to '" << conn << "' . . ." << std::endl; // TODO REMOVE
		if(!m_client->connect(conn, nodeId)) {
			std::cerr << "BandwidthPerfThread::BandwidthPerfThread(): Could not connect to '" << conn << "'" << std::endl;
			throw invalid_argument("BandwidthPerfThread connection failed");
		}
		//std::cout << "Thread connected to '" << conn << "'" << std::endl; // TODO REMOVE
		m_addr.push_back(nodeId);
		m_client->remoteAlloc(conn, m_memory_size_per_thread, m_remOffsets[i]);
	}

	m_local_memory = m_client->localMalloc(m_memory_size_per_thread);
	m_local_memory->openContext();
	m_local_memory->setMemory(1);
}


rdma::BandwidthPerfClientThread::~BandwidthPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_memory_size_per_thread, m_remOffsets[i]);
	}
    delete m_remOffsets;
	delete m_local_memory; // implicitly deletes local allocs in RDMAClient and also closes context
	delete m_client;
}


void rdma::BandwidthPerfClientThread::run() {
	unique_lock<mutex> lck(BandwidthPerfTest::waitLock);
	if (!BandwidthPerfTest::signaled) {
		m_ready = true;
		BandwidthPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	int sendCounter = 0, receiveCounter = 0, totalBudget = m_iterations_per_thread;
	uint32_t localBaseOffset = (uint32_t)m_local_memory->getRootOffset();
	auto start = rdma::PerfTest::startTimer();
	switch(BandwidthPerfTest::testOperation){
		case WRITE_OPERATION: // Write
			switch(m_write_mode){
				case WRITE_MODE_NORMAL:
					for(size_t i = 0; i < m_iterations_per_thread; i++){
						size_t connIdx = i % m_rdma_addresses.size();
						bool signaled = (i == (m_iterations_per_thread - 1));
						int offset = (i % m_buffer_slots) * m_packet_size;
						m_client->write(m_addr[connIdx], m_remOffsets[connIdx]+offset, m_local_memory->pointer(offset), m_packet_size, signaled);
					}
					break;
				case WRITE_MODE_IMMEDIATE:
					for(size_t i = 0; i < m_iterations_per_thread; i+=2*m_max_rdma_wr_per_thread){
						int budgetS = totalBudget;
						if(budgetS > (int)m_max_rdma_wr_per_thread){ budgetS = m_max_rdma_wr_per_thread; }
						totalBudget -= budgetS;

						size_t fi = i + budgetS;
						int budgetR = totalBudget;
						if(budgetR >(int)m_max_rdma_wr_per_thread){ budgetR = m_max_rdma_wr_per_thread; }
						totalBudget -= budgetR;

						for(int j = 0; j < budgetR; j++){
							m_client->receiveWriteImm(m_addr[(fi+j) % m_rdma_addresses.size()]);
						}
						for(int j = 0; j < budgetS; j++){
							size_t connIdx = i % m_rdma_addresses.size();
							int offset = (i % m_buffer_slots) * m_packet_size;
							m_client->writeImm(m_addr[(i+j) % m_rdma_addresses.size()], m_remOffsets[connIdx]+offset, m_local_memory->pointer(offset), m_packet_size, localBaseOffset, (j+1)==budgetS);
						}

						for(int j = 0; j < budgetR; j++){
							m_client->pollReceive(m_addr[(fi+j) % m_rdma_addresses.size()], true);
						}
					}
					break;
				default: throw invalid_argument("BandwidthPerfClientThread unknown write mode");
			}
			m_elapsedWriteMs = rdma::PerfTest::stopTimer(start);
			break;

		case READ_OPERATION: // Read
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations_per_thread - 1));
				int offset = (i % m_buffer_slots) * m_packet_size;
				m_client->read(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), m_packet_size, signaled);
			}
			m_elapsedReadMs = rdma::PerfTest::stopTimer(start);
			break;

		case SEND_RECEIVE_OPERATION: // Send & Receive
			// alternating send/receive blocks to not overfill queues
			for(size_t i = 0; i < m_iterations_per_thread; i+=2*m_max_rdma_wr_per_thread){
				int budgetS = m_iterations_per_thread - i;
				if(budgetS > (int)m_max_rdma_wr_per_thread){ budgetS = m_max_rdma_wr_per_thread; }
				totalBudget -= budgetS;

				size_t fi = i + budgetS;
				int budgetR = totalBudget;
				if(budgetR >(int)m_max_rdma_wr_per_thread){ budgetR = m_max_rdma_wr_per_thread; }
				totalBudget -= budgetR;

				for(int j = 0; j < budgetR; j++){
					receiveCounter = (receiveCounter+1) % m_buffer_slots;
					int receiveOffset = receiveCounter * m_packet_size;
					m_client->receive(m_addr[(fi+j) % m_rdma_addresses.size()], m_local_memory->pointer(receiveOffset), m_packet_size);
				}

				for(int j = 0; j < budgetS; j++){
					sendCounter = (sendCounter+1) % m_buffer_slots;
					int sendOffset = sendCounter * m_packet_size + m_memory_size_per_thread;
					m_client->send(m_addr[(i+j) % m_rdma_addresses.size()], m_local_memory->pointer(sendOffset), m_packet_size, (j+1)==budgetS); // signaled: (j+1)==budget
				}

				for(int j = 0; j < budgetR; j++){
					m_client->pollReceive(m_addr[(fi+j) % m_rdma_addresses.size()], true);
				}
			}
			m_elapsedSendMs = rdma::PerfTest::stopTimer(start);
			break;
		default: throw invalid_argument("BandwidthPerfClientThread unknown test mode");
	}
}



rdma::BandwidthPerfServerThread::BandwidthPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t packet_size, int buffer_slots, size_t iterations_per_thread, size_t max_rdma_wr_per_thread, WriteMode write_mode, int thread_id) {
	this->m_server = server;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size_per_thread = packet_size * buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_max_rdma_wr_per_thread = max_rdma_wr_per_thread;
	this->m_write_mode = write_mode;
	this->m_thread_id = thread_id;
	this->m_local_memory = server->localMalloc(this->m_memory_size_per_thread);
	this->m_local_memory->openContext();
}

rdma::BandwidthPerfServerThread::~BandwidthPerfServerThread() {
	delete m_local_memory;  // implicitly deletes local allocs in RDMAServer
}

void rdma::BandwidthPerfServerThread::run() {
	unique_lock<mutex> lck(BandwidthPerfTest::waitLock);
	if (!BandwidthPerfTest::signaled) {
		m_ready = true;
		BandwidthPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	const std::vector<size_t> clientIds = m_server->getConnectedConnIDs();

	size_t i = 0;
	int sendCounter = 0, receiveCounter = 0, totalBudget = m_iterations_per_thread, budgetR = totalBudget, budgetS;
	uint32_t remoteBaseOffset;

	switch(BandwidthPerfTest::testOperation){
		case WRITE_OPERATION:
			if(m_write_mode == WRITE_MODE_IMMEDIATE){
				// Calculate receive budget for even blocks
				if(budgetR > (int)m_max_rdma_wr_per_thread){ budgetR = m_max_rdma_wr_per_thread; }
				totalBudget -= budgetR;
				i += budgetR;
				for(int j = 0; j < budgetR; j++){
					m_server->receiveWriteImm(clientIds[m_thread_id]);
				}

				do {
					// Calculate send budget for odd blocks
					budgetS = totalBudget;
					if(budgetS > (int)m_max_rdma_wr_per_thread){ budgetS = m_max_rdma_wr_per_thread; }
					totalBudget -= budgetS;
					for(int j = 0; j < budgetR; j++){
						m_server->pollReceive(clientIds[m_thread_id], true, &remoteBaseOffset);
					}

					// Calculate receive budget for even blocks
					budgetR = totalBudget; 
					if(budgetR > (int)m_max_rdma_wr_per_thread){ budgetR = m_max_rdma_wr_per_thread; }
					totalBudget -= budgetR;

					for(int j = 0; j < budgetR; j++){
						m_server->receiveWriteImm(clientIds[m_thread_id]);
					}
					for(int j = 0; j < budgetS; j++){
						sendCounter = (sendCounter+1) % m_buffer_slots;
						int sendOffset = sendCounter * m_packet_size + m_memory_size_per_thread;
						uint32_t remoteOffset = remoteBaseOffset + sendCounter * m_packet_size; 
						m_server->writeImm(clientIds[m_thread_id], remoteOffset, m_local_memory->pointer(sendOffset), m_packet_size, (uint32_t)(i+j), (j+1)==budgetS); // signaled: (j+1)==budget
					}

					i += 2 * m_max_rdma_wr_per_thread;
				} while(i < m_iterations_per_thread);

				for(int j = 0; j < budgetR; j++){ // final poll to sync up with client again
					m_server->pollReceive(clientIds[m_thread_id], true, &remoteBaseOffset);
				}
			}
			break;

		case SEND_RECEIVE_OPERATION:
			// Calculate receive budget for even blocks
			if(budgetR > (int)m_max_rdma_wr_per_thread){ budgetR = m_max_rdma_wr_per_thread; }
			totalBudget -= budgetR;
			i += budgetR;
			for(int j = 0; j < budgetR; j++){
				receiveCounter = (receiveCounter+1) % m_buffer_slots;
				int receiveOffset = receiveCounter * m_packet_size;
				m_server->receive(clientIds[m_thread_id], m_local_memory->pointer(receiveOffset), m_packet_size);
			}

			do {
				// Calculate send budget for odd blocks
				budgetS = totalBudget;
				if(budgetS > (int)m_max_rdma_wr_per_thread){ budgetS = m_max_rdma_wr_per_thread; }
				totalBudget -= budgetS;

				for(int j = 0; j < budgetR; j++){
					m_server->pollReceive(clientIds[m_thread_id], true);
				}

				// Calculate receive budget for even blocks
				budgetR = totalBudget; 
				if(budgetR > (int)m_max_rdma_wr_per_thread){ budgetR = m_max_rdma_wr_per_thread; }
				totalBudget -= budgetR;
				for(int j = 0; j < budgetR; j++){
					receiveCounter = (receiveCounter+1) % m_buffer_slots;
					int receiveOffset = receiveCounter * m_packet_size;
					m_server->receive(clientIds[m_thread_id], m_local_memory->pointer(receiveOffset), m_packet_size);
				}

				for(int j = 0; j < budgetS; j++){
					sendCounter = (sendCounter+1) % m_buffer_slots;
					int sendOffset = sendCounter * m_packet_size + m_memory_size_per_thread;
					m_server->send(clientIds[m_thread_id], m_local_memory->pointer(sendOffset), m_packet_size, (j+1)==budgetS); // signaled: (j+1)==budget
				}

				i += 2 * m_max_rdma_wr_per_thread;
			} while(i < m_iterations_per_thread);
			
			for(int j = 0; j < budgetR; j++){ // Finaly poll to sync up with client again
				m_server->pollReceive(clientIds[m_thread_id], true);
			}
			break;

		default: break;
	}
	//m_elapsedReceive = rdma::PerfTest::stopTimer(start);
}



rdma::BandwidthPerfTest::BandwidthPerfTest(int testOperations, bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int thread_count, uint64_t packet_size, int buffer_slots, uint64_t iterations_per_thread, WriteMode write_mode) : PerfTest(testOperations){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_ownIpPort = ownIpPort;
	this->m_sequencerIpPort = sequencerIpPort;
	this->m_local_gpu_index = local_gpu_index;
	this->m_actual_gpu_index = -1;
	this->m_remote_gpu_index = remote_gpu_index;
	this->m_thread_count = thread_count;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size = 2 * thread_count * packet_size * buffer_slots; // 2x because for send & receive separat
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_write_mode = (write_mode!=WRITE_MODE_AUTO ? write_mode : rdma::BandwidthPerfTest::DEFAULT_WRITE_MODE);
	this->m_rdma_addresses = rdma_addresses;
}
rdma::BandwidthPerfTest::~BandwidthPerfTest(){
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
	delete m_memory;
}

std::string rdma::BandwidthPerfTest::getTestParameters(bool forCSV){
	std::ostringstream oss;
	oss << (m_is_server ? "Server" : "Client") << ", threads=" << m_thread_count << ", bufferslots=" << m_buffer_slots;
	if(!forCSV){ oss << ", packetsize=" << m_packet_size; }
	oss << ", memory=" << m_memory_size << " (2x " << m_thread_count << "x " << m_buffer_slots << "x " << m_packet_size << ")";
	oss << ", memory_type=" << getMemoryName(m_local_gpu_index, m_actual_gpu_index) << (m_remote_gpu_index!=-404 ? "->"+getMemoryName(m_remote_gpu_index) : "");
	oss << ", iterations=" << (m_iterations_per_thread*m_thread_count) << ", writemode=" << (m_write_mode==WRITE_MODE_NORMAL ? "Normal" : "Immediate");
	return oss.str();
}
std::string rdma::BandwidthPerfTest::getTestParameters(){
	return getTestParameters(false);
}

void rdma::BandwidthPerfTest::makeThreadsReady(TestOperation testOperation){
	BandwidthPerfTest::testOperation = testOperation;
	BandwidthPerfTest::signaled = false;

	for(BandwidthPerfServerThread* perfThread : m_server_threads){ perfThread->start(); }
	for(BandwidthPerfServerThread* perfThread : m_server_threads){ while(!perfThread->ready()) usleep(Config::RDMA_SLEEP_INTERVAL); }

	for(BandwidthPerfClientThread* perfThread : m_client_threads){
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
	for (size_t i = 0; i < m_server_threads.size(); i++) {
		m_server_threads[i]->join();
	}
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		m_client_threads[i]->join();
	}
}

void rdma::BandwidthPerfTest::setupTest(){
	m_elapsedWriteMs = -1;
	m_elapsedReadMs = -1;
	m_elapsedSendMs = -1;
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

	const size_t max_rdma_wr_per_thread = rdma::Config::RDMA_MAX_WR;

	if(m_is_server){
		// Server
		m_server = new RDMAServer<ReliableRDMA>("BandwidthTestRDMAServer", m_rdma_port, Network::getAddressOfConnection(m_ownIpPort), m_memory, m_sequencerIpPort);
		for (int thread_id = 0; thread_id < m_thread_count; thread_id++) {
			BandwidthPerfServerThread* perfThread = new BandwidthPerfServerThread(m_server, m_packet_size, m_buffer_slots, m_iterations_per_thread, max_rdma_wr_per_thread, m_write_mode, thread_id);
			m_server_threads.push_back(perfThread);
		}

	} else {
		// Client
		for (int i = 0; i < m_thread_count; i++) {
			BandwidthPerfClientThread* perfThread = new BandwidthPerfClientThread(m_memory, m_rdma_addresses, m_ownIpPort, m_sequencerIpPort, m_packet_size, m_buffer_slots, m_iterations_per_thread, max_rdma_wr_per_thread, m_write_mode);
			m_client_threads.push_back(perfThread);
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

		// waiting until clients have connected to sync
		while(m_server->getConnectedConnIDs().size() < (size_t)m_thread_count) usleep(Config::RDMA_SLEEP_INTERVAL);
		
		if(hasTestOperation(WRITE_OPERATION)){
			makeThreadsReady(WRITE_OPERATION); // receive
			//auto startReceive = rdma::PerfTest::startTimer();
			runThreads();
			//m_elapsedReceiveMs = rdma::PerfTest::stopTimer(startReceive);
		}
		
		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			makeThreadsReady(SEND_RECEIVE_OPERATION); // receive
			//auto startReceive = rdma::PerfTest::startTimer();
			runThreads();
			//m_elapsedReceiveMs = rdma::PerfTest::stopTimer(startReceive);
		}

		// wait until server is done
		while (m_server->isRunning() && m_server->getConnectedConnIDs().size() > 0) usleep(Config::RDMA_SLEEP_INTERVAL);
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client


        // Measure bandwidth for writing
		if(hasTestOperation(WRITE_OPERATION)){
			makeThreadsReady(WRITE_OPERATION); // write
			usleep(4 * Config::RDMA_SLEEP_INTERVAL); // let server first post the receives
			auto startWrite = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedWriteMs = rdma::PerfTest::stopTimer(startWrite);
		}

		// Measure bandwidth for reading
		if(hasTestOperation(READ_OPERATION)){
			makeThreadsReady(READ_OPERATION); // read
			auto startRead = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedReadMs = rdma::PerfTest::stopTimer(startRead);
		}

		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			// Measure bandwidth for sending
			makeThreadsReady(SEND_RECEIVE_OPERATION); // send
			usleep(4 * Config::RDMA_SLEEP_INTERVAL); // let server first post the receives
			auto startSend = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedSendMs = rdma::PerfTest::stopTimer(startSend);
		}
	}
}


std::string rdma::BandwidthPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
	if(m_is_server){
		return "only client";
	} else {
		
		/*	There are  n  threads
			Each thread computes  iterations_per_thread = total_iterations / n
			Each thread takes  n  times more time compared to a single thread
			 Bandwidth 	= transferedBytes / elapsedTime
						= (n * iterations_per_thread * packetSize) / (elapsedTime_per_thread)
		*/

		const long double tu = (long double)NANO_SEC; // 1sec (nano to seconds as time unit)
		
		uint64_t transferedBytes = m_thread_count * m_iterations_per_thread * m_packet_size;
		int64_t maxWriteMs=-1, minWriteMs=std::numeric_limits<int64_t>::max();
		int64_t maxReadMs=-1, minReadMs=std::numeric_limits<int64_t>::max();
		int64_t maxSendMs=-1, minSendMs=std::numeric_limits<int64_t>::max();
		int64_t arrWriteMs[m_thread_count];
		int64_t arrReadMs[m_thread_count];
		int64_t arrSendMs[m_thread_count];
		long double avgWriteMs=0, medianWriteMs, avgReadMs=0, medianReadMs, avgSendMs=0, medianSendMs;
		const long double divAvg = m_client_threads.size();
		for(size_t i=0; i<m_client_threads.size(); i++){
			BandwidthPerfClientThread *thr = m_client_threads[i];
			if(thr->m_elapsedWriteMs < minWriteMs) minWriteMs = thr->m_elapsedWriteMs;
			if(thr->m_elapsedWriteMs > maxWriteMs) maxWriteMs = thr->m_elapsedWriteMs;
			avgWriteMs += (long double) thr->m_elapsedWriteMs / divAvg;
			arrWriteMs[i] = thr->m_elapsedWriteMs;
			if(thr->m_elapsedReadMs < minReadMs) minReadMs = thr->m_elapsedReadMs;
			if(thr->m_elapsedReadMs > maxReadMs) maxReadMs = thr->m_elapsedReadMs;
			avgReadMs += (long double) thr->m_elapsedReadMs / divAvg;
			arrReadMs[i] = thr->m_elapsedReadMs;
			if(thr->m_elapsedSendMs < minSendMs) minSendMs = thr->m_elapsedSendMs;
			if(thr->m_elapsedSendMs > maxSendMs) maxSendMs = thr->m_elapsedSendMs;
			avgSendMs += (long double) thr->m_elapsedSendMs / divAvg;
			arrSendMs[i] = thr->m_elapsedSendMs;
		}

		std::sort(arrWriteMs, arrWriteMs + m_thread_count);
		std::sort(arrReadMs, arrReadMs + m_thread_count);
		std::sort(arrSendMs, arrSendMs + m_thread_count);

		medianWriteMs = arrWriteMs[(int)(m_thread_count/2)];
		medianReadMs = arrReadMs[(int)(m_thread_count/2)];
		medianSendMs = arrSendMs[(int)(m_thread_count/2)];

		// write results into CSV file
		if(!csvFileName.empty()){
			const long double su = 1000*1000*1000; // size unit (bytes to GigaBytes) | use 1024*1024 for MebiBytes
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			ofs << rdma::CSV_PRINT_NOTATION << rdma::CSV_PRINT_PRECISION;
			if(csvAddHeader){
				ofs << std::endl << "BANDWIDTH, " << getTestParameters(true) << std::endl;
				ofs << "PacketSize [Bytes], Transfered [Bytes]";
				if(hasTestOperation(WRITE_OPERATION)){
					ofs << ", Write [GB/s], Min Write [GB/s],Max Write [GB/s], Avg Write [GB/s], Median Write [GB/s], ";
					ofs << "Write [Sec], Min Write [Sec], Max Write [Sec], Avg Write [Sec], Median Write [Sec]";
				}
				if(hasTestOperation(READ_OPERATION)){
					ofs << ", Read [GB/s], Min Read [GB/s], Max Read [GB/s], Avg Read [GB/s], Median Read [GB/s], ";
					ofs << "Read [Sec], Min Read [Sec], Max Read [Sec], Avg Read [Sec], Median Read [Sec]";
				}
				if(hasTestOperation(SEND_RECEIVE_OPERATION)){
					ofs << ", Send/Recv [GB/s], Min Send/Recv [GB/s], Max Send/Recv [GB/s], Avg Send/Recv [GB/s], Median Send/Recv [GB/s], ";
					ofs << "Send/Recv [Sec], Min Send/Recv [Sec], Max Send/Recv [Sec], Avg Send/Recv [Sec], Median Send/Recv [Sec]";
				}
				ofs << std::endl;
			}
			ofs << m_packet_size << ", " << transferedBytes; // packet size Bytes
			if(hasTestOperation(WRITE_OPERATION)){
				ofs << ", " << (round(transferedBytes*tu/su/m_elapsedWriteMs * 100000)/100000.0) << ", "; // write GB/s
				ofs << (round(transferedBytes*tu/su/maxWriteMs * 100000)/100000.0) << ", "; // min write GB/s
				ofs << (round(transferedBytes*tu/su/minWriteMs * 100000)/100000.0) << ", "; // max write GB/s
				ofs << (round(transferedBytes*tu/su/avgWriteMs * 100000)/100000.0) << ", "; // avg write GB/s
				ofs << (round(transferedBytes*tu/su/medianWriteMs * 100000)/100000.0) << ", "; // median write GB/s
				ofs << (round(m_elapsedWriteMs/tu * 100000)/100000.0) << ", " ; // write Sec
				ofs << (round(minWriteMs/tu * 100000)/100000.0) << ", "; // min write Sec
				ofs << (round(maxWriteMs/tu * 100000)/100000.0) << ", "; // max write Sec
				ofs << (round(avgWriteMs/tu * 100000)/100000.0) << ", "; // avg write Sec
				ofs << (round(medianWriteMs/tu * 100000)/100000.0); // median write Sec
			}
			if(hasTestOperation(READ_OPERATION)){
				ofs << ", " << (round(transferedBytes*tu/su/m_elapsedReadMs * 100000)/100000.0) << ", "; // read GB/s
				ofs << (round(transferedBytes*tu/su/maxReadMs * 100000)/100000.0) << ", "; // min read GB/s
				ofs << (round(transferedBytes*tu/su/minReadMs * 100000)/100000.0) << ", "; // max read GB/s
				ofs << (round(transferedBytes*tu/su/avgReadMs * 100000)/100000.0) << ", "; // avg read GB/s
				ofs << (round(transferedBytes*tu/su/medianReadMs * 100000)/100000.0) << ", "; // median read GB/s
				ofs << (round(m_elapsedReadMs/tu * 100000)/100000.0) << ", "; // read Sec
				ofs << (round(minReadMs/tu * 100000)/100000.0) << ", "; // min read Sec
				ofs << (round(maxReadMs/tu * 100000)/100000.0) << ", "; // max read Sec
				ofs << (round(avgReadMs/tu * 100000)/100000.0) << ", "; // avg read Sec
				ofs << (round(medianReadMs/tu * 100000)/100000.0); // median read Sec
			}
			if(hasTestOperation(SEND_RECEIVE_OPERATION)){
				ofs << ", " << (round(transferedBytes*tu/su/m_elapsedSendMs * 100000)/100000.0) << ", "; // send/recv GB/s
				ofs << (round(transferedBytes*tu/su/maxSendMs * 100000)/100000.0) << ", "; // min send/recv GB/s
				ofs << (round(transferedBytes*tu/su/minSendMs * 100000)/100000.0) << ", "; // max send/recv GB/s
				ofs << (round(transferedBytes*tu/su/avgSendMs * 100000)/100000.0) << ", "; // avg send/recv GB/s
				ofs << (round(transferedBytes*tu/su/medianSendMs * 100000)/100000.0) << ", "; // median send/recv GB/s
				ofs << (round(m_elapsedSendMs/tu * 100000)/100000.0) << ", "; // send Sec
				ofs << (round(minSendMs/tu * 100000)/100000.0) << ", "; // min send Sec
				ofs << (round(maxSendMs/tu * 100000)/100000.0) << ", "; // max send Sec
				ofs << (round(avgSendMs/tu * 100000)/100000.0) << ", "; // avg send Sec
				ofs << (round(medianSendMs/tu * 100000)/100000.0); // median send Sec
			}

			ofs << std::endl; ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		oss << rdma::CONSOLE_PRINT_NOTATION << rdma::CONSOLE_PRINT_PRECISION;
		oss << " measurement for sending and writeImm is executed as alternating send/receive bursts with " << Config::RDMA_MAX_WR << " operations per burst" << std::endl;
		oss << "transfered = " << rdma::PerfTest::convertByteSize(transferedBytes) << std::endl;
		if(hasTestOperation(WRITE_OPERATION)){
			oss << " - Write:         bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/m_elapsedWriteMs); 
			oss << "  (range = " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/maxWriteMs) << " - " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/minWriteMs);
			oss << " ; avg=" << rdma::PerfTest::convertBandwidth(transferedBytes*tu/avgWriteMs) << " ; median=";
			oss << rdma::PerfTest::convertBandwidth(transferedBytes*tu/minWriteMs) << ")";
			oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedWriteMs) << "  (range=";
			oss << rdma::PerfTest::convertTime(minWriteMs) << "-" << rdma::PerfTest::convertTime(maxWriteMs);
			oss << " ; avg=" << rdma::PerfTest::convertTime(avgWriteMs) << " ; median=" << rdma::PerfTest::convertTime(medianWriteMs) << ")" << std::endl;
		}
		if(hasTestOperation(READ_OPERATION)){
			oss << " - Read:          bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/m_elapsedReadMs);
			oss << "  (range = " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/maxReadMs) << " - " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/minReadMs);
			oss << " ; avg=" << rdma::PerfTest::convertBandwidth(transferedBytes*tu/avgReadMs) << " ; median=";
			oss << rdma::PerfTest::convertBandwidth(transferedBytes*tu/minReadMs) << ")";
			oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedReadMs) << "  (range=";
			oss << rdma::PerfTest::convertTime(minReadMs) << "-" << rdma::PerfTest::convertTime(maxReadMs);
			oss << " ; avg=" << rdma::PerfTest::convertTime(avgReadMs) << " ; median=" << rdma::PerfTest::convertTime(medianReadMs) << ")" << std::endl;
		}
		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			oss << " - Send:          bandwidth = " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/m_elapsedSendMs);
			oss << "  (range = " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/maxSendMs) << " - " << rdma::PerfTest::convertBandwidth(transferedBytes*tu/minSendMs);
			oss << " ; avg=" << rdma::PerfTest::convertBandwidth(transferedBytes*tu/avgSendMs) << " ; median=";
			oss << rdma::PerfTest::convertBandwidth(transferedBytes*tu/minSendMs) << ")";
			oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedSendMs) << "  (range=";
			oss << rdma::PerfTest::convertTime(minSendMs) << "-" << rdma::PerfTest::convertTime(maxSendMs);
			oss << " ; avg=" << rdma::PerfTest::convertTime(avgSendMs) << " ; median=" << rdma::PerfTest::convertTime(medianSendMs) << ")" << std::endl;
		}
		return oss.str();

	}
	return NULL;
}