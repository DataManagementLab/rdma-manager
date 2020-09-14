#include "OperationsCountPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"

#include <limits>
#include <algorithm>

mutex rdma::OperationsCountPerfTest::waitLock;
condition_variable rdma::OperationsCountPerfTest::waitCv;
bool rdma::OperationsCountPerfTest::signaled;
rdma::TestOperation rdma::OperationsCountPerfTest::testOperation;

rdma::OperationsCountPerfClientThread::OperationsCountPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, size_t packet_size, int buffer_slots, size_t iterations_per_thread, size_t max_rdma_wr_per_thread, WriteMode write_mode) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "OperationsCountPerfTestClient", ownIpPort, sequencerIpPort);
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
			std::cerr << "OperationsCountPerfClientThread::OperationsCountPerfClientThread(): Could not connect to '" << conn << "'" << std::endl;
			throw invalid_argument("OperationsCountPerfClientThread connection failed");
		}
		//std::cout << "Thread connected to '" << conn << "'" << std::endl; // TODO REMOVE
		m_addr.push_back(nodeId);
		m_client->remoteAlloc(conn, m_memory_size_per_thread, m_remOffsets[i]); // one chunk needed on remote side for write & read
	}

	m_local_memory = m_client->localMalloc(this->m_memory_size_per_thread * 2); // two chunks needed on local side for send & receive
	m_local_memory->openContext();
	m_local_memory->setMemory(1);
}

rdma::OperationsCountPerfClientThread::~OperationsCountPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_memory_size_per_thread, m_remOffsets[i]);
	}
    delete m_remOffsets;
	delete m_local_memory; // implicitly deletes local allocs in RDMAClient
	delete m_client;
}

void rdma::OperationsCountPerfClientThread::run() {
	unique_lock<mutex> lck(OperationsCountPerfTest::waitLock);
	if (!OperationsCountPerfTest::signaled) {
		m_ready = true;
		OperationsCountPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	m_ready = false;
	
	int sendCounter = 0, receiveCounter = 0, totalBudget = m_iterations_per_thread;
	uint32_t localBaseOffset = (uint32_t)m_local_memory->getRootOffset();
	auto start = rdma::PerfTest::startTimer();
	switch(OperationsCountPerfTest::testOperation){
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
				default: throw invalid_argument("OperationsCountPerfClientThread unknown write mode");
			}
			m_elapsedWrite = rdma::PerfTest::stopTimer(start);
			break;
		case READ_OPERATION: // Read
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations_per_thread - 1));
				int offset = (i % m_buffer_slots) * m_packet_size;
				m_client->read(m_addr[connIdx], m_remOffsets[connIdx]+offset, m_local_memory->pointer(offset), m_packet_size, signaled);
			}
			m_elapsedRead = rdma::PerfTest::stopTimer(start);
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
			m_elapsedSend = rdma::PerfTest::stopTimer(start);
			break;
		default: throw invalid_argument("OperationsCountPerfClientThread unknown test mode");
	}
}



rdma::OperationsCountPerfServerThread::OperationsCountPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t packet_size, int buffer_slots, size_t iterations_per_thread, size_t max_rdma_wr_per_thread, WriteMode write_mode, int thread_id) {
	this->m_server = server;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size_per_thread = packet_size * buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_max_rdma_wr_per_thread = max_rdma_wr_per_thread;
	this->m_write_mode = write_mode;
	this->m_thread_id = thread_id;
	this->m_local_memory = server->localMalloc(this->m_memory_size_per_thread * 2); // two chunks needed on local side for send & receive + one chunk by remote
	this->m_local_memory->openContext();
}


rdma::OperationsCountPerfServerThread::~OperationsCountPerfServerThread() {
	delete m_local_memory;  // implicitly deletes local allocs in RDMAServer
}


void rdma::OperationsCountPerfServerThread::run() {
	unique_lock<mutex> lck(OperationsCountPerfTest::waitLock);
	if (!OperationsCountPerfTest::signaled) {
		m_ready = true;
		OperationsCountPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	m_ready = false;

	const std::vector<size_t> clientIds = m_server->getConnectedConnIDs();

	// Measure operations/s for receiving
	//auto start = rdma::PerfTest::startTimer();
	size_t i = 0;
	int sendCounter = 0, receiveCounter = 0, totalBudget = m_iterations_per_thread, budgetR = totalBudget, budgetS;
	uint32_t remoteBaseOffset;

	switch(OperationsCountPerfTest::testOperation){
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


rdma::OperationsCountPerfTest::OperationsCountPerfTest(int testOperations, bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int thread_count, uint64_t packet_size, int buffer_slots, uint64_t iterations_per_thread, WriteMode write_mode) : PerfTest(testOperations){
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
	this->m_memory_size = thread_count * packet_size * buffer_slots * 3; // 3x because for send + receive + write/read separat
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_write_mode = (write_mode!=WRITE_MODE_AUTO ? write_mode : rdma::OperationsCountPerfTest::DEFAULT_WRITE_MODE);
	this->m_rdma_addresses = rdma_addresses;
}
rdma::OperationsCountPerfTest::~OperationsCountPerfTest(){
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

std::string rdma::OperationsCountPerfTest::getTestParameters(bool forCSV){
	std::ostringstream oss;
	oss << (m_is_server ? "Server" : "Client") << ", threads=" << m_thread_count << ", bufferslots=" << m_buffer_slots;
	if(!forCSV){ oss << ", packetsize=" << m_packet_size; }
	oss << ", memory=" << m_memory_size << " (2x " << m_thread_count << "x " << m_buffer_slots << "x " << m_packet_size << ")";
	oss << ", memory_type=" << getMemoryName(m_local_gpu_index, m_actual_gpu_index) << (m_remote_gpu_index!=-404 ? "->"+getMemoryName(m_remote_gpu_index) : "");
	oss << ", iterations=" << (m_iterations_per_thread*m_thread_count) << ", writemode=" << (m_write_mode==WRITE_MODE_NORMAL ? "Normal" : "Immediate");
	return oss.str();
}

std::string rdma::OperationsCountPerfTest::getTestParameters(){
	return getTestParameters(false);
}

void rdma::OperationsCountPerfTest::makeThreadsReady(TestOperation testOperation){
	OperationsCountPerfTest::testOperation = testOperation;
	OperationsCountPerfTest::signaled = false;

	for(OperationsCountPerfServerThread* perfThread : m_server_threads){ perfThread->start(); }
	for(OperationsCountPerfServerThread* perfThread : m_server_threads){ while(!perfThread->ready()) usleep(Config::RDMA_SLEEP_INTERVAL); }

	for(OperationsCountPerfClientThread* perfThread : m_client_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
}

void rdma::OperationsCountPerfTest::runThreads(){
	OperationsCountPerfTest::signaled = false;
	unique_lock<mutex> lck(OperationsCountPerfTest::waitLock);
	OperationsCountPerfTest::waitCv.notify_all();
	OperationsCountPerfTest::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_server_threads.size(); i++) {
		m_server_threads[i]->join();
	}
	for (size_t i = 0; i < m_client_threads.size(); i++) {
		m_client_threads[i]->join();
	}
}

void rdma::OperationsCountPerfTest::setupTest(){
	m_elapsedWrite = -1;
	m_elapsedRead = -1;
	m_elapsedSend = -1;
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
		m_server = new RDMAServer<ReliableRDMA>("OperationsCountTestRDMAServer", m_rdma_port, Network::getAddressOfConnection(m_ownIpPort), m_memory, m_sequencerIpPort);
		for (int thread_id = 0; thread_id < m_thread_count; thread_id++) {
			OperationsCountPerfServerThread* perfThread = new OperationsCountPerfServerThread(m_server, m_packet_size, m_buffer_slots, m_iterations_per_thread, max_rdma_wr_per_thread, m_write_mode, thread_id);
			m_server_threads.push_back(perfThread);
		}
		/* If server only allows to be single threaded
		OperationsCountPerfServerThread* perfThread = new OperationsCountPerfServerThread(m_server, m_memory_size_per_thread*m_thread_count, m_iterations*m_thread_count);
		m_server_threads.push_back(perfThread); */

	} else {
		// Client
		for (int i = 0; i < m_thread_count; i++) {
			OperationsCountPerfClientThread* perfThread = new OperationsCountPerfClientThread(m_memory, m_rdma_addresses, m_ownIpPort, m_sequencerIpPort, m_packet_size, m_buffer_slots, m_iterations_per_thread, max_rdma_wr_per_thread, m_write_mode);
			m_client_threads.push_back(perfThread);
		}
	}
}

void rdma::OperationsCountPerfTest::runTest(){
	if(m_is_server){
		// Server
		std::cout << "Starting server on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "' . . ." << std::endl;
		if(!m_server->startServer()){
			std::cerr << "OperationsCountPerfTest::runTest(): Could not start server" << std::endl;
			throw invalid_argument("OperationsCountPerfTest server startup failed");
		} else {
			std::cout << "Server running on '" << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_rdma_port << "'" << std::endl; // TODO REMOVE
		}

		// waiting until clients have connected
		while(m_server->getConnectedConnIDs().size() < (size_t)m_thread_count) usleep(Config::RDMA_SLEEP_INTERVAL);

		if(hasTestOperation(WRITE_OPERATION)){
			makeThreadsReady(WRITE_OPERATION); // receive
			//auto startReceive = rdma::PerfTest::startTimer();
			runThreads();
			//m_elapsedReceive = rdma::PerfTest::stopTimer(startReceive);
		}

		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			makeThreadsReady(SEND_RECEIVE_OPERATION); // receive
			//auto startReceive = rdma::PerfTest::startTimer();
			runThreads();
			//m_elapsedReceive = rdma::PerfTest::stopTimer(startReceive);
		}

		// wait until server is done
		while (m_server->isRunning() && m_server->getConnectedConnIDs().size() > 0) usleep(Config::RDMA_SLEEP_INTERVAL);
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client

        // Measure operations/s for writing
		if(hasTestOperation(WRITE_OPERATION)){
			makeThreadsReady(WRITE_OPERATION); // write
			usleep(4 * Config::RDMA_SLEEP_INTERVAL); // let server first post the receives if writeImm
			auto startWrite = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedWrite = rdma::PerfTest::stopTimer(startWrite);
		}

		// Measure operations/s for reading
		if(hasTestOperation(READ_OPERATION)){
			makeThreadsReady(READ_OPERATION); // read
			auto startRead = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedRead = rdma::PerfTest::stopTimer(startRead);
		}

		// Measure operations/s for sending
		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			makeThreadsReady(SEND_RECEIVE_OPERATION); // send
			usleep(4 * Config::RDMA_SLEEP_INTERVAL); // let server first post the receives
			auto startSend = rdma::PerfTest::startTimer();
			runThreads();
			m_elapsedSend = rdma::PerfTest::stopTimer(startSend);
		}
	}
}


std::string rdma::OperationsCountPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
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

		int64_t maxWriteNs=-1, minWriteNs=std::numeric_limits<int64_t>::max();
		int64_t maxReadNs=-1, minReadNs=std::numeric_limits<int64_t>::max();
		int64_t maxSendNs=-1, minSendNs=std::numeric_limits<int64_t>::max();
		int64_t arrWriteNs[m_thread_count];
		int64_t arrReadNs[m_thread_count];
		int64_t arrSendNs[m_thread_count];
		long double avgWriteNs=0, medianWriteNs, avgReadNs=0, medianReadNs, avgSendNs=0, medianSendNs;
		const long double div = 1; // TODO not sure why additional   m_thread_count  is too much
		const long double divAvg = m_client_threads.size() * div;
		for(size_t i=0; i<m_client_threads.size(); i++){
			OperationsCountPerfClientThread *thr = m_client_threads[i];
			if(thr->m_elapsedWrite < minWriteNs) minWriteNs = thr->m_elapsedWrite;
			if(thr->m_elapsedWrite > maxWriteNs) maxWriteNs = thr->m_elapsedWrite;
			avgWriteNs += (long double) thr->m_elapsedWrite / divAvg;
			arrWriteNs[i] = thr->m_elapsedWrite;
			if(thr->m_elapsedRead < minReadNs) minReadNs = thr->m_elapsedRead;
			if(thr->m_elapsedRead > maxReadNs) maxReadNs = thr->m_elapsedRead;
			avgReadNs += (long double) thr->m_elapsedRead / divAvg;
			arrReadNs[i] = thr->m_elapsedRead;
			if(thr->m_elapsedSend < minSendNs) minSendNs = thr->m_elapsedSend;
			if(thr->m_elapsedSend > maxSendNs) maxSendNs = thr->m_elapsedSend;
			avgSendNs += (long double) thr->m_elapsedSend / divAvg;
			arrSendNs[i] = thr->m_elapsedSend;
		}
		minWriteNs /= div; maxWriteNs /= div;
		minReadNs /= div; maxReadNs /= div;
		minSendNs /= div; maxSendNs /= div;

		std::sort(arrWriteNs, arrWriteNs + m_thread_count);
		std::sort(arrReadNs, arrReadNs + m_thread_count);
		std::sort(arrSendNs, arrSendNs + m_thread_count);
		medianWriteNs = arrWriteNs[(int)(m_thread_count/2)] / div;
		medianReadNs = arrReadNs[(int)(m_thread_count/2)] / div;
		medianSendNs = arrSendNs[(int)(m_thread_count/2)] / div;

		// write results into CSV file
		if(!csvFileName.empty()){
			const long double su = 1000*1000; // size unit (operations to megaOps)
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			ofs << rdma::CSV_PRINT_NOTATION << rdma::CSV_PRINT_PRECISION;
			if(csvAddHeader){
				ofs << std::endl << "OPERATIONS COUNT, " << getTestParameters(true) << std::endl;
				ofs << "PacketSize [Bytes]";
				if(hasTestOperation(WRITE_OPERATION)){
					ofs << ", Write [megaOp/s], Min Write [megaOp/s], Max Write [megaOp/s], Avg Write [megaOp/s], Median Write [megaOp/s], ";
					ofs << "Write [Sec], Min Write [Sec], Max Write [Sec], Avg Write [Sec], Median Write [Sec]";
				}
				if(hasTestOperation(READ_OPERATION)){
					ofs << ", Read [megaOp/s], Min Read [megaOp/s], Max Read [megaOp/s], Avg Read [megaOp/s], Median Read [megaOp/s], ";
					ofs << "Read [Sec], Min Read [Sec], Max Read [Sec], Avg Read [Sec], Median Read [Sec]";
				}
				if(hasTestOperation(SEND_RECEIVE_OPERATION)){
					ofs << ", Send/Recv [megaOp/s], Min Send/Recv [megaOp/s], Max Send/Recv [megaOp/s], Avg Send/Recv [megaOp/s], Median Send/Recv [megaOp/s], ";
					ofs << "Send/Recv [Sec], Min Send/Recv [Sec], Max Send/Recv [Sec], Avg Send/Recv [Sec], Median Send/Recv [Sec]";
				}
				ofs << std::endl;
			}
			ofs << m_packet_size; // packet size Bytes
			if(hasTestOperation(WRITE_OPERATION)){
				ofs << ", " << (round(iters*tu/su/m_elapsedWrite * 100000)/100000.0) << ", "; // write Op/s
				ofs << (round(iters*tu/su/maxWriteNs * 100000)/100000.0) << ", "; // min write Op/s
				ofs << (round(iters*tu/su/minWriteNs * 100000)/100000.0) << ", "; // max write Op/s
				ofs << (round(iters*tu/su/avgWriteNs * 100000)/100000.0) << ", "; // avg write Op/s
				ofs << (round(iters*tu/su/medianWriteNs * 100000)/100000.0) << ", "; // median write Op/s
				ofs << (round(m_elapsedWrite/tu * 100000)/100000.0) << ", "; // write Sec
				ofs << (round(minWriteNs/tu * 100000)/100000.0) << ", "; // min write Sec
				ofs << (round(maxWriteNs/tu * 100000)/100000.0) << ", "; // max write Sec
				ofs << (round(avgWriteNs/tu * 100000)/100000.0) << ", "; // avg write Sec
				ofs << (round(medianWriteNs/tu * 100000)/100000.0); // median write Sec
			}
			if(hasTestOperation(READ_OPERATION)){
				ofs << ", " << (round(iters*tu/su/m_elapsedRead * 100000)/100000.0) << ", "; // read Op/s
				ofs << (round(iters*tu/su/maxReadNs * 100000)/100000.0) << ", "; // min read Op/s
				ofs << (round(iters*tu/su/minReadNs * 100000)/100000.0) << ", "; // max read Op/s
				ofs << (round(iters*tu/su/avgReadNs * 100000)/100000.0) << ", "; // avg read Op/s
				ofs << (round(iters*tu/su/medianReadNs * 100000)/100000.0) << ", "; // median read Op/s
				ofs << (round(m_elapsedRead/tu * 100000)/100000.0) << ", "; // read Sec
				ofs << (round(minReadNs/tu * 100000)/100000.0) << ", "; // min read Sec
				ofs << (round(maxReadNs/tu * 100000)/100000.0) << ", "; // max read Sec
				ofs << (round(avgReadNs/tu * 100000)/100000.0) << ", "; // avg read Sec
				ofs << (round(medianReadNs/tu * 100000)/100000.0); // median read Sec
			}
			if(hasTestOperation(SEND_RECEIVE_OPERATION)){
				ofs << ", " << (round(iters*tu/su/m_elapsedSend * 100000)/100000.0) << ", "; // send/recv Op/s
				ofs << (round(iters*tu/su/maxSendNs * 100000)/100000.0) << ", "; // min send/recv Op/s
				ofs << (round(iters*tu/su/minSendNs * 100000)/100000.0) << ", "; // max send/recv Op/s
				ofs << (round(iters*tu/su/avgSendNs * 100000)/100000.0) << ", "; // avg send/recv Op/s
				ofs << (round(iters*tu/su/medianSendNs * 100000)/100000.0) << ", "; // median send/recv Op/s
				ofs << (round(m_elapsedSend/tu * 100000)/100000.0) << ", "; // send Sec
				ofs << (round(minSendNs/tu * 100000)/100000.0) << ", "; // min send Sec
				ofs << (round(maxSendNs/tu * 100000)/100000.0) << ", "; // max send Sec
				ofs << (round(avgSendNs/tu * 100000)/100000.0) << ", "; // avg send Sec
				ofs << (round(medianSendNs/tu * 100000)/100000.0); // median send Sec
			}
			ofs << std::endl; ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		oss << rdma::CONSOLE_PRINT_NOTATION << rdma::CONSOLE_PRINT_PRECISION;
		oss << " measurement for sending and writeImm is executed as alternating send/receive bursts with " << Config::RDMA_MAX_WR << " operations per burst" << std::endl;
		if(hasTestOperation(WRITE_OPERATION)){
			oss << " - Write:         operations = " << rdma::PerfTest::convertCountPerSec(iters*tu/m_elapsedWrite); 
			oss << "  (range = " << rdma::PerfTest::convertCountPerSec(iters*tu/maxWriteNs) << " - " << rdma::PerfTest::convertCountPerSec(iters*tu/minWriteNs);
			oss << " ; avg=" << rdma::PerfTest::convertCountPerSec(iters*tu/avgWriteNs) << " ; median=";
			oss << rdma::PerfTest::convertCountPerSec(iters*tu/minWriteNs) << ")";
			oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedWrite) << "  (range=";
			oss << rdma::PerfTest::convertTime(minWriteNs) << "-" << rdma::PerfTest::convertTime(maxWriteNs);
			oss << " ; avg=" << rdma::PerfTest::convertTime(avgWriteNs) << " ; median=" << rdma::PerfTest::convertTime(medianWriteNs) << ")" << std::endl;
		}
		if(hasTestOperation(READ_OPERATION)){
			oss << " - Read:          operations = " << rdma::PerfTest::convertCountPerSec(iters*tu/m_elapsedRead);
			oss << "  (range = " << rdma::PerfTest::convertCountPerSec(iters*tu/maxReadNs) << " - " << rdma::PerfTest::convertCountPerSec(iters*tu/minReadNs);
			oss << " ; avg=" << rdma::PerfTest::convertCountPerSec(iters*tu/avgReadNs) << " ; median=";
			oss << rdma::PerfTest::convertCountPerSec(iters*tu/minReadNs) << ")";
			oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedRead) << "  (range=";
			oss << rdma::PerfTest::convertTime(minReadNs) << "-" << rdma::PerfTest::convertTime(maxReadNs);
			oss << " ; avg=" << rdma::PerfTest::convertTime(avgReadNs) << " ; median=" << rdma::PerfTest::convertTime(medianReadNs) << ")" << std::endl;
		}
		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			oss << " - Send:          operations = " << rdma::PerfTest::convertCountPerSec(iters*tu/m_elapsedSend);
			oss << "  (range = " << rdma::PerfTest::convertCountPerSec(iters*tu/maxSendNs) << " - " << rdma::PerfTest::convertCountPerSec(iters*tu/minSendNs);
			oss << " ; avg=" << rdma::PerfTest::convertCountPerSec(iters*tu/avgSendNs) << " ; median=";
			oss << rdma::PerfTest::convertCountPerSec(iters*tu/minSendNs) << ")";
			oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedSend) << "  (range=";
			oss << rdma::PerfTest::convertTime(minSendNs) << "-" << rdma::PerfTest::convertTime(maxSendNs);
			oss << " ; avg=" << rdma::PerfTest::convertTime(avgSendNs) << " ; median=" << rdma::PerfTest::convertTime(medianSendNs) << ")" << std::endl;
		}
		return oss.str();

	}
	return NULL;
}