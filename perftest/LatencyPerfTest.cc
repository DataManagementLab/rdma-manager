#include "LatencyPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"

mutex rdma::LatencyPerfTest::waitLock;
condition_variable rdma::LatencyPerfTest::waitCv;
bool rdma::LatencyPerfTest::signaled;
rdma::TestMode rdma::LatencyPerfTest::testMode;
int rdma::LatencyPerfTest::thread_count;


rdma::LatencyPerfClientThread::LatencyPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, size_t packet_size, int buffer_slots, size_t iterations, WriteMode write_mode) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "LatencyPerfTestClient");
	this->m_rdma_addresses = rdma_addresses;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size_per_thread = packet_size * buffer_slots;
	this->m_iterations = iterations;
	this->m_write_mode = write_mode;
	this->m_remOffsets = new size_t[m_rdma_addresses.size()];

	this->m_arrWriteMs = new int64_t[iterations];
	this->m_arrReadMs = new int64_t[iterations];
	this->m_arrSendMs = new int64_t[iterations];

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

	m_local_memory = m_client->localMalloc(m_memory_size_per_thread * 2); // write: receive via first half and send via second half
	m_local_memory->openContext();
	m_local_memory->setMemory(1);
}

rdma::LatencyPerfClientThread::~LatencyPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_memory_size_per_thread, m_remOffsets[i]);
	}
    delete m_remOffsets;
	delete m_local_memory; // implicitly deletes local allocs in RDMAClient

	delete m_arrWriteMs;
	delete m_arrReadMs;
	delete m_arrSendMs;
	delete m_client;
}

void rdma::LatencyPerfClientThread::run() {
	unique_lock<mutex> lck(LatencyPerfTest::waitLock);
	if (!LatencyPerfTest::signaled) {
		m_ready = true;
		LatencyPerfTest::waitCv.wait(lck);
	}
	lck.unlock();

	volatile void *arrSend = nullptr;
	volatile char value;
	uint32_t localBaseOffset = (uint32_t)m_local_memory->getRootOffset();
	switch(LatencyPerfTest::testMode){
		case TEST_WRITE: // Write
			switch(m_write_mode){
				case WRITE_MODE_NORMAL:
					m_local_memory->setMemory(0);
					for(size_t i = 0; i < m_iterations; i++){
						size_t connIdx = i % m_rdma_addresses.size();
						value = (i % 100) + 1;
						size_t receiveOffset = (i % m_buffer_slots) * m_packet_size;
						size_t sendOffset = receiveOffset + m_memory_size_per_thread;
						size_t remoteOffset = m_remOffsets[connIdx] + receiveOffset;
						m_local_memory->set(value, sendOffset); // send payload value
						if(m_local_memory->getChar(sendOffset) != value) // prevents compiler to switch statements
							throw runtime_error("Compiler makes stupid stuff with payload");
						arrSend = m_local_memory->pointer(sendOffset);
						auto start = rdma::PerfTest::startTimer(); 
						m_client->write(m_addr[connIdx], remoteOffset, (void*)arrSend, m_packet_size, true); // true=signaled
						int counter = 0;
						while(m_local_memory->getChar(receiveOffset) != value){
							if((++counter) % 100000000 == 0){ std::cout << "KILL ME, I'M FROZEN" << std::endl; }
						}
						int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
						m_sumWriteMs += time;
						if(m_minWriteMs > time) m_minWriteMs = time;
						if(m_maxWriteMs < time) m_maxWriteMs = time;
						m_arrWriteMs[i] = time;
					}
					break;
				case WRITE_MODE_IMMEDIATE:
					m_local_memory->setMemory(1);
					for(size_t i = 0; i < m_iterations; i++){
						size_t connIdx = i % m_rdma_addresses.size();
						size_t offset = (i % m_buffer_slots) * m_packet_size;
						size_t sendOffset =  offset + m_memory_size_per_thread;
						size_t remoteOffset = m_remOffsets[connIdx] + offset;
						arrSend = m_local_memory->pointer(sendOffset);
						auto start = rdma::PerfTest::startTimer();
						m_client->receiveWriteImm(m_addr[connIdx]);
						m_client->writeImm(m_addr[connIdx], remoteOffset, (void*)arrSend, m_packet_size, localBaseOffset, true);
						m_client->pollReceive(m_addr[connIdx], true);
						int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
						m_sumWriteMs += time;
						if(m_minWriteMs > time) m_minWriteMs = time;
						if(m_maxWriteMs < time) m_maxWriteMs = time;
						m_arrWriteMs[i] = time;
					}
					break;
				default: throw invalid_argument("LatencyPerfClientThread unknown write mode"); 
			}
			std::cout << std::endl << "WRITE DONE" << std::endl << std::endl; // TODO REMOVE
			break;

		case TEST_READ: // Read
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				int offset = (i % m_buffer_slots) * m_packet_size;
				auto start = rdma::PerfTest::startTimer();
				m_client->read(m_addr[connIdx], m_remOffsets[connIdx] + offset, m_local_memory->pointer(offset), m_packet_size, true); // true=signaled
				int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
				m_sumReadMs += time;
				if(m_minReadMs > time) m_minReadMs = time;
				if(m_maxReadMs < time) m_maxReadMs = time;
				m_arrReadMs[i] = time;
			}
			break;

		case TEST_SEND_AND_RECEIVE: // Send & Receive
			for(size_t i = 0; i < m_rdma_addresses.size(); i++){
				m_client->receive(m_addr[i], m_local_memory->pointer(), m_packet_size);
			}
			for(size_t i = 0; i < m_iterations; i++){
				size_t clientId = m_addr[i % m_rdma_addresses.size()];
				int offset = (i % m_buffer_slots) * m_packet_size, nextOffset = ((i+1) % m_buffer_slots) * m_packet_size;
				auto start = rdma::PerfTest::startTimer();
				m_client->send(clientId, m_local_memory->pointer(offset), m_packet_size, true); // true=signaled
				m_client->pollReceive(clientId, true); // true=poll
				m_client->receive(clientId, m_local_memory->pointer(nextOffset), m_packet_size);
				int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
				m_sumSendMs += time;
				if(m_minSendMs > time) m_minSendMs = time;
				if(m_maxSendMs < time) m_maxSendMs = time;
				m_arrSendMs[i] = time;
			}
			break;
		default: throw invalid_argument("LatencyPerfClientThread unknown test mode");
	}
}


rdma::LatencyPerfServerThread::LatencyPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t packet_size, int buffer_slots, size_t iterations, WriteMode write_mode, int thread_id) {
	this->m_server = server;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size_per_thread = packet_size * buffer_slots;
	this->m_iterations = iterations;
	this->m_write_mode = write_mode;
	this->m_thread_id = thread_id;
	this->m_local_memory = server->localMalloc(this->m_memory_size_per_thread);
	this->m_local_memory->openContext();
}

rdma::LatencyPerfServerThread::~LatencyPerfServerThread() {
	delete m_local_memory;  // implicitly deletes local allocs in RDMAServer
}

void rdma::LatencyPerfServerThread::run() {
	m_server->getBufferObj()->setMemory(0);
	unique_lock<mutex> lck(LatencyPerfTest::waitLock);
	if (!LatencyPerfTest::signaled) {
		m_ready = true;
		LatencyPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	const std::vector<size_t> clientIds = m_server->getConnectedConnIDs();
	const size_t clientId = clientIds[m_thread_id];
	//volatile void *arrSend = nullptr;
	volatile void *arrRecv = nullptr;
	volatile char value;
	size_t sendOffset, receiveOffset;
	uint32_t remoteBaseOffset = m_thread_id * 2 * m_memory_size_per_thread;
	switch(LatencyPerfTest::testMode){
		case TEST_WRITE: // Write
			switch(m_write_mode){
				case WRITE_MODE_NORMAL:
					for(size_t i = 0; i < m_iterations; i++){
						value = (i % 100) + 1;
						sendOffset = (i % m_buffer_slots) * m_packet_size;
						receiveOffset = sendOffset + rdma::LatencyPerfTest::thread_count * m_memory_size_per_thread;
						//arrSend = m_local_memory->pointer(sendOffset);
						arrRecv = m_local_memory->pointer(receiveOffset);
						int counter = 0;
						while(m_local_memory->getChar(receiveOffset) != value){
							if((++counter) % 100000000 == 0){ std::cout << "KILL ME, I'M FROZEN" << std::endl; }
						}
						/*m_local_memory->set(value, sendOffset); // send payload value
						if(m_local_memory->getChar(sendOffset) != value) // prevents compiler to switch statements
							throw runtime_error("Compiler makes stupid stuff with payload");
						m_server->write(clientId, remoteBaseOffset, (void*)arrSend, m_packet_size, true); // true=signaled*/
						m_server->write(clientId, remoteBaseOffset+sendOffset, (void*)arrRecv, m_packet_size, true); // true=signaled
					}
					break;
				case WRITE_MODE_IMMEDIATE:
					m_server->receiveWriteImm(clientId);
					for(size_t i = 1; i < m_iterations; i++){
						sendOffset = (i % m_buffer_slots) * m_packet_size;
						receiveOffset = sendOffset + rdma::LatencyPerfTest::thread_count * m_memory_size_per_thread;
						arrRecv = m_local_memory->pointer(receiveOffset);
						m_server->pollReceive(clientId, true, &remoteBaseOffset);
						m_server->receiveWriteImm(clientId);
						m_server->writeImm(clientId, remoteBaseOffset+sendOffset, (void*)arrRecv, m_packet_size, (uint32_t)0, true);
					}
					sendOffset = (m_iterations % m_buffer_slots) * m_packet_size;
					receiveOffset = sendOffset + rdma::LatencyPerfTest::thread_count * m_memory_size_per_thread;
					arrRecv = m_local_memory->pointer(receiveOffset);
					m_server->pollReceive(clientId, true, &remoteBaseOffset);
					m_server->writeImm(clientId, remoteBaseOffset+sendOffset, (void*)arrRecv, m_packet_size, (uint32_t)0, true);
					break;
				default: break;
			}
			break;

		case TEST_SEND_AND_RECEIVE: // Send & Receive
			m_server->receive(clientId, m_local_memory->pointer(), m_packet_size);
			for(size_t i = 0; i < m_iterations; i++){
				int off = (i % m_buffer_slots) * m_packet_size, nextOff = ((i+1) % m_buffer_slots) * m_packet_size;
				m_server->pollReceive(clientId, true); // true=poll
				m_server->receive(clientId, m_local_memory->pointer(nextOff), m_packet_size);
				m_server->send(clientId, m_local_memory->pointer(off), m_packet_size, true); // true=signaled
			}
			break;
		default: throw invalid_argument("LatencyPerfClientThread unknown test mode");
	}
}



rdma::LatencyPerfTest::LatencyPerfTest(bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, int gpu_index, int thread_count, uint64_t packet_size, int buffer_slots, uint64_t iterations, WriteMode write_mode) : PerfTest(){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_gpu_index = gpu_index;
	this->thread_count = thread_count;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size = thread_count * packet_size * buffer_slots * 2; // two times because send & receive
	this->m_iterations = iterations;
	this->m_write_mode = (write_mode!=WRITE_MODE_AUTO ? write_mode : rdma::LatencyPerfTest::DEFAULT_WRITE_MODE);
	this->m_rdma_addresses = rdma_addresses;
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
	delete m_memory;
}

std::string rdma::LatencyPerfTest::getTestParameters(bool forCSV){
	std::ostringstream oss;
	oss << (m_is_server ? "Server" : "Client") << ", threads=" << thread_count << ", bufferslots=" << m_buffer_slots;
	if(!forCSV){ oss << ", packetsize=" << m_packet_size; }
	oss << ", memory=" << m_memory_size << " (2x " << thread_count << "x " << m_buffer_slots << "x " << m_packet_size << ") [";
	if(m_gpu_index < 0){
		oss << "MAIN";
	} else {
		oss << "GPU." << m_gpu_index; 
	}
	oss << " mem], iterations=" << (m_iterations*thread_count) << ", writemode=" << (m_write_mode==WRITE_MODE_NORMAL ? "Normal" : "Immediate");
	return oss.str();
}
std::string rdma::LatencyPerfTest::getTestParameters(){
	return getTestParameters(false);
}

void rdma::LatencyPerfTest::makeThreadsReady(TestMode testMode){
	LatencyPerfTest::testMode = testMode;
	LatencyPerfTest::signaled = false;
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
	#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
		m_memory = (m_gpu_index<0 ? (rdma::BaseMemory*)new rdma::MainMemory(m_memory_size) : (rdma::BaseMemory*)new rdma::CudaMemory(m_memory_size, m_gpu_index));
	#else
		m_memory = (rdma::BaseMemory*)new MainMemory(m_memory_size);
	#endif

	if(m_is_server){
		// Server
		m_server = new RDMAServer<ReliableRDMA>("LatencyTestRDMAServer", m_rdma_port, m_memory);
		for (int thread_id = 0; thread_id < thread_count; thread_id++) {
			LatencyPerfServerThread* perfThread = new LatencyPerfServerThread(m_server, m_packet_size, m_buffer_slots, m_iterations, m_write_mode, thread_id);
			m_server_threads.push_back(perfThread);
		}

	} else {
		// Client
		for (int i = 0; i < thread_count; i++) {
			LatencyPerfClientThread* perfThread = new LatencyPerfClientThread(m_memory, m_rdma_addresses, m_packet_size, m_buffer_slots, m_iterations, m_write_mode);
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
		while(m_server->getConnectedConnIDs().size() < (size_t)thread_count) usleep(Config::RDMA_SLEEP_INTERVAL);

		// Measure Latency for writing
		makeThreadsReady(TEST_WRITE);
        runThreads();

		// Measure Latency for receiving
		makeThreadsReady(TEST_SEND_AND_RECEIVE); // receive
        runThreads();

		// wait until server is done
		while (m_server->isRunning() && m_server->getConnectedConnIDs().size() > 0) usleep(Config::RDMA_SLEEP_INTERVAL);
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client

        // Measure Latency for writing
		makeThreadsReady(TEST_WRITE); // write
		usleep(4 * Config::RDMA_SLEEP_INTERVAL); // let server first post the receives
        runThreads();

		// Measure Latency for reading
		makeThreadsReady(TEST_READ); // read
		usleep(Config::RDMA_SLEEP_INTERVAL); // let server first make ready (shouldn't be necessary)
        runThreads();

		// Measure Latency for sending
		makeThreadsReady(TEST_SEND_AND_RECEIVE); // send
		usleep(4 * Config::RDMA_SLEEP_INTERVAL); // let server first post the receives
        runThreads();
	}
}


std::string rdma::LatencyPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
	if(m_is_server){
		return "only client";
	} else {

		int64_t minWriteMs=std::numeric_limits<int64_t>::max(), maxWriteMs=-1, medianWriteMs=-1;
		int64_t minReadMs=std::numeric_limits<int64_t>::max(), maxReadMs=-1, medianReadMs=-1;
		int64_t minSendMs=std::numeric_limits<int64_t>::max(), maxSendMs=-1, medianSendMs=-1;
		long double avgWriteMs=0, avgReadMs=0, avgSendMs=0;
		int64_t mediansWriteNs[thread_count], mediansReadNs[thread_count], mediansSendNs[thread_count];

		for(size_t i=0; i<m_client_threads.size(); i++){
			long double div = (long double)m_iterations / (long double)thread_count;
			LatencyPerfClientThread *thr = m_client_threads[i];
			if(minWriteMs > thr->m_minWriteMs) minWriteMs = thr->m_minWriteMs;
			if(maxWriteMs < thr->m_maxWriteMs) maxWriteMs = thr->m_maxWriteMs;
			avgWriteMs += thr->m_sumWriteMs / div;
			if(minReadMs > thr->m_minReadMs) minReadMs = thr->m_minReadMs;
			if(maxReadMs < thr->m_maxReadMs) maxReadMs = thr->m_maxReadMs;
			avgReadMs += thr->m_sumReadMs / div;
			if(minSendMs > thr->m_minSendMs) minSendMs = thr->m_minSendMs;
			if(maxSendMs < thr->m_maxSendMs) maxSendMs = thr->m_maxSendMs;
			avgSendMs += thr->m_sumSendMs / div;

			std::sort(thr->m_arrWriteMs, thr->m_arrWriteMs + m_iterations);
			mediansWriteNs[i] = thr->m_arrWriteMs[(int)(m_iterations/2)];
			std::sort(thr->m_arrReadMs, thr->m_arrReadMs + m_iterations);
			mediansReadNs[i] = thr->m_arrReadMs[(int)(m_iterations/2)];
			std::sort(thr->m_arrSendMs, thr->m_arrSendMs + m_iterations);
			mediansSendNs[i] = thr->m_arrSendMs[(int)(m_iterations/2)];
		}
		std::sort(mediansWriteNs, mediansWriteNs + thread_count);
		medianWriteMs = mediansWriteNs[(int)(thread_count/2)] / (long double)thread_count;
		std::sort(mediansReadNs, mediansReadNs + thread_count);
		medianReadMs = mediansReadNs[(int)(thread_count/2)] / (long double)thread_count;
		std::sort(mediansSendNs, mediansSendNs + thread_count);
		medianSendMs = mediansSendNs[(int)(thread_count/2)] / (long double)thread_count;

		minWriteMs /= (long double)thread_count;
		maxWriteMs /= (long double)thread_count;
		minReadMs /= (long double)thread_count;
		maxReadMs /= (long double)thread_count;
		minSendMs /= (long double)thread_count;
		maxSendMs /= (long double)thread_count;

		// write results into CSV file
		if(!csvFileName.empty()){
			const uint64_t ustu = 1000; // nanosec to microsec
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			if(csvAddHeader){
				ofs << std::endl << "LATENCY, " << getTestParameters(true) << std::endl;
				ofs << "PacketSize [Bytes], Avg Write [usec], Avg Read [usec], Avg Send/Recv [usec], ";
				ofs << "Median Write [usec], Median Read [usec], Median Send/Recv [usec], ";
				ofs << "Min Write [usec], Min Read [usec], Min Send/Recv [usec], ";
				ofs << "Max Write [usec], Max Read [usec], Max Send/Recv [usec]" << std::endl;
			}
			ofs << m_packet_size << ", "; // packet size Bytes
			ofs << (round(avgWriteMs/ustu * 10)/10.0) << ", " << (round(avgReadMs/ustu * 10)/10.0) << ", "; // avg write, read us
			ofs << (round(avgSendMs/ustu * 10)/10.0) << ", "; // avg send us
			ofs << (round(medianWriteMs/ustu * 10)/10.0) << ", " << (round(medianReadMs/ustu * 10)/10.0) << ", "; // median write, read us
			ofs << (round(medianSendMs/ustu * 10)/10.0) << ", "; // median send us
			ofs << (round(minWriteMs/ustu * 10)/10.0) << ", " << (round(minReadMs/ustu * 10)/10.0) << ", "; // min write, read us
			ofs << (round(minSendMs/ustu * 10)/10.0) << ", "; // min send us
			ofs << (round(maxWriteMs/ustu * 10)/10.0) << ", " << (round(maxReadMs/ustu * 10)/10.0) << ", "; // max write, read us
			ofs << (round(maxSendMs/ustu * 10)/10.0) << ", " << std::endl; // max send us
			ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		std::cout << "Measured as 'one trip time' latencies:" << std::endl;
		std::cout << " - Write:           average = " << rdma::PerfTest::convertTime(avgWriteMs) << "    median = " << rdma::PerfTest::convertTime(medianWriteMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minWriteMs) << " - " << rdma::PerfTest::convertTime(maxWriteMs) << std::endl;
		std::cout << " - Read:            average = " << rdma::PerfTest::convertTime(avgReadMs) << "    median = " << rdma::PerfTest::convertTime(medianReadMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minReadMs) << " - " << rdma::PerfTest::convertTime(maxReadMs) << std::endl;
		std::cout << " - Send:            average = " << rdma::PerfTest::convertTime(avgSendMs) << "    median = " << rdma::PerfTest::convertTime(medianSendMs);
		std::cout << "    range = " <<  rdma::PerfTest::convertTime(minSendMs) << " - " << rdma::PerfTest::convertTime(maxSendMs) << std::endl;
		return oss.str();

	}
	return NULL;
}
