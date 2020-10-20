#include "LatencyPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "../src/utils/Config.h"

mutex rdma::LatencyPerfTest::waitLock;
condition_variable rdma::LatencyPerfTest::waitCv;
bool rdma::LatencyPerfTest::signaled;
rdma::TestOperation rdma::LatencyPerfTest::testOperation;
int rdma::LatencyPerfTest::client_count;
int rdma::LatencyPerfTest::thread_count;

/*	LT: Local Thread, RT: Remote Thread, B: BufferSlot, C: Connection, R: Receive Packet Size, S: Send Packet Size
 *
 *	Client Memory:	LT1{ B1[ C1(R, S), C2(R, S), ... ], B2[ C1(R, S), C2(R, S), ... ] }, T2{ B1[ C1(R, S), C2(R, S), ... ], LT2[ C1(R, S), C2(R, S), ... ] }, ...
 *	Server Memory:	LT1{ B1[ S ], B2[ S ], ... }, LT2{ B1[ S ], B2[ S ], ... }, ..., RT1{ B1[ R ], B2[ R ], ... }, RT2{ B1[ R ], B2[ R ], ... }
 */


rdma::LatencyPerfClientThread::LatencyPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, std::string ownIpPort, std::string sequencerIpPort, size_t packet_size, int buffer_slots, size_t iterations_per_thread, WriteMode write_mode) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "LatencyPerfTestClient", ownIpPort, sequencerIpPort);
	this->m_rdma_addresses = rdma_addresses;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_remote_memory_size_per_thread = packet_size * buffer_slots; // remote memory size per thread
	this->m_memory_size_per_thread = m_remote_memory_size_per_thread * rdma_addresses.size() * 2; // local memory size per thread (*2 because send/recv separat)
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_write_mode = write_mode;
	this->m_remOffsets = new size_t[m_rdma_addresses.size()];

	this->m_arrWriteMs = new int64_t[iterations_per_thread];
	this->m_arrReadMs = new int64_t[iterations_per_thread];
	this->m_arrSendMs = new int64_t[iterations_per_thread];

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
		m_client->remoteAlloc(conn, m_remote_memory_size_per_thread, m_remOffsets[i]);
	}
	m_local_memory = m_client->localMalloc(m_memory_size_per_thread);
	m_local_memory->openContext();

	// send nodeID to tell remote thread how to respond
	size_t nodeIdMsgSize = sizeof(uint32_t);
	if(m_local_memory->isGPUMemory() && nodeIdMsgSize < rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE) nodeIdMsgSize=rdma::Config::GPUDIRECT_MINIMUM_MSG_SIZE; // TODO REMOVE
	for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
		m_local_memory->set((uint32_t)m_client->getOwnNodeID(), 0);
		m_client->write(m_addr[connIdx], m_remOffsets[connIdx], m_local_memory->pointer(), nodeIdMsgSize, true);
	}
	m_local_memory->setMemory(1);
}

rdma::LatencyPerfClientThread::~LatencyPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_remote_memory_size_per_thread, m_remOffsets[i]);
	}
	delete m_remOffsets;
	delete m_local_memory;
    

	delete m_arrWriteMs;
	delete m_arrReadMs;
	delete m_arrSendMs;
	delete m_client;
}

void rdma::LatencyPerfClientThread::run() {
	rdma::PerfTest::global_barrier_client(m_client, m_addr); // global barrier
	unique_lock<mutex> lck(LatencyPerfTest::waitLock); // local barrier
	if (!LatencyPerfTest::signaled) {
		m_ready = true;
		LatencyPerfTest::waitCv.wait(lck);
	}
	lck.unlock();
	m_ready = false;

	size_t offset, sendOffset, receiveOffset, remoteOffset;
	volatile void *arrSend = nullptr;
	volatile void *arrRecv = nullptr;
	volatile uint32_t receiveRootOffsetIdx, tmp = 0;
	switch(LatencyPerfTest::testOperation){
		case WRITE_OPERATION: // Write
			switch(m_write_mode){
				case WRITE_MODE_NORMAL:
					for(size_t i = 0; i < m_iterations_per_thread; i++){
						const size_t flip = 1 + ((i/m_buffer_slots) % 2); // each buffer slot has alternating +1 or +0
						offset = (i % m_buffer_slots) * m_packet_size;
						if(offset==0){ m_local_memory->setMemory(0); } // reset threads memory block to be able to listen again in while loop
						for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
							receiveOffset = offset * (connIdx+1) * 2;
							sendOffset = receiveOffset + m_packet_size;
							receiveRootOffsetIdx = (m_local_memory->getRootOffset() + receiveOffset) / m_packet_size * 2 + flip; // +flip just to have any payload and for detection on remote side
							m_local_memory->set(receiveRootOffsetIdx, sendOffset);
							remoteOffset = m_remOffsets[connIdx] + offset;
							arrSend = m_local_memory->pointer(sendOffset);

							/*if(m_local_memory->getUInt32(sendOffset) != receiveRootOffsetIdx)
								throw runtime_error("Compiler makes stupid stuff with payload");*/
							int counter = 0;
							auto start = rdma::PerfTest::startTimer(); 
							m_client->write(m_addr[connIdx], remoteOffset, (void*)arrSend, m_packet_size, true); // true=signaled
							while(true){
								tmp = m_local_memory->getUInt32(receiveOffset);
								if(tmp == receiveRootOffsetIdx && tmp > 0) break;
								if((++counter) % 100000000 == 0){ std::cout << "KILL ME, I'M FROZEN " << std::endl; }
							}
							int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
							m_sumWriteMs += time;
							if(m_minWriteMs > time) m_minWriteMs = time;
							if(m_maxWriteMs < time) m_maxWriteMs = time;
							m_arrWriteMs[i] += time;
						}
						m_arrWriteMs[i] /= m_rdma_addresses.size();
					}
					break;
				case WRITE_MODE_IMMEDIATE:
					for(size_t i = 0; i < m_iterations_per_thread; i++){
						offset = (i % m_buffer_slots) * m_packet_size;
						for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
							receiveOffset = offset * (connIdx+1) * 2;
							sendOffset = receiveOffset + m_packet_size;
							receiveRootOffsetIdx = (m_local_memory->getRootOffset() + receiveOffset) / m_packet_size * 2;
							size_t remoteOffset = m_remOffsets[connIdx] + offset;

							arrSend = m_local_memory->pointer(sendOffset);
							auto start = rdma::PerfTest::startTimer();
							m_client->receiveWriteImm(m_addr[connIdx]);
							m_client->writeImm(m_addr[connIdx], remoteOffset, (void*)arrSend, m_packet_size, receiveRootOffsetIdx, true);
							m_client->pollReceive(m_addr[connIdx], true);
							int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
							m_sumWriteMs += time;
							if(m_minWriteMs > time) m_minWriteMs = time;
							if(m_maxWriteMs < time) m_maxWriteMs = time;
							m_arrWriteMs[i] += time;
						}
						m_arrWriteMs[i] /= m_rdma_addresses.size();
					}
					break;
				default: throw invalid_argument("LatencyPerfClientThread unknown write mode"); 
			}
			break;

		case READ_OPERATION: // Read
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				offset = (i % m_buffer_slots) * m_packet_size;
				for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
					receiveOffset = offset * (connIdx+1);
					arrRecv = m_local_memory->pointer(receiveOffset);
					remoteOffset = m_remOffsets[connIdx] + offset;
					auto start = rdma::PerfTest::startTimer();
					m_client->read(m_addr[connIdx], remoteOffset, (void*)arrRecv, m_packet_size, true); // true=signaled
					int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
					m_sumReadMs += time;
					if(m_minReadMs > time) m_minReadMs = time;
					if(m_maxReadMs < time) m_maxReadMs = time;
					m_arrReadMs[i] += time;
				}
				m_arrReadMs[i] /= m_rdma_addresses.size();
			}
			break;

		case SEND_RECEIVE_OPERATION: // Send & Receive
			for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
				m_client->receive(m_addr[connIdx], m_local_memory->pointer(), m_packet_size);
			}
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				offset = (i % m_buffer_slots) * m_packet_size;
				for(size_t connIdx=0; connIdx < m_rdma_addresses.size(); connIdx++){
					receiveOffset = offset * (connIdx+1) * 2;
					sendOffset = receiveOffset + m_packet_size;

					arrSend = m_local_memory->pointer(sendOffset);
					arrRecv = m_local_memory->pointer(receiveOffset);

					auto start = rdma::PerfTest::startTimer();
					m_client->send(m_addr[connIdx], (void*)arrSend, m_packet_size, true); // true=signaled
					m_client->pollReceive(m_addr[connIdx], true); // true=poll
					m_client->receive(m_addr[connIdx], (void*)arrRecv, m_packet_size);
					int64_t time = rdma::PerfTest::stopTimer(start) / 2; // one trip time
					m_sumSendMs += time;
					if(m_minSendMs > time) m_minSendMs = time;
					if(m_maxSendMs < time) m_maxSendMs = time;
					m_arrSendMs[i] += time;
				}
				m_arrSendMs[i] /= m_rdma_addresses.size();
			}
			break;
		default: throw invalid_argument("LatencyPerfClientThread unknown test mode");
	}
}


rdma::LatencyPerfServerThread::LatencyPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t packet_size, int buffer_slots, size_t iterations_per_thread, WriteMode write_mode) {
	this->m_server = server;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size_per_thread = packet_size * buffer_slots;
	this->m_iterations_per_thread = iterations_per_thread;
	this->m_write_mode = write_mode;
	this->m_local_memory = server->localMalloc(this->m_memory_size_per_thread);
	this->m_local_memory->openContext();
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
	m_ready = false;

	const size_t receiveBaseOffset = rdma::LatencyPerfTest::thread_count * m_memory_size_per_thread;

	// receive nodeID to respond to
	if(m_respond_conn_id < 0){
		m_respond_conn_id = m_local_memory->getInt32(receiveBaseOffset);
		m_local_memory->set((int32_t)0, receiveBaseOffset);
	}

	//volatile void *arrSend = nullptr;
	volatile void *arrRecv = nullptr;
	size_t offset, sendOffset, receiveOffset, remoteOffset;
	volatile uint32_t remoteBaseOffsetIndex = 0;
	switch(LatencyPerfTest::testOperation){
		case WRITE_OPERATION: // Write
			switch(m_write_mode){
				case WRITE_MODE_NORMAL:
					m_local_memory->setMemory(0);
					for(size_t i = 0; i < m_iterations_per_thread; i++){
						const size_t flip = 1 + ((i/m_buffer_slots) % 2); // each buffer slot has alternating +1 or +0
						offset = (i % m_buffer_slots) * m_packet_size;
						receiveOffset = offset + receiveBaseOffset; // second half contains remote allocs
						int counter = 0;

						while(true){
							remoteBaseOffsetIndex = m_local_memory->getUInt32(receiveOffset);
							if((remoteBaseOffsetIndex % 2) == (flip % 2) && remoteBaseOffsetIndex > 0) break;
							if((++counter) % 100000000 == 0){ std::cout << "KILL ME, I'M FROZEN" << std::endl; }
						}
						remoteOffset = (remoteBaseOffsetIndex-flip) / 2 * m_packet_size;
						
						arrRecv = m_local_memory->pointer(receiveOffset);
						m_server->write(m_respond_conn_id, remoteOffset, (void*)arrRecv, m_packet_size, true); // true=signaled
					}
					break;
				case WRITE_MODE_IMMEDIATE:
					m_server->receiveWriteImm(m_respond_conn_id);
					for(size_t i = 1; i < m_iterations_per_thread; i++){
						sendOffset = (i % m_buffer_slots) * m_packet_size;
						receiveOffset = sendOffset + receiveBaseOffset;
						arrRecv = m_local_memory->pointer(receiveOffset);
						m_server->pollReceive(m_respond_conn_id, true, (uint32_t*)&remoteBaseOffsetIndex);
						remoteOffset = remoteBaseOffsetIndex / 2 * m_packet_size;
						m_server->receiveWriteImm(m_respond_conn_id);
						std::cout << "BEFORE: " << m_packet_size << std::endl; // TODO REMOVE
						m_server->writeImm(m_respond_conn_id, remoteOffset, (void*)arrRecv, m_packet_size, (uint32_t)0, true);
						std::cout << "AFTER" << std::endl; // TODO REMOVE
					}
					sendOffset = (m_iterations_per_thread % m_buffer_slots) * m_packet_size;
					receiveOffset = sendOffset + receiveBaseOffset;
					arrRecv = m_local_memory->pointer(receiveOffset);
					m_server->pollReceive(m_respond_conn_id, true, (uint32_t*)&remoteBaseOffsetIndex);
					remoteOffset = remoteBaseOffsetIndex / 2 * m_packet_size;
					m_server->writeImm(m_respond_conn_id, remoteOffset, (void*)arrRecv, m_packet_size, (uint32_t)0, true);
					break;
				default: break;
			}
			break;

		case READ_OPERATION:
			m_local_memory->setMemory(1);
			break;

		case SEND_RECEIVE_OPERATION: // Send & Receive
			m_server->receive(m_respond_conn_id, m_local_memory->pointer(), m_packet_size);
			for(size_t i = 0; i < m_iterations_per_thread; i++){
				sendOffset = (i % m_buffer_slots) * m_packet_size;
				receiveOffset = sendOffset + receiveBaseOffset;
				m_server->pollReceive(m_respond_conn_id, true); // true=poll
				m_server->receive(m_respond_conn_id, m_local_memory->pointer(receiveOffset), m_packet_size);
				m_server->send(m_respond_conn_id, m_local_memory->pointer(sendOffset), m_packet_size, true); // true=signaled
			}
			break;

		default: throw invalid_argument("LatencyPerfServerThread unknown test mode");
	}
}



rdma::LatencyPerfTest::LatencyPerfTest(int testOperations, bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, std::string ownIpPort, std::string sequencerIpPort, int local_gpu_index, int remote_gpu_index, int client_count, int thread_count, uint64_t packet_size, int buffer_slots, uint64_t iterations_per_thread, WriteMode write_mode) : PerfTest(testOperations){
	thread_count *= client_count;
	
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_ownIpPort = ownIpPort;
	this->m_sequencerIpPort = sequencerIpPort;
	this->m_local_gpu_index = local_gpu_index;
	this->m_actual_gpu_index = -1;
	this->m_remote_gpu_index = remote_gpu_index;
	this->client_count = client_count;
	this->thread_count = thread_count;
	this->m_packet_size = packet_size;
	this->m_buffer_slots = buffer_slots;
	this->m_memory_size = thread_count * packet_size * buffer_slots * rdma_addresses.size() * 2; // two times because send & receive
	this->m_iterations_per_thread = iterations_per_thread;
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
	if(!forCSV){ 
		oss << ", packetsize=" << m_packet_size;
		oss << ", memory=" << m_memory_size << " (2x " << thread_count << "x " << m_buffer_slots << "x ";
		if(!m_is_server){ oss << m_rdma_addresses.size() << "x "; } oss << m_packet_size << ")";
	}
	oss << ", memory_type=" << getMemoryName(m_local_gpu_index, m_actual_gpu_index) << (m_remote_gpu_index!=-404 ? "->"+getMemoryName(m_remote_gpu_index) : "");
	oss << ", iterations=" << (m_iterations_per_thread*thread_count) << ", writemode=" << (m_write_mode==WRITE_MODE_NORMAL ? "Normal" : "Immediate");
	return oss.str();
}
std::string rdma::LatencyPerfTest::getTestParameters(){
	return getTestParameters(false);
}

void rdma::LatencyPerfTest::makeThreadsReady(TestOperation testOperation){
	LatencyPerfTest::testOperation = testOperation;
	LatencyPerfTest::signaled = false;
	if(m_is_server){
		// Server
		for(LatencyPerfServerThread* perfThread : m_server_threads){ perfThread->start(); }
		for(LatencyPerfServerThread* perfThread : m_server_threads){ while(!perfThread->ready()) usleep(Config::RDMA_SLEEP_INTERVAL); }
		rdma::PerfTest::global_barrier_server(m_server, (size_t)thread_count);

	} else {
		// Client
		for(LatencyPerfClientThread* perfThread : m_client_threads){ perfThread->start(); }
		for(LatencyPerfClientThread* perfThread : m_client_threads){ while(!perfThread->ready()) usleep(Config::RDMA_SLEEP_INTERVAL); }
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
		m_server = new RDMAServer<ReliableRDMA>("LatencyTestRDMAServer", m_rdma_port, Network::getAddressOfConnection(m_ownIpPort), m_memory, m_sequencerIpPort);
		for (int thread_id = 0; thread_id < thread_count; thread_id++) {
			LatencyPerfServerThread* perfThread = new LatencyPerfServerThread(m_server, m_packet_size, m_buffer_slots, m_iterations_per_thread, m_write_mode);
			m_server_threads.push_back(perfThread);
		}

	} else {
		// Client
		for (int i = 0; i < thread_count; i++) {
			LatencyPerfClientThread* perfThread = new LatencyPerfClientThread(m_memory, m_rdma_addresses, m_ownIpPort, m_sequencerIpPort, m_packet_size, m_buffer_slots, m_iterations_per_thread, m_write_mode);
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

		// Measure Latency for writing
		if(hasTestOperation(WRITE_OPERATION)){
			makeThreadsReady(WRITE_OPERATION);
			runThreads();
		}

		// Measure Latency for receiving
		if(hasTestOperation(READ_OPERATION)){
			makeThreadsReady(READ_OPERATION); // receive
			runThreads();
		}

		// Measure Latency for receiving
		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			makeThreadsReady(SEND_RECEIVE_OPERATION); // receive
			runThreads();
		}

		// wait until server is done
		while (m_server->isRunning() && m_server->getConnectedConnIDs().size() > 0) usleep(Config::RDMA_SLEEP_INTERVAL);
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client

        // Measure Latency for writing
		if(hasTestOperation(WRITE_OPERATION)){
			makeThreadsReady(WRITE_OPERATION); // write
			usleep(Config::PERFORMANCE_TEST_SERVER_TIME_ADVANTAGE); // let server first post the receives
			runThreads();
		}

		// Measure Latency for reading
		if(hasTestOperation(READ_OPERATION)){
			makeThreadsReady(READ_OPERATION); // read
			usleep(Config::PERFORMANCE_TEST_SERVER_TIME_ADVANTAGE); // let server first make ready (shouldn't be necessary)
			runThreads();
		}

		// Measure Latency for sending
		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			makeThreadsReady(SEND_RECEIVE_OPERATION); // send
			usleep(Config::PERFORMANCE_TEST_SERVER_TIME_ADVANTAGE); // let server first post the receives
			runThreads();
		}
	}
}


std::string rdma::LatencyPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
	if(m_is_server){
		return "only client";
	} else {
		
		/*	There are  n  threads
			Each thread computes  iterations_per_thread = total_iterations / n
			Each thread takes  n  times more time compared to a single thread
			  Latency 	= elapsedTime / iterations
						= (elpasedTime_per_thread) / (n * iterations_per_thread)
		*/

		int64_t minWriteMs=std::numeric_limits<int64_t>::max(), maxWriteMs=-1, medianWriteMs=-1;
		int64_t minReadMs=std::numeric_limits<int64_t>::max(), maxReadMs=-1, medianReadMs=-1;
		int64_t minSendMs=std::numeric_limits<int64_t>::max(), maxSendMs=-1, medianSendMs=-1;
		long double avgWriteMs=0, avgReadMs=0, avgSendMs=0;
		int64_t mediansWriteNs[thread_count], mediansReadNs[thread_count], mediansSendNs[thread_count];
		const long double divAvg = m_client_threads.size() * m_iterations_per_thread * m_rdma_addresses.size(); // for calculating average
		for(size_t i=0; i<m_client_threads.size(); i++){
			LatencyPerfClientThread *thr = m_client_threads[i];
			if(minWriteMs > thr->m_minWriteMs) minWriteMs = thr->m_minWriteMs;
			if(maxWriteMs < thr->m_maxWriteMs) maxWriteMs = thr->m_maxWriteMs;
			avgWriteMs += thr->m_sumWriteMs / divAvg;
			if(minReadMs > thr->m_minReadMs) minReadMs = thr->m_minReadMs;
			if(maxReadMs < thr->m_maxReadMs) maxReadMs = thr->m_maxReadMs;
			avgReadMs += thr->m_sumReadMs / divAvg;
			if(minSendMs > thr->m_minSendMs) minSendMs = thr->m_minSendMs;
			if(maxSendMs < thr->m_maxSendMs) maxSendMs = thr->m_maxSendMs;
			avgSendMs += thr->m_sumSendMs / divAvg;

			std::sort(thr->m_arrWriteMs, thr->m_arrWriteMs + m_iterations_per_thread);
			std::sort(thr->m_arrReadMs, thr->m_arrReadMs + m_iterations_per_thread);
			std::sort(thr->m_arrSendMs, thr->m_arrSendMs + m_iterations_per_thread);
			mediansWriteNs[i] = thr->m_arrWriteMs[(int)(m_iterations_per_thread/2)];
			mediansReadNs[i] = thr->m_arrReadMs[(int)(m_iterations_per_thread/2)];
			mediansSendNs[i] = thr->m_arrSendMs[(int)(m_iterations_per_thread/2)];
		}

		std::sort(mediansWriteNs, mediansWriteNs + thread_count);
		std::sort(mediansReadNs, mediansReadNs + thread_count);
		std::sort(mediansSendNs, mediansSendNs + thread_count);
		medianWriteMs = mediansWriteNs[(int)(thread_count/2)];
		medianReadMs = mediansReadNs[(int)(thread_count/2)];
		medianSendMs = mediansSendNs[(int)(thread_count/2)];

		// write results into CSV file
		if(!csvFileName.empty()){
			const long double ustu = 1000; // nanosec to microsec
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			ofs << rdma::CSV_PRINT_NOTATION << rdma::CSV_PRINT_PRECISION;
			if(csvAddHeader){
				ofs << std::endl << "LATENCY, " << getTestParameters(true) << std::endl;
				ofs << "PacketSize [Bytes]";
				if(hasTestOperation(WRITE_OPERATION)){
					ofs << ", Avg Write [usec], Median Write [usec], Min Write [usec], Max Write [usec]";
				}
				if(hasTestOperation(READ_OPERATION)){
					ofs << ", Avg Read [usec], Median Read [usec], Min Read [usec], Max Read [usec]";
				}
				if(hasTestOperation(SEND_RECEIVE_OPERATION)){
					ofs << ", Avg Send/Recv [usec], Median Send/Recv [usec], Min Send/Recv [usec], Max Send/Recv [usec]";
				}
				ofs << std::endl;
			}
			ofs << m_packet_size; // packet size Bytes
			if(hasTestOperation(WRITE_OPERATION)){
				ofs << ", " << (round(avgWriteMs/ustu * 10)/10.0) << ", "; // avg write us
				ofs << (round(medianWriteMs/ustu * 10)/10.0) << ", "; // median write us
				ofs << (round(minWriteMs/ustu * 10)/10.0) << ", "; // min write us
				ofs << (round(maxWriteMs/ustu * 10)/10.0); // max write us
			}
			if(hasTestOperation(READ_OPERATION)){
				ofs << ", " << (round(avgReadMs/ustu * 10)/10.0) << ", "; // avg read us
				ofs << (round(medianReadMs/ustu * 10)/10.0) << ", "; // median read us
				ofs << (round(minReadMs/ustu * 10)/10.0) << ", "; // min read us
				ofs << (round(maxReadMs/ustu * 10)/10.0); // max read us
			}
			if(hasTestOperation(SEND_RECEIVE_OPERATION)){
				ofs << ", " << (round(avgSendMs/ustu * 10)/10.0) << ", "; // avg send us
				ofs << (round(medianSendMs/ustu * 10)/10.0) << ", "; // median send us
				ofs << (round(minSendMs/ustu * 10)/10.0) << ", "; // min send us
				ofs << (round(maxSendMs/ustu * 10)/10.0); // max send us
			}
			ofs << std::endl; ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		oss << rdma::CONSOLE_PRINT_NOTATION << rdma::CONSOLE_PRINT_PRECISION;
		oss << "Measured as 'one trip time' latencies:" << std::endl;
		if(hasTestOperation(WRITE_OPERATION)){
			oss << " - Write:           average = " << rdma::PerfTest::convertTime(avgWriteMs) << "    median = " << rdma::PerfTest::convertTime(medianWriteMs);
			oss << "    range = " <<  rdma::PerfTest::convertTime(minWriteMs) << " - " << rdma::PerfTest::convertTime(maxWriteMs) << std::endl;
		}
		if(hasTestOperation(READ_OPERATION)){
			oss << " - Read:            average = " << rdma::PerfTest::convertTime(avgReadMs) << "    median = " << rdma::PerfTest::convertTime(medianReadMs);
			oss << "    range = " <<  rdma::PerfTest::convertTime(minReadMs) << " - " << rdma::PerfTest::convertTime(maxReadMs) << std::endl;
		}
		if(hasTestOperation(SEND_RECEIVE_OPERATION)){
			oss << " - Send:            average = " << rdma::PerfTest::convertTime(avgSendMs) << "    median = " << rdma::PerfTest::convertTime(medianSendMs);
			oss << "    range = " <<  rdma::PerfTest::convertTime(minSendMs) << " - " << rdma::PerfTest::convertTime(maxSendMs) << std::endl;
		}
		return oss.str();
	}
	return NULL;
}
