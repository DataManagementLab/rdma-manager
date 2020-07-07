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
rdma::TestMode rdma::OperationsCountPerfTest::testMode;

rdma::OperationsCountPerfClientThread::OperationsCountPerfClientThread(BaseMemory *memory, std::vector<std::string>& rdma_addresses, size_t memory_size_per_thread, size_t iterations) {
	this->m_client = new RDMAClient<ReliableRDMA>(memory, "OperationsCountPerfTestClient");
	this->m_rdma_addresses = rdma_addresses;
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_iterations = iterations;
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
		m_client->remoteAlloc(conn, m_memory_size_per_thread, m_remOffsets[i]);
	}

	m_local_memory = m_client->localMalloc(m_memory_size_per_thread);
	m_local_memory->setMemory(1);
}

rdma::OperationsCountPerfClientThread::~OperationsCountPerfClientThread() {
	for (size_t i = 0; i < m_rdma_addresses.size(); ++i) {
		string addr = m_rdma_addresses[i];
		m_client->remoteFree(addr, m_remOffsets[i], m_memory_size_per_thread);
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
	auto start = rdma::PerfTest::startTimer();
	switch(OperationsCountPerfTest::testMode){
		case TEST_WRITE: // Write
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1));
				m_client->write(m_addr[connIdx], m_remOffsets[connIdx], m_local_memory->pointer(), m_memory_size_per_thread, signaled);
			}
			m_elapsedWrite = rdma::PerfTest::stopTimer(start);
			break;
		case TEST_READ: // Read
			for(size_t i = 0; i < m_iterations; i++){
				size_t connIdx = i % m_rdma_addresses.size();
				bool signaled = (i == (m_iterations - 1));
				m_client->read(m_addr[connIdx], m_remOffsets[connIdx], m_local_memory->pointer(), m_memory_size_per_thread, signaled);
			}
			m_elapsedRead = rdma::PerfTest::stopTimer(start);
			break;
		case TEST_SEND_AND_RECEIVE: // Send & Receive
			// alternating send/receive blocks to not overfill queues
			for(size_t i = 0; i < m_iterations; i++){

				size_t budgetS = m_iterations - i;
				if(budgetS > Config::RDMA_MAX_WR){ budgetS = Config::RDMA_MAX_WR; }

				size_t fi = i + budgetS;
				size_t budgetR = m_iterations - fi;
				if(budgetR > Config::RDMA_MAX_WR){ budgetR = Config::RDMA_MAX_WR; }

				for(size_t j = 0; j < budgetR; j++){
					// TODO REMOVE std::cout << "Send: " << (i+j) << std::endl; // TODO REMOVE
					m_client->receive(m_addr[(fi+j) % m_rdma_addresses.size()], m_local_memory->pointer(), m_memory_size_per_thread);
				}

				for(size_t j = 0; j < budgetS; j++){
					// TODO REMOVE std::cout << "Send: " << (i+j) << std::endl; // TODO REMOVE
					m_client->send(m_addr[(i+j) % m_rdma_addresses.size()], m_local_memory->pointer(), m_memory_size_per_thread, (j+1)==budgetS); // signaled: (j+1)==budget
				}

				for(size_t j = 0; j < budgetR; j++){
					m_client->pollReceive(m_addr[(fi+j) % m_rdma_addresses.size()], true);
				}

				i += budgetS + budgetR;
			}
			m_elapsedSend = rdma::PerfTest::stopTimer(start);
			break;
		default: throw invalid_argument("OperationsCountPerfClientThread unknown test mode");
	}
}



rdma::OperationsCountPerfServerThread::OperationsCountPerfServerThread(RDMAServer<ReliableRDMA> *server, size_t memory_size_per_thread, size_t iterations) {
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_iterations = iterations;
	this->m_server = server;
	this->m_local_memory = server->localMalloc(memory_size_per_thread);
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
	const std::vector<size_t> clientIds = m_server->getConnectedConnIDs();

	// Measure operations/s for receiving
	//auto start = rdma::PerfTest::startTimer();
	for(size_t i = 0; i < m_iterations; i++){

		size_t budgetR = m_iterations - i;
		if(budgetR > Config::RDMA_MAX_WR){ budgetR = Config::RDMA_MAX_WR; }

		size_t fi = i + budgetR;
		size_t budgetS = m_iterations - fi;
		if(budgetS > Config::RDMA_MAX_WR){ budgetS = Config::RDMA_MAX_WR; }

		for(size_t j = 0; j < budgetR; j++){
			// TODO REMOVE std::cout << "Send: " << (i+j) << std::endl; // TODO REMOVE
			m_server->receive(clientIds[(i+j) % clientIds.size()], m_local_memory->pointer(), m_memory_size_per_thread);
		}

		for(size_t j = 0; j < budgetR; j++){
			m_server->pollReceive(clientIds[(i+j) % clientIds.size()], true);
		}

		for(size_t j = 0; j < budgetS; j++){
			// TODO REMOVE std::cout << "Send: " << (i+j) << std::endl; // TODO REMOVE
			m_server->send(clientIds[(fi+j) % clientIds.size()], m_local_memory->pointer(), m_memory_size_per_thread, (j+1)==budgetS); // signaled: (j+1)==budget
		}

		i += budgetR + budgetS;
	}
	//m_elapsedReceive = rdma::PerfTest::stopTimer(start);
}



rdma::OperationsCountPerfTest::OperationsCountPerfTest(bool is_server, std::vector<std::string> rdma_addresses, int rdma_port, int gpu_index, int thread_count, uint64_t memory_size_per_thread, uint64_t iterations) : PerfTest(){
	this->m_is_server = is_server;
	this->m_rdma_port = rdma_port;
	this->m_gpu_index = gpu_index;
	this->m_thread_count = thread_count;
	this->m_memory_size_per_thread = memory_size_per_thread;
	this->m_memory_size = 2 * thread_count * memory_size_per_thread; // 2x because for send & receive separat
	this->m_iterations = iterations;
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

std::string rdma::OperationsCountPerfTest::getTestParameters(){
	std::ostringstream oss;
	if(m_is_server){
		oss << "Server, memory=";
	} else {
		oss << "Client, threads=" << m_thread_count << ", memory=";
	}
	oss << m_memory_size << " (2x " << m_thread_count << "x " << m_memory_size_per_thread << ") [";
	if(m_gpu_index < 0){
		oss << "MAIN";
	} else {
		oss << "GPU." << m_gpu_index; 
	}
	oss << " mem], iterations=" << m_iterations;
	return oss.str();
}

void rdma::OperationsCountPerfTest::makeThreadsReady(TestMode testMode){
	OperationsCountPerfTest::testMode = testMode;
	OperationsCountPerfTest::signaled = false;
	for(OperationsCountPerfServerThread* perfThread : m_server_threads){
		perfThread->start();
		while(!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
	}
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
	#ifdef CUDA_ENABLED /* defined in CMakeLists.txt to globally enable/disable CUDA support */
		m_memory = (m_gpu_index<0 ? (rdma::BaseMemory*)new rdma::MainMemory(m_memory_size) : (rdma::BaseMemory*)new rdma::CudaMemory(m_memory_size, m_gpu_index));
	#else
		m_memory = (rdma::BaseMemory*)new MainMemory(m_memory_size);
	#endif

	if(m_is_server){
		// Server
		m_server = new RDMAServer<ReliableRDMA>("OperationsCountTestRDMAServer", m_rdma_port, m_memory);
		for (int i = 0; i < m_thread_count; i++) {
			OperationsCountPerfServerThread* perfThread = new OperationsCountPerfServerThread(m_server, m_memory_size_per_thread, m_iterations);
			m_server_threads.push_back(perfThread);
		}
		/* If server only allows to be single threaded
		OperationsCountPerfServerThread* perfThread = new OperationsCountPerfServerThread(m_server, m_memory_size_per_thread*m_thread_count, m_iterations*m_thread_count);
		m_server_threads.push_back(perfThread); */

	} else {
		// Client
		for (int i = 0; i < m_thread_count; i++) {
			OperationsCountPerfClientThread* perfThread = new OperationsCountPerfClientThread(m_memory, m_rdma_addresses, m_memory_size_per_thread, m_iterations);
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
		
		makeThreadsReady(TEST_SEND_AND_RECEIVE); // receive
		//auto startReceive = rdma::PerfTest::startTimer();
        runThreads();
		//m_elapsedReceive = rdma::PerfTest::stopTimer(startReceive);

		// wait until server is done
		while (m_server->isRunning() && m_server->getConnectedConnIDs().size() > 0) usleep(Config::RDMA_SLEEP_INTERVAL);
		std::cout << "Server stopped" << std::endl;

	} else {
		// Client


        // Measure operations/s for writing
		makeThreadsReady(TEST_WRITE); // write
		auto startWrite = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedWrite = rdma::PerfTest::stopTimer(startWrite);

		// Measure operations/s for reading
		makeThreadsReady(TEST_READ); // read
		auto startRead = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedRead = rdma::PerfTest::stopTimer(startRead);

		// Measure operations/s for sending
		makeThreadsReady(TEST_SEND_AND_RECEIVE); // send
		usleep(2 * Config::RDMA_SLEEP_INTERVAL); // let server first post the receives
		auto startSend = rdma::PerfTest::startTimer();
        runThreads();
		m_elapsedSend = rdma::PerfTest::stopTimer(startSend);
	}
}


std::string rdma::OperationsCountPerfTest::getTestResults(std::string csvFileName, bool csvAddHeader){
	if(m_is_server){
		return "only client";
	} else {

		const long double tu = (long double)NANO_SEC; // 1sec (nano to seconds as time unit)
        const long double itrs = (long double)m_iterations, totalItrs = itrs*m_thread_count;

		int64_t maxWrite=-1, minWrite=std::numeric_limits<int64_t>::max();
		int64_t maxRead=-1, minRead=std::numeric_limits<int64_t>::max();
		int64_t maxSend=-1, minSend=std::numeric_limits<int64_t>::max();
		int64_t arrWrite[m_thread_count];
		int64_t arrRead[m_thread_count];
		int64_t arrSend[m_thread_count];
		long double avgWrite=0, medianWrite, avgRead=0, medianRead, avgSend=0, medianSend;

		for(size_t i=0; i<m_client_threads.size(); i++){
			OperationsCountPerfClientThread *thr = m_client_threads[i];
			if(thr->m_elapsedWrite < minWrite) minWrite = thr->m_elapsedWrite;
			if(thr->m_elapsedWrite > maxWrite) maxWrite = thr->m_elapsedWrite;
			avgWrite += (long double) thr->m_elapsedWrite;
			arrWrite[i] = thr->m_elapsedWrite;
			if(thr->m_elapsedRead < minRead) minRead = thr->m_elapsedRead;
			if(thr->m_elapsedRead > maxRead) maxRead = thr->m_elapsedRead;
			avgRead += (long double) thr->m_elapsedRead;
			arrRead[i] = thr->m_elapsedRead;
			if(thr->m_elapsedSend < minSend) minSend = thr->m_elapsedSend;
			if(thr->m_elapsedSend > maxSend) maxSend = thr->m_elapsedSend;
			avgSend += (long double) thr->m_elapsedSend;
			arrSend[i] = thr->m_elapsedSend;
		}
		avgWrite /= (long double) m_thread_count;
		avgRead /= (long double) m_thread_count;
		avgSend /= (long double) m_thread_count;
		std::sort(arrWrite, arrWrite + m_thread_count);
		std::sort(arrRead, arrRead + m_thread_count);
		std::sort(arrSend, arrSend + m_thread_count);
		medianWrite = arrWrite[(int)(m_thread_count/2)];
		medianRead = arrRead[(int)(m_thread_count/2)];
		medianSend = arrSend[(int)(m_thread_count/2)];

		// write results into CSV file
		if(!csvFileName.empty()){
			const uint64_t su = 1000*1000; // size unit (operations to megaOps)
			std::ofstream ofs;
			ofs.open(csvFileName, std::ofstream::out | std::ofstream::app);
			if(csvAddHeader){
				ofs << std::endl << "OPERATIONS PER SECOND, " << getTestParameters() << std::endl;
				ofs << "PacketSize [Bytes], Write [megaOp/s], Read [megaOp/s], Send/Recv [megaOp/s], ";
				ofs << "Min Write [megaOp/s], Min Read [megaOp/s], Min Send/Recv [megaOp/s], ";
				ofs << "Max Write [megaOp/s], Max Read [megaOp/s], Max Send/Recv [megaOp/s], ";
				ofs << "Avg Write [megaOp/s], Avg Read [megaOp/s], Avg Send/Recv [megaOp/s], ";
				ofs << "Median Write [megaOp/s], Median Read [megaOp/s], Median Send/Recv [megaOp/s], ";
				ofs << "Write [Sec], Read [Sec], Send/Recv [Sec], ";
				ofs << "Min Write [Sec], Min Read [Sec], Min Send/Recv [Sec], ";
				ofs << "Max Write [Sec], Max Read [Sec], Max Send/Recv [Sec], ";
				ofs << "Avg Write [Sec], Avg Read [Sec], Avg Send/Recv [Sec], ";
				ofs << "Median Write [Sec], Median Read [Sec], Median Send/Recv [Sec]" << std::endl;
			}
			ofs << m_memory_size_per_thread << ", "; // packet size Bytes
			ofs << (round(totalItrs*tu/su/m_elapsedWrite * 100000)/100000.0) << ", "; // write Op/s
			ofs << (round(totalItrs*tu/su/m_elapsedRead * 100000)/100000.0) << ", "; // read Op/s
			ofs << (round(totalItrs*tu/su/m_elapsedSend * 100000)/100000.0) << ", "; // send/recv Op/s
			ofs << (round(itrs*tu/su/maxWrite * 100000)/100000.0) << ", "; // min write Op/s
			ofs << (round(itrs*tu/su/maxRead * 100000)/100000.0) << ", "; // min read Op/s
			ofs << (round(itrs*tu/su/maxSend * 100000)/100000.0) << ", "; // min send/recv Op/s
			ofs << (round(itrs*tu/su/minWrite * 100000)/100000.0) << ", "; // max write Op/s
			ofs << (round(itrs*tu/su/minRead * 100000)/100000.0) << ", "; // max read Op/s
			ofs << (round(itrs*tu/su/minSend * 100000)/100000.0) << ", "; // max send/recv Op/s
			ofs << (round(itrs*tu/su/avgWrite * 100000)/100000.0) << ", "; // avg write Op/s
			ofs << (round(itrs*tu/su/avgRead * 100000)/100000.0) << ", "; // avg read Op/s
			ofs << (round(itrs*tu/su/avgSend * 100000)/100000.0) << ", "; // avg send/recv Op/s
			ofs << (round(itrs*tu/su/medianWrite * 100000)/100000.0) << ", "; // median write Op/s
			ofs << (round(itrs*tu/su/medianRead * 100000)/100000.0) << ", "; // median read Op/s
			ofs << (round(itrs*tu/su/medianSend * 100000)/100000.0) << ", "; // median send/recv Op/s
			ofs << (round(m_elapsedWrite/tu * 100000)/100000.0) << ", " << (round(m_elapsedRead/tu * 100000)/100000.0) << ", "; // write, read Sec
			ofs << (round(m_elapsedSend/tu * 100000)/100000.0) << ", "; // send Sec
			ofs << (round(minWrite/tu * 100000)/100000.0) << ", " << (round(minRead/tu * 100000)/100000.0) << ", "; // min write, read Sec
			ofs << (round(minSend/tu * 100000)/100000.0) << ", "; // min send Sec
			ofs << (round(maxWrite/tu * 100000)/100000.0) << ", " << (round(maxRead/tu * 100000)/100000.0) << ", "; // max write, read Sec
			ofs << (round(maxSend/tu * 100000)/100000.0) << ", "; // max send Sec
			ofs << (round(avgWrite/tu * 100000)/100000.0) << ", " << (round(avgRead/tu * 100000)/100000.0) << ", "; // avg write, read Sec
			ofs << (round(avgSend/tu * 100000)/100000.0) << ", "; // avg send Sec
			ofs << (round(medianWrite/tu * 100000)/100000.0) << ", " << (round(medianRead/tu * 100000)/100000.0) << ", "; // median write, read Sec
			ofs << (round(medianSend/tu * 100000)/100000.0) << std::endl; // median send Sec
			ofs.close();
		}

		// generate result string
		std::ostringstream oss;
		oss << " measurement for sending is executed as alternating send/receive bursts of size " << Config::RDMA_MAX_WR << std::endl;
		oss << " - Write:         operations = " << rdma::PerfTest::convertCountPerSec(totalItrs*tu/m_elapsedWrite); 
		oss << "  (range = " << rdma::PerfTest::convertCountPerSec(itrs*tu/maxWrite) << " - " << rdma::PerfTest::convertCountPerSec(itrs*tu/minWrite);
		oss << " ; avg=" << rdma::PerfTest::convertCountPerSec(itrs*tu/avgWrite) << " ; median=";
		oss << rdma::PerfTest::convertCountPerSec(itrs*tu/minWrite) << ")";
		oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedWrite) << "  (range=";
		oss << rdma::PerfTest::convertTime(minWrite) << "-" << rdma::PerfTest::convertTime(maxWrite);
		oss << " ; avg=" << rdma::PerfTest::convertTime(avgWrite) << " ; median=" << rdma::PerfTest::convertTime(medianWrite) << ")" << std::endl;
		oss << " - Read:          operations = " << rdma::PerfTest::convertCountPerSec(totalItrs*tu/m_elapsedRead);
		oss << "  (range = " << rdma::PerfTest::convertCountPerSec(itrs*tu/maxRead) << " - " << rdma::PerfTest::convertCountPerSec(itrs*tu/minRead);
		oss << ", avg=" << rdma::PerfTest::convertCountPerSec(itrs*tu/avgRead) << " ; median=";
		oss << rdma::PerfTest::convertCountPerSec(itrs*tu/minRead) << ")";
		oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedRead) << "  (range=";
		oss << rdma::PerfTest::convertTime(minRead) << "-" << rdma::PerfTest::convertTime(maxRead);
		oss << " ; avg=" << rdma::PerfTest::convertTime(avgRead) << " ; median=" << rdma::PerfTest::convertTime(medianRead) << ")" << std::endl;
		oss << " - Send:          operations = " << rdma::PerfTest::convertCountPerSec(totalItrs*tu/m_elapsedSend);
		oss << "  (range = " << rdma::PerfTest::convertCountPerSec(itrs*tu/maxSend) << " - " << rdma::PerfTest::convertCountPerSec(itrs*tu/minSend);
		oss << " ; avg=" << rdma::PerfTest::convertCountPerSec(itrs*tu/avgSend) << " ; median=";
		oss << rdma::PerfTest::convertCountPerSec(itrs*tu/minSend) << ")";
		oss << "   &   time = " << rdma::PerfTest::convertTime(m_elapsedSend) << "  (range=";
		oss << rdma::PerfTest::convertTime(minSend) << "-" << rdma::PerfTest::convertTime(maxSend);
		oss << " ; avg=" << rdma::PerfTest::convertTime(avgSend) << " ; median=" << rdma::PerfTest::convertTime(medianSend) << ")" << std::endl;
		return oss.str();

	}
	return NULL;
}