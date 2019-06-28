/*
 * ScanMemory_BW.cc
 *
 *  Created on: 26.11.2015
 *      Author: cbinnig
 */

#include "RemoteScanPerf.h"

mutex RemoteScanPerf::waitLock;
condition_variable RemoteScanPerf::waitCv;
bool RemoteScanPerf::signaled;

RemoteScanPerfThread::RemoteScanPerfThread(string& conn, size_t size,
		size_t threadId, size_t numThreads) {
	uint64_t value = 0;
	m_pageSize = size;
	m_tuplesPerPage = size / sizeof(value);
	m_prefetchCount = Config::PTEST_SCAN_PREFETCH;

	size_t totalPages = Config::RDMA_MEMSIZE / m_pageSize;
	size_t pagesPerThread = totalPages / numThreads;

	m_startIdx = pagesPerThread * threadId;
	m_readIdx = m_startIdx;
	m_remReadIdx = m_startIdx;
	m_writeIdx = m_startIdx;
	m_endIdx = m_startIdx + pagesPerThread;

	//connect to remote memory server
	if (!m_client.connect(conn, m_addr)) {
		throw invalid_argument("RemoteScanPerfThread connection failed");
	}

	//allocate local window
	m_data = (char*) m_client.localAlloc(m_pageSize * m_prefetchCount);
	memset((void*) m_data, 0, m_pageSize * m_prefetchCount);
}

RemoteScanPerfThread::~RemoteScanPerfThread() {
	//free local memory
	m_client.localFree(m_data);
}

void RemoteScanPerfThread::run() {
	unique_lock < mutex > lck(RemoteScanPerf::waitLock);
	if (!RemoteScanPerf::signaled) {
		m_ready = true;
		RemoteScanPerf::waitCv.wait(lck);
	}
	lck.unlock();

	//scan remote table
	startTimer();

	//prefetch pages
	if (!prefetch()) {
		return;
	}

	size_t total = 0;
	while (m_readIdx < m_endIdx) {
		//wait for next page
		if (!m_client.pollSend(m_addr)) {
			return;
		}
		
		//read values in page
		uint64_t* localReadPtr = (uint64_t*) (m_data
				+ ((m_readIdx % m_prefetchCount) * m_pageSize));
		for (size_t i = 0; i < m_tuplesPerPage; ++i) {
			total += localReadPtr[i];
		}
		m_readIdx++;

		//prefetch next page
		if (!prefetch()) {
			return;
		}
	}

	endTimer();
}

RemoteScanPerf::RemoteScanPerf(config_t config, bool isClient) :
		RemoteScanPerf(config.server, config.port, config.data, config.iter,
				config.threads) {
	this->isClient(isClient);

	//check parameters
	if (isClient && config.server.length() > 0) {
		this->isRunnable(true);
	} else if (!isClient) {
		this->isRunnable(true);
	}
}

RemoteScanPerf::RemoteScanPerf(string& conn, size_t serverPort, size_t size,
		size_t iter, size_t threads) {
	m_conn = conn;
	m_serverPort = serverPort;
	m_size = size;
	m_iter = iter;
	m_numThreads = threads;
	RemoteScanPerf::signaled = false;

	if(Config::PTEST_SCAN_PREFETCH<=0){
		throw invalid_argument("PTEST_SCAN_PREFETCH must be greater than 0");
	}
}

RemoteScanPerf::~RemoteScanPerf() {

	if (this->isClient()) {
		for (size_t i = 0; i < m_threads.size(); i++) {
			delete m_threads[i];
		}
		m_threads.clear();
	} else {
		if (m_dServer != nullptr) {
			m_dServer->stopServer();
			delete m_dServer;
		}
		m_dServer = nullptr;
	}
}

void RemoteScanPerf::runServer() {
	m_dServer = new RDMAServer(m_serverPort);

	//fill data
	uint64_t* table = (uint64_t*) m_dServer->getBuffer();
	size_t tableSize = Config::RDMA_MEMSIZE / sizeof(uint64_t);
	for (size_t i = 0; i < tableSize; ++i) {
		table[i] = 1;
	}

	//start server
	if (!m_dServer->startServer()) {
		throw invalid_argument("RemoteScanPerf could not start server!");
	}

	while (m_dServer->isRunning()) {
		usleep(Config::ISTORE_SLEEP_INTERVAL);
	}
}

void RemoteScanPerf::runClient() {
	//start all client threads
	for (size_t i = 0; i < m_numThreads; i++) {
		RemoteScanPerfThread* perfThread = new RemoteScanPerfThread(m_conn,
				m_size, i, m_numThreads);
		perfThread->start();
		if (!perfThread->ready()) {
			usleep(Config::ISTORE_SLEEP_INTERVAL);
		}
		m_threads.push_back(perfThread);
	}

	//wait for user input
	cout << "Press Enter to run Benchmark!" << flush << endl;
	char temp;
	cin.get(temp);

	//send signal to run benchmark
	RemoteScanPerf::signaled = false;
	unique_lock < mutex > lck(RemoteScanPerf::waitLock);
	RemoteScanPerf::waitCv.notify_all();
	RemoteScanPerf::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_threads.size(); i++) {
		m_threads[i]->join();
	}
}

double RemoteScanPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_threads.size(); i++) {
		totalTime += m_threads[i]->time();
	}
	return ((double) totalTime) / m_threads.size();
}

