/*
 * FetchAndAddPerf.cc
 *
 *  Created on: 01.03.2020
 *      Author: lthostrup
 */

#include "FetchAndAddPerf.h"
#include "PerfEvent.hpp"

mutex rdma::FetchAndAddPerf::waitLock;
condition_variable rdma::FetchAndAddPerf::waitCv;
bool rdma::FetchAndAddPerf::signaled;

rdma::FetchAndAddPerfThread::FetchAndAddPerfThread(std::string &conns, size_t iter) {
	m_iter = iter;
	m_conns = conns;

	//ib_addr_t ibAddr;
	if (!m_client.connect(m_conns, m_serverID)) {
		throw invalid_argument("FetchAndAddPerfThread connection failed");
	}

	m_localCounter = reinterpret_cast<size_t*>(m_client.localAlloc(Config::CACHELINE_SIZE));
}

rdma::FetchAndAddPerfThread::~FetchAndAddPerfThread() {
}

void rdma::FetchAndAddPerfThread::run() {
	unique_lock <mutex> lck(FetchAndAddPerf::waitLock);
	if (!FetchAndAddPerf::signaled) {
		m_ready = true;
		FetchAndAddPerf::waitCv.wait(lck);
	}
	lck.unlock();

	{
		std::cout << "Perf Measurement enabled" << std::endl;
		// PerfEventBlock pb(m_iter);
		startTimer();
		for (size_t i = 0; i < m_iter; ++i) {
			m_client.fetchAndAdd(m_serverID, 0, m_localCounter, sizeof(size_t), true);
		}
		endTimer();
	}	

}

rdma::FetchAndAddPerf::FetchAndAddPerf(config_t config, bool isClient) :
		FetchAndAddPerf(config.server, config.port, config.iter, config.threads) {
	this->isClient(isClient);

	//check parameters
	if (isClient && config.server.length() > 0) {
		this->isRunnable(true);
	} else if (!isClient) {
		this->isRunnable(true);
	}
}

rdma::FetchAndAddPerf::FetchAndAddPerf(std::string& conn, size_t serverPort, size_t iter, size_t threads) {
	m_conn = conn + ":" + to_string(serverPort);
	m_serverPort = serverPort;
	m_iter = iter;
	m_numThreads = threads;
	FetchAndAddPerf::signaled = false;
}

rdma::FetchAndAddPerf::~FetchAndAddPerf() {
	if (this->isClient()) {
		for (size_t i = 0; i < m_threads.size(); i++) {
			delete m_threads[i];
		}
		m_threads.clear();
	}
}

void rdma::FetchAndAddPerf::runServer() {
	m_nodeIDSequencer = new NodeIDSequencer();
	
	m_dServer = new RDMAServer<ReliableRDMA>("test", m_serverPort);
	m_dServer->startServer();
	auto ptr = m_dServer->localAlloc(rdma::Config::CACHELINE_SIZE * 2);

	*reinterpret_cast<uint64_t*>(ptr) = 0;

	// size_t space = rdma::Config::CACHELINE_SIZE * 2;
	// auto newptr = cache_align(sizeof(size_t), ptr, space);
	// if (ptr == nullptr)
	// {
	// 	std::cout << "Error allocating atomic counter!" << std::endl;
	// 	exit(1);
	// }
	// std::cout << "Server ready! counter offset: " << (char*)newptr - (char*)ptr << std::endl;

	while (true) {
		usleep(Config::RDMA_SLEEP_INTERVAL);
	}
}

void rdma::FetchAndAddPerf::runClient() {
	//start all client threads
	for (size_t i = 0; i < m_numThreads; i++) {
		FetchAndAddPerfThread* perfThread = new FetchAndAddPerfThread(m_conn, m_iter);
		perfThread->start(rdma::Config::NUMA_THREAD_CPUS[rdma::Config::RDMA_NUMAREGION][i]);
		std::cout << "Starting thread on core: " << rdma::Config::NUMA_THREAD_CPUS[rdma::Config::RDMA_NUMAREGION][i] << std::endl;
		while (!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
		m_threads.push_back(perfThread);
	}

	//wait for user input
	waitForUser();

	//send signal to run benchmark
	FetchAndAddPerf::signaled = false;
	unique_lock < mutex > lck(FetchAndAddPerf::waitLock);
	FetchAndAddPerf::waitCv.notify_all();
	FetchAndAddPerf::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_threads.size(); i++) {
		m_threads[i]->join();
	}
}

double rdma::FetchAndAddPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_threads.size(); i++) {
		totalTime += m_threads[i]->time();
	}
	return ((double) totalTime) / m_threads.size();
}

