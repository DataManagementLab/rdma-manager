/*
 * RemoteMemory_BW.cc
 *
 *  Created on: 26.11.2015
 *      Author: cbinnig
 */

#include "RemoteMemoryPerf.h"

mutex rdma::RemoteMemoryPerf::waitLock;
condition_variable rdma::RemoteMemoryPerf::waitCv;
bool rdma::RemoteMemoryPerf::signaled;

rdma::RemoteMemoryPerfThread::RemoteMemoryPerfThread(vector<string>& conns,
		size_t size, size_t iter) {
	m_size = size;
	m_iter = iter;
	m_conns = conns;
	m_remOffsets = new size_t[m_conns.size()];

	for (size_t i = 0; i < m_conns.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_conns[i];
		if (!m_client.connect(conn, nodeId)) {
			throw invalid_argument(
					"RemoteMemoryPerfThread connection failed");
		}
		m_addr.push_back(nodeId);
		m_client.remoteAlloc(conn, m_size, m_remOffsets[i]);
	}

	m_data = m_client.localAlloc(m_size);
	memset(m_data, 1, m_size);
}

rdma::RemoteMemoryPerfThread::~RemoteMemoryPerfThread() {

	m_client.localFree(m_data);

	for (size_t i = 0; i < m_conns.size(); ++i) {
		string conn = m_conns[i];
		m_client.remoteFree(conn, m_remOffsets[i], m_size);
	}
    delete m_remOffsets;

}

void rdma::RemoteMemoryPerfThread::run() {
	unique_lock < mutex > lck(RemoteMemoryPerf::waitLock);
	if (!RemoteMemoryPerf::signaled) {
		m_ready = true;
		RemoteMemoryPerf::waitCv.wait(lck);
	}
	lck.unlock();

	startTimer();
	for (size_t i = 0; i < m_iter; ++i) {
		size_t connIdx = i % m_conns.size();
		bool signaled = (i == (m_iter - 1));
		m_client.write(m_addr[connIdx],m_remOffsets[connIdx],m_data,m_size,signaled);


	}
	endTimer();
}

rdma::RemoteMemoryPerf::RemoteMemoryPerf(config_t config, bool isClient) :
		RemoteMemoryPerf(config.server, config.port, config.data, config.iter,
				config.threads) {
	this->isClient(isClient);

	//check parameters
	if (isClient && config.server.length() > 0) {
		this->isRunnable(true);
	} else if (!isClient) {
		this->isRunnable(true);
	}
}

rdma::RemoteMemoryPerf::RemoteMemoryPerf(string& conns, size_t serverPort,
		size_t size, size_t iter, size_t threads) {
	m_conns = StringHelper::split(conns);
	m_serverPort = serverPort;
	m_size = size;
	m_iter = iter;
	m_numThreads = threads;
	RemoteMemoryPerf::signaled = false;
}

rdma::RemoteMemoryPerf::~RemoteMemoryPerf() {
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

void rdma::RemoteMemoryPerf::runServer() {
	m_nodeIDSequencer = new NodeIDSequencer();
	m_nodeIDSequencer->startServer();
	
	m_dServer = new RDMAServer<ReliableRDMA>("test", m_serverPort);
	if (!m_dServer->startServer()) {
		throw invalid_argument("RemoteMemoryPerf could not start server!");
	}

	while (m_dServer->isRunning()) {
		usleep(Config::RDMA_SLEEP_INTERVAL);
	}
}

void rdma::RemoteMemoryPerf::runClient() {
	//start all client threads
	for (size_t i = 0; i < m_numThreads; i++) {
		RemoteMemoryPerfThread* perfThread = new RemoteMemoryPerfThread(m_conns,
				m_size, m_iter);
		perfThread->start();
		if (!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
		m_threads.push_back(perfThread);
	}

	//wait for user input
	waitForUser();

	//send signal to run benchmark
	RemoteMemoryPerf::signaled = false;
	unique_lock < mutex > lck(RemoteMemoryPerf::waitLock);
	RemoteMemoryPerf::waitCv.notify_all();
	RemoteMemoryPerf::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_threads.size(); i++) {
		m_threads[i]->join();
	}
}

double rdma::RemoteMemoryPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_threads.size(); i++) {
		totalTime += m_threads[i]->time();
	}
	return ((double) totalTime) / m_threads.size();
}

