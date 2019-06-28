/*
 * RemoteMemory_BW.cc
 *
 *  Created on: 26.11.2015
 *      Author: cbinnig
 */

#include "RemoteMemoryPerf.h"

mutex RemoteMemoryPerf::waitLock;
condition_variable RemoteMemoryPerf::waitCv;
bool RemoteMemoryPerf::signaled;

RemoteMemoryPerfThread::RemoteMemoryPerfThread(vector<string>& conns,
		size_t size, size_t iter) {
	m_size = size;
	m_iter = iter;
	m_conns = conns;
	m_remOffsets = new size_t[m_conns.size()];

	for (size_t i = 0; i < m_conns.size(); ++i) {
		ib_addr_t ibAddr;
		string conn = m_conns[i];
		if (!m_client.connect(conn, ibAddr)) {
			throw invalid_argument(
					"RemoteMemoryPerfThread connection failed");
		}
		m_addr.push_back(ibAddr);
		m_client.remoteAlloc(conn, m_size, m_remOffsets[i]);
	}

	m_data = m_client.localAlloc(m_size);
	memset(m_data, 1, m_size);
}

RemoteMemoryPerfThread::~RemoteMemoryPerfThread() {
	delete m_remOffsets;
	m_client.localFree(m_data);
	for (size_t i = 0; i < m_conns.size(); ++i) {
		string conn = m_conns[i];
		m_client.remoteFree(conn, m_remOffsets[i], m_size);
	}
}

void RemoteMemoryPerfThread::run() {
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
		m_client.write(m_conns[connIdx], m_remOffsets[connIdx], m_data, m_size,
				signaled);
	}
	endTimer();
}

RemoteMemoryPerf::RemoteMemoryPerf(config_t config, bool isClient) :
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

RemoteMemoryPerf::RemoteMemoryPerf(string& conns, size_t serverPort,
		size_t size, size_t iter, size_t threads) {
	m_conns = StringHelper::split(conns);
	m_serverPort = serverPort;
	m_size = size;
	m_iter = iter;
	m_numThreads = threads;
	RemoteMemoryPerf::signaled = false;
}

RemoteMemoryPerf::~RemoteMemoryPerf() {
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

void RemoteMemoryPerf::runServer() {
	m_dServer = new RDMAServer(m_serverPort);
	if (!m_dServer->startServer()) {
		throw invalid_argument("RemoteMemoryPerf could not start server!");
	}

	while (m_dServer->isRunning()) {
		usleep(Config::ISTORE_SLEEP_INTERVAL);
	}
}

void RemoteMemoryPerf::runClient() {
	//start all client threads
	for (size_t i = 0; i < m_numThreads; i++) {
		RemoteMemoryPerfThread* perfThread = new RemoteMemoryPerfThread(m_conns,
				m_size, m_iter);
		perfThread->start();
		if (!perfThread->ready()) {
			usleep(Config::ISTORE_SLEEP_INTERVAL);
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

double RemoteMemoryPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_threads.size(); i++) {
		totalTime += m_threads[i]->time();
	}
	return ((double) totalTime) / m_threads.size();
}

