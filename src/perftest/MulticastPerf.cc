/*
 * Multicast.cc
 *
 *  Created on: Dec 3, 2016
 *      Author: cbinnig
 */

#include "MulticastPerf.h"
#include "../utils/Timer.h"

mutex MulticastPerf::waitLock;
condition_variable MulticastPerf::waitCv;
bool MulticastPerf::signaled;

/********* client threads *********/
MCClientPerfThread::MCClientPerfThread(string mcastGroup, size_t size,
		size_t iter, size_t budget) {
	//init from parameters
	m_size = size;
	m_iter = iter;
	m_budget = budget;

	//join multicast group
	m_client = new RDMAClient();
	if (!m_client->joinMCastGroup(mcastGroup, m_mcastConn)) {
		throw invalid_argument(
				"MCClientPerfThread could not join multicast group");
	}

	//connect clients to all servers
	for (size_t i = 0; i < Config::PTEST_MCAST_NODES.size(); i++) {
		ib_addr_t ibAddr;
		string conn = Config::PTEST_MCAST_NODES[i];
		if (!m_client->connect(conn, ibAddr)) {
			throw invalid_argument("MCClientPerfThread connection failed");
			;
		}
		m_serverConns.push_back(ibAddr);
	}

	//allocate memory for data and signals
	m_data = m_client->localAlloc(m_size * m_budget);
	m_signal =
			(int*) (m_client->localAlloc(sizeof(int) * m_serverConns.size()));
	if (m_data == nullptr || m_signal == nullptr) {
		throw invalid_argument("MCClientPerfThread could not allocate memory");
	}
}

MCClientPerfThread::~MCClientPerfThread() {
	//free memory
	m_client->localFree(m_data);
	m_client->localFree(m_signal);
}

void MCClientPerfThread::run() {
	//wait for signal from perftest to start running
	unique_lock < mutex > lck(MulticastPerf::waitLock);
	m_ready = true;
	if (!MulticastPerf::signaled) {
		MulticastPerf::waitCv.wait(lck);
	}
	lck.unlock();

	//prepare perftest to receive signals
	for (size_t i = 0; i < m_serverConns.size(); ++i) {
		if (!m_client->receive(m_serverConns[i], &m_signal[i], sizeof(int))) {
			cout << " Failed!" << endl << flush;
			return;
		}
	}

	//multicast perftest
	startTimer();
	uint128_t startTime = Timer::timestamp();
	for (size_t iter = 0; iter < m_iter; ++iter) {
		//send mcast data
		for (size_t i = 0; i < m_budget; ++i) {
			void* data = (void*) ((char*) m_data + i * m_size);
			if (!m_client->sendMCast(m_mcastConn, data, m_size,
					(i + 1 == m_budget))) {
				cout << " Failed!" << endl << flush;
				return;
			}
		}

		//poll for receives for signals
		for (size_t i = 0; i < m_serverConns.size(); ++i) {
			if (!m_client->pollReceive(m_serverConns[i])) {
				cout << " Failed!" << endl << flush;
				return;
			}
		}

		//prepare receiving signals for next iteration
		for (size_t i = 0; i < m_serverConns.size(); ++i) {
			if (!m_client->receive(m_serverConns[i], &m_signal[i],
					sizeof(int))) {
				cout << " Failed!" << endl << flush;
				return;
			}
		}
	}
	endTimer();
	uint128_t endTime = Timer::timestamp();
	resultedTime = endTime - startTime;
	cout << "resulted time:" << resultedTime << endl;
}

/********* server threads *********/
MCServerPerfThread::MCServerPerfThread(string mcastGroup, size_t serverPort,
		size_t size, size_t iter, size_t budget, size_t numThreads) {
	//init from parameters
	m_size = size;
	m_iter = iter;
	m_budget = budget;
	m_numThreads = numThreads;

	//start server
	m_server = new RDMAServer(serverPort);
	if (!m_server->startServer()) {
		throw invalid_argument("MCServerPerfThread could not start RDMAServer");
	}

	//join mcast group
	if (!m_server->joinMCastGroup(mcastGroup, m_mcastConn)) {
		throw invalid_argument(
				"MCServerPerfThread could not join multicast group");
	}
	//allocate memory
	m_data = m_server->localAlloc(size * m_budget);
	m_signal = (int*) (m_server->localAlloc(sizeof(int) * m_numThreads));
	if (m_data == nullptr || m_signal == nullptr) {
		throw invalid_argument("MCServerPerfThread could not allocate memory");
	}
}

MCServerPerfThread::~MCServerPerfThread() {
	//free memory
	m_server->localFree(m_data);
	m_server->localFree(m_signal);
}

void MCServerPerfThread::run() {
	//server thread is ready and running
	m_ready = true;

	//prepare mcast for receiving data
	for (size_t i = 0; i < m_budget; ++i) {
		void* data = (void*) ((char*) m_data + m_size);
		if (!m_server->receiveMCast(m_mcastConn, data, m_size)) {
			cout << " Failed!" << endl << flush;
			return;
		}
	}

	//wait for client to connects
	vector<ib_addr_t> clientConns = m_server->getQueues();
	while (clientConns.size() < m_numThreads) {
		clientConns = m_server->getQueues();
	}

	//start multicast perftest
	startTimer();
	for (size_t iter = 0; iter < m_iter; ++iter) {
		//receive mcast data
		for (size_t i = 0; i < m_budget; ++i) {
			if (!m_server->pollReceiveMCast(m_mcastConn)) {
				cout << " Failed!" << endl << flush;
				return;
			}
		}
		//prepare receiving data for next round
		for (size_t i = 0; i < m_budget; ++i) {
			void* data = (void*) ((char*) m_data + m_size);
			if (!m_server->receiveMCast(m_mcastConn, data, m_size)) {
				cout << " Failed!" << endl << flush;
				return;
			}
		}
		// send signal to clients to start next iteration
		size_t numConns = clientConns.size();
		for (size_t i = 0; i < numConns; ++i) {
			if (!m_server->send(clientConns[i], &m_signal[i], sizeof(int),
					((i + 1) == numConns))) {
				cout << " Failed!" << endl << flush;
				return;
			}
		}
	}
	endTimer();
}

MulticastPerf::MulticastPerf(config_t config, bool isClient) :
		MulticastPerf(config.server, config.port, config.data, config.iter,
				config.threads) {
	this->isClient(isClient);

	//check parameters
	if (config.server.length() > 0) {
		this->isRunnable(true);
	}
}

MulticastPerf::MulticastPerf(string& group, size_t serverPort, size_t size,
		size_t iter, size_t threads) {
	m_group = group;
	m_serverPort = serverPort;
	m_size = size;
	m_budget = Config::RDMA_MAX_WR / threads;
	m_iter = iter / m_budget;
	m_client = nullptr;
	m_server = nullptr;
	m_numThreads = threads;
	MulticastPerf::signaled = false;
}

MulticastPerf::~MulticastPerf() {
	if (m_client != nullptr) {
		delete m_client;
	} else if (m_server != nullptr) {
		m_server->stopServer();
		delete m_server;
	}
}

void MulticastPerf::runServer() {
	//start server thread
	MCServerPerfThread* perfThread = new MCServerPerfThread(m_group,
			m_serverPort, m_size, m_iter, m_budget * m_numThreads,
			m_numThreads);
	perfThread->start();
	if (!perfThread->ready()) {
		usleep(Config::ISTORE_SLEEP_INTERVAL);
	}
	m_sthread = perfThread;

	//wait for server thread to finish
	m_sthread->join();
	delete m_sthread;
}

void MulticastPerf::runClient() {
	//prepare all client threads
	for (size_t i = 0; i < m_numThreads; i++) {
		MCClientPerfThread* perfThread = new MCClientPerfThread(m_group, m_size,
				m_iter, m_budget);
		perfThread->start();
		if (!perfThread->ready()) {
			usleep(Config::ISTORE_SLEEP_INTERVAL);
		}
		m_cthreads.push_back(perfThread);
	}

	//simultaneously start all clients
	MulticastPerf::signaled = false;
	unique_lock < mutex > lck(MulticastPerf::waitLock);
	MulticastPerf::waitCv.notify_all();
	MulticastPerf::signaled = true;
	lck.unlock();

	//wait for clients threads to finish
	for (size_t i = 0; i < m_numThreads; i++) {
		m_cthreads[i]->join();
		delete m_cthreads[i];
	}
}

double MulticastPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_cthreads.size(); i++) {
		totalTime += m_cthreads[i]->resultedTime;
	}
	return ((double) totalTime) / m_cthreads.size();
}
