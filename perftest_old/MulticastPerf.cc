/*
 * Multicast.cc
 *
 *  Created on: Dec 3, 2016
 *      Author: cbinnig
 */

#include "MulticastPerf.h"
#include "../utils/Timer.h"


mutex rdma::MulticastPerf::waitLock;
condition_variable rdma::MulticastPerf::waitCv;
bool rdma::MulticastPerf::signaled;

/********* client threads *********/
rdma::MCClientPerfThread::MCClientPerfThread(int threadid, string mcastGroup, vector<string> servers, size_t size,
		size_t iter, size_t budget) {
	//init from parameters
	m_size = size;
	m_iter = iter;
	m_budget = budget;
	m_threadid = threadid;
	//join multicast group
	m_client = new RDMAClient<UnreliableRDMA>();
	m_client->joinMCastGroup(mcastGroup, m_mcastConn);

	//connect clients to all servers
	for (size_t i = 0; i < servers.size(); i++) {
		NodeID ibAddr;
		string conn = servers[i];
		if (!m_client->connect(conn, ibAddr)) {
			throw invalid_argument("MCClientPerfThread connection failed");
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

rdma::MCClientPerfThread::~MCClientPerfThread() {
	//free memory
	m_client->localFree(m_data);
	m_client->localFree(m_signal);
}

void rdma::MCClientPerfThread::run() {
	//wait for signal from perftest to start running
	unique_lock < mutex > lck(MulticastPerf::waitLock);
	m_ready = true;
	if (!MulticastPerf::signaled) {
		MulticastPerf::waitCv.wait(lck);
	}
	lck.unlock();

	//prepare perftest to receive signals
	for (size_t i = 0; i < m_serverConns.size(); ++i) {
		m_client->receive(m_serverConns[i], &m_signal[i], sizeof(int));
	}

	//multicast perftest
	startTimer();
	uint128_t startTime = Timer::timestamp();
	for (size_t iter = 0; iter < m_iter; ++iter) {
		//send mcast data
		for (size_t i = 0; i < m_budget; ++i) {
			void* data = (void*) ((char*) m_data + i * m_size);
			m_client->sendMCast(m_mcastConn, data, m_size, (i + 1 == m_budget));
		}
		// std::cout << "Client " + to_string(m_threadid) + " sent " + to_string(m_budget) << std::endl;
		//poll for receives for signals
		for (size_t i = 0; i < m_serverConns.size(); ++i) {
			m_client->pollReceive(m_serverConns[i], true);
		}

		//prepare receiving signals for next iteration
		for (size_t i = 0; i < m_serverConns.size(); ++i) {
			m_client->receive(m_serverConns[i], &m_signal[i], sizeof(int));
		}
	}
	endTimer();
	uint128_t endTime = Timer::timestamp();
	resultedTime = endTime - startTime;
	cout << "resulted time:" << resultedTime << endl;
}

/********* server threads *********/
rdma::MCServerPerfThread::MCServerPerfThread(string mcastGroup, size_t serverPort,
		size_t size, size_t iter, size_t budget, size_t numThreads) {
	//init from parameters
	m_size = size;
	m_iter = iter;
	m_budget = budget;
	m_numThreads = numThreads;

	//start sequencer
	m_nodeIDSequencer = std::make_unique<NodeIDSequencer>();

	//start server
	m_server = new RDMAServer<UnreliableRDMA>("RDMAServer", serverPort, size * (m_budget+Config::RDMA_UD_OFFSET) + (sizeof(int)+Config::RDMA_UD_OFFSET) * m_numThreads);
	if (!m_server->startServer()) {
		throw invalid_argument("MCServerPerfThread could not start RDMAServer");
	}

	//join mcast group
	m_server->joinMCastGroup(mcastGroup, m_mcastConn);
	//allocate memory
	m_data = m_server->localAlloc(size * m_budget);
	m_signal = (int*) (m_server->localAlloc(sizeof(int) * m_numThreads));
	if (m_data == nullptr || m_signal == nullptr) {
		throw invalid_argument("MCServerPerfThread could not allocate memory");
	}
}

rdma::MCServerPerfThread::~MCServerPerfThread() {
	//free memory
	m_server->localFree(m_data);
	m_server->localFree(m_signal);
}

void rdma::MCServerPerfThread::run() {
	//server thread is ready and running
	m_ready = true;

	//prepare mcast for receiving data
	for (size_t i = 0; i < m_budget; ++i) {
		void* data = (void*) ((char*) m_data + m_size);
		m_server->receiveMCast(m_mcastConn, data, m_size);
	}

	//wait for client to connects
	std::vector<size_t> connectedRdmaIds;
	while ((connectedRdmaIds = m_server->getConnectedConnIDs()).size() < m_numThreads) {
	}

	//start multicast perftest
	startTimer();
	for (size_t iter = 0; iter < m_iter; ++iter) {
		//receive mcast data
		for (size_t i = 0; i < m_budget; ++i) {
			m_server->pollReceiveMCast(m_mcastConn, true);
		}
		//prepare receiving data for next round
		for (size_t i = 0; i < m_budget; ++i) {
			void* data = (void*) ((char*) m_data + m_size);
			m_server->receiveMCast(m_mcastConn, data, m_size);
		}
		// send signal to clients to start next iteration
		for (size_t i = 0; i < connectedRdmaIds.size(); ++i) {
			m_server->send(connectedRdmaIds[i], &m_signal[i], sizeof(int),((i + 1) == connectedRdmaIds.size()));
		}
		// std::cout << ".";
	}
	endTimer();
}

rdma::MulticastPerf::MulticastPerf(config_t config, bool isClient) :
		MulticastPerf(config.server, config.port, config.data, config.iter,
				config.threads) {
	this->isClient(isClient);
	m_logfile = config.logfile;
	if (m_servers.size() > 0) {
		this->isRunnable(true);
	}
}

rdma::MulticastPerf::MulticastPerf(string& servers, size_t serverPort, size_t size,
		size_t iter, size_t threads) {

	auto split_servers = StringHelper::split(servers);

	//check parameters
	if (servers.length() == 0) {
		this->isRunnable(false);
		return;
	}

	for (auto s : split_servers)
	{
		m_servers.push_back(s + ":" + to_string(serverPort));
	}
	Config::SEQUENCER_IP = split_servers[0];
	m_group = split_servers[0];
	m_serverPort = serverPort;
	m_size = size;
	m_budget = Config::RDMA_MAX_WR / threads;
	m_iter = iter / m_budget;
	m_client = nullptr;
	m_server = nullptr;
	m_numThreads = threads;
	MulticastPerf::signaled = false;
}

rdma::MulticastPerf::~MulticastPerf() {
	if (m_client != nullptr) {
		delete m_client;
	} else if (m_server != nullptr) {
		m_server->stopServer();
		delete m_server;
	}
}

void rdma::MulticastPerf::runServer() {
	std::cout << "Server budget: " << m_budget * m_numThreads << std::endl;
	//start server thread
	MCServerPerfThread* perfThread = new MCServerPerfThread(m_group,
			m_serverPort, m_size, m_iter, m_budget * m_numThreads,
			m_numThreads);
	perfThread->start();
	if (!perfThread->ready()) {
		usleep(Config::RDMA_SLEEP_INTERVAL);
	}
	m_sthread = perfThread;

	//wait for server thread to finish
	m_sthread->join();
	delete m_sthread;
}

void rdma::MulticastPerf::runClient() {
	//prepare all client threads
	std::cout << "Client budget: " << m_budget << std::endl;
	for (size_t i = 0; i < m_numThreads; i++) {
		MCClientPerfThread* perfThread = new MCClientPerfThread(0, m_group, m_servers, m_size,
				m_iter, m_budget);
		perfThread->start();
		if (!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
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

double rdma::MulticastPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_cthreads.size(); i++) {
		totalTime += m_cthreads[i]->resultedTime;
	}
	return ((double) totalTime) / m_cthreads.size();
}
