/*
 * SwMulticastPerf.cc
 *
 *  Created on: Feb 7, 2017
 *      Author: cbinnig
 */

/*
 * Multicast.cc
 *
 *  Created on: Dec 3, 2016
 *      Author: cbinnig
 */

#include "SwMulticastPerf.h"
#include "../utils/Timer.h"
/********* client threads *********/
istore2::SWMCClientPerfThread::SWMCClientPerfThread(size_t size, size_t iter,
		size_t budget) {
	//init from parameters
	m_size = size;
	m_iter = iter;
	m_budget = budget;

	cout << "Client Thread budget " << m_budget << " iteration: " << m_iter
			<< endl;

	m_client = new RDMAClient();

	// HW Multicast, client should join mcast group

	//connect clients to all servers
	for (size_t i = 0; i < Config::PTEST_MCAST_NODES.size(); i++) {
		ib_addr_t ibAddr;
		string conn = Config::PTEST_MCAST_NODES[i];
		cout << "connect client to " << conn << endl;

		if (!m_client->connect(conn, ibAddr)) {
			throw invalid_argument("SWMCClientPerfThread connection failed");
			;
		}
		m_serverConns.push_back(ibAddr);
	}

	//allocate memory for data and signals
	// HW Multicast, only one memory region for all servers,
	// SW Multicast, for each server connection, a region in memory should be allocated
	// multiply the allocated data area with the number of the server connections
	m_data = m_client->localAlloc(m_size * m_budget * m_serverConns.size());
	m_signal =
			(int*) (m_client->localAlloc(sizeof(int) * m_serverConns.size()));
	if (m_data == nullptr || m_signal == nullptr) {
		throw invalid_argument("MCClientPerfThread could not allocate memory");
	}
}

istore2::SWMCClientPerfThread::~SWMCClientPerfThread() {
	//free memory
	m_client->localFree(m_data);
	m_client->localFree(m_signal);
}

void istore2::SWMCClientPerfThread::run() {

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
		//send budget to all connections sequenially
		size_t j = 0;
		for (size_t i = 0; i < m_serverConns.size(); i++) {

			for (; j < m_budget * (i + 1); ++j) {
				//cout<<"budget "<<j<<endl;
				void* data = (void*) ((char*) m_data + j * m_size);
				if (!m_client->send(m_serverConns[i], data, m_size, false)) {
					cout << " Failed!" << endl << flush;
					return;
				}
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
	uint128_t endTime = Timer::timestamp();
	resultedTime = endTime - startTime;
	endTimer();
}

/********* server threads *********/
istore2::SWMCServerPerfThread::SWMCServerPerfThread(size_t serverPort, size_t size,
		size_t iter, size_t budget, size_t numThreads) {
	//init from parameters
	m_size = size;
	m_iter = iter;
	m_budget = budget;
	m_numThreads = numThreads;

//start server
	m_server = new RDMAServer(serverPort);
	if (!m_server->startServer()) {
		throw invalid_argument(
				"SWMCServerPerfThread could not start RDMAServer");
	}

	//allocate memory
	m_data = m_server->localAlloc(size * m_budget);
	m_signal = (int*) (m_server->localAlloc(sizeof(int) * m_numThreads));
	if (m_data == nullptr || m_signal == nullptr) {
		throw invalid_argument(
				"SWMCServerPerfThread could not allocate memory");
	}
}

istore2::SWMCServerPerfThread::~SWMCServerPerfThread() {
	//free memory
	m_server->localFree(m_data);
	m_server->localFree(m_signal);
}

void istore2::SWMCServerPerfThread::run() {
	//server thread is ready and running
	m_ready = true;

	//wait for client to connects
	vector<ib_addr_t> clientConns = m_server->getQueues();
	while (clientConns.size() < m_numThreads) {
		clientConns = m_server->getQueues();
	}

	size_t j = 0;
	for (size_t i = 0; i < clientConns.size(); i++) {
		for (; j < m_budget; ++j) {
			void* data = (void*) ((char*) m_data + j * m_size);
			if (!m_server->receive(clientConns[i], data, m_size)) {
				cout << " Failed!" << endl << flush;
				return;
			}
		}
	}

	//start multicast perftest
	startTimer();
	for (size_t iter = 0; iter < m_iter; ++iter) {
		//receive data messages
		for (size_t i = 0; i < clientConns.size(); i++) {
			for (size_t j = 0; j < m_budget; ++j) {
				if (!m_server->pollReceive(clientConns[i])) {
					cout << " Failed!" << endl << flush;
					return;
				}
			}
		}

		//prepare receiving data for next round
		size_t j = 0;
		for (size_t i = 0; i < clientConns.size(); i++) {
			for (; j < m_budget; ++j) {
				void* data = (void*) ((char*) m_data + j * m_size);
				if (!m_server->receive(clientConns[i], data, m_size)) {
					cout << " Failed!" << endl << flush;
					return;
				}
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

istore2::SWMulticastPerf::SWMulticastPerf(config_t config, bool isClient) :
		SWMulticastPerf(config.port, config.data, config.iter, config.threads) {
	this->isClient(isClient);

	//check parameters
	//if (config.port > 0) {
	this->isRunnable(true);
	//}
}

istore2::SWMulticastPerf::SWMulticastPerf(size_t serverPort, size_t size, size_t iter,
		size_t threads) {
	m_serverPort = serverPort;
	m_size = size;
	m_budget = Config::RDMA_MAX_WR / threads;
	m_iter = iter / m_budget;
	m_client = nullptr;
	m_server = nullptr;
	m_numThreads = threads;
}

istore2::SWMulticastPerf::~SWMulticastPerf() {
	if (m_client != nullptr) {
		delete m_client;
	} else if (m_server != nullptr) {
		m_server->stopServer();
		delete m_server;
	}
}

void istore2::SWMulticastPerf::runServer() {
	//start server thread
	SWMCServerPerfThread* perfThread = new SWMCServerPerfThread(m_serverPort,
			m_size, m_iter, m_budget * m_numThreads, m_numThreads);
	perfThread->start();
	if (!perfThread->ready()) {
		usleep(Config::RDMA_SLEEP_INTERVAL);
	}
	m_sthread = perfThread;

	//wait for server thread to finish
	m_sthread->join();
	delete m_sthread;
}

void istore2::SWMulticastPerf::runClient() {
	// run only one thread
	SWMCClientPerfThread* perfThread = new SWMCClientPerfThread(m_size, m_iter,
			m_budget);
	perfThread->start();
	if (!perfThread->ready()) {
		usleep(Config::RDMA_SLEEP_INTERVAL);
	}
	m_cthreads.push_back(perfThread);

	//wait for clients threads to finish
	for (size_t i = 0; i < m_numThreads; i++) {
		m_cthreads[i]->join();
		delete m_cthreads[i];
	}
}

double istore2::SWMulticastPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_cthreads.size(); i++) {
		totalTime += m_cthreads[i]->resultedTime;
	}
	return ((double) totalTime) / m_cthreads.size();
}
