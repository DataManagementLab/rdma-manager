/*
 * XRC_RemoteMemoryPerf.cc
 *
 *  Created on: 06.09.2020
 *  Original: RemoteMemoryPerf.cc
 *  OrigAuthor: cbinnig
 *  Author: dfailing
 */

#include "XRC_RemoteMemoryPerf.h"
#include "PerfEvent.hpp"

// perf-counter
#include "timed_reporter.hpp"
#include "analyzing_scope.hpp"
#include "rdma_aggregator.hpp"
#include "perf_aggregator.hpp"
#include "time_aggregator.hpp"
#include "stdout_output.hpp"
#include "csv_output.hpp"
#include <chrono>

mutex rdma::XRC_RemoteMemoryPerf::waitLock;
condition_variable rdma::XRC_RemoteMemoryPerf::waitCv;
bool rdma::XRC_RemoteMemoryPerf::signaled;

rdma::XRC_RemoteMemoryPerfThread::XRC_RemoteMemoryPerfThread(vector<string>& conns,
		size_t size, size_t iter, std::string logfile) {
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
					"XRC_RemoteMemoryPerfThread connection failed");
		}
		m_addr.push_back(nodeId);
		m_client.remoteAlloc(conn, m_size, m_remOffsets[i]);
	}

	/*reporter->addAggregator(std::make_shared<TimeAggregator>());
	reporter->addAggregator(std::make_shared<FileAggregator>("/sys/class/infiniband_verbs/uverbs0/num_page_faults"));
	reporter->addAggregator(std::make_shared<FileAggregator>("/sys/class/infiniband_verbs/uverbs1/num_page_faults"));
	reporter->addAggregator(std::make_shared<RdmaAggregator>(rx_write_requests));
	reporter->addAggregator(std::make_shared<RdmaAggregator>(rx_read_requests));
	reporter->addAggregator(std::make_shared<RdmaAggregator>("mlx5_1", rx_write_requests));
	reporter->addAggregator(std::make_shared<RdmaAggregator>("mlx5_1", rx_read_requests));
	reporter->addOutput(std::make_shared<StdOut_Output>());
	if(!logfile.empty()) {
		reporter->addOutput(std::make_shared<Csv_Output>(logfile));
	}*/
  
	m_data = m_client.localAlloc(m_size);
	memset(m_data, 1, m_size);
}

rdma::XRC_RemoteMemoryPerfThread::~XRC_RemoteMemoryPerfThread() {

	m_client.localFree(m_data);

	for (size_t i = 0; i < m_conns.size(); ++i) {
		string conn = m_conns[i];
		m_client.remoteFree(conn, m_remOffsets[i], m_size);
	}
    delete m_remOffsets;

}

void rdma::XRC_RemoteMemoryPerfThread::run() {
	unique_lock < mutex > lck(XRC_RemoteMemoryPerf::waitLock);
	if (!XRC_RemoteMemoryPerf::signaled) {
		m_ready = true;
		XRC_RemoteMemoryPerf::waitCv.wait(lck);
	}
	lck.unlock();

	{
		std::cout << "Perf Measurement enabled" << std::endl;
		PerfEventBlock pb(m_iter);
		AnalyzingScope scope(reporter);
		startTimer();
		for (size_t i = 0; i < m_iter; ++i) {
			size_t connIdx = i % m_conns.size();
			bool signaled = (i == (m_iter - 1));
			m_client.read(m_addr[connIdx],m_remOffsets[connIdx],m_data,m_size,signaled);


		}
		endTimer();
	}	

}

rdma::XRC_RemoteMemoryPerf::XRC_RemoteMemoryPerf(config_t config, bool isClient) :
		XRC_RemoteMemoryPerf(config.server, config.port, config.data, config.iter,
				config.threads) {
	this->m_logfile = config.logfile;
	this->isClient(isClient);

	//check parameters
	if (config.server.length() > 0) {
		this->isRunnable(true);
		Config::SEQUENCER_IP = rdma::Network::getAddressOfConnection(m_conns[0]);
	}
}

rdma::XRC_RemoteMemoryPerf::XRC_RemoteMemoryPerf(string& conns, size_t serverPort,
		size_t size, size_t iter, size_t threads) {
	m_conns = StringHelper::split(conns);
	for (auto &conn : m_conns)
	{
		conn += ":" + to_string(serverPort);
	}
	
	m_serverPort = serverPort;
	m_size = size;
	m_iter = iter;
	m_numThreads = threads;
	XRC_RemoteMemoryPerf::signaled = false;
}

rdma::XRC_RemoteMemoryPerf::~XRC_RemoteMemoryPerf() {
	if (this->isClient()) {
		for (size_t i = 0; i < m_threads.size(); i++) {
			delete m_threads[i];
		}
		m_threads.clear();
	}
}

void rdma::XRC_RemoteMemoryPerf::runServer() {
	if (rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) == rdma::Network::getAddressOfConnection(m_conns[0]))
	{
		std::cout << "Starting NodeIDSequencer on: " << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << rdma::Config::SEQUENCER_PORT << std::endl;
		m_nodeIDSequencer = new NodeIDSequencer();
	}
	
	std::cout << "Starting RDMAServer on: " << rdma::Config::getIP(rdma::Config::RDMA_INTERFACE) << ":" << m_serverPort << std::endl;

  // starting 
  auto reporter = std::make_shared<TimedReporter>(std::chrono::seconds(1));
	reporter->addAggregator(std::make_shared<TimeAggregator>());
	reporter->addAggregator(std::make_shared<FileAggregator>("/sys/class/infiniband_verbs/uverbs0/num_page_faults"));
	reporter->addAggregator(std::make_shared<FileAggregator>("/sys/class/infiniband_verbs/uverbs1/num_page_faults"));
	reporter->addAggregator(std::make_shared<RdmaAggregator>(rx_write_requests));
	reporter->addAggregator(std::make_shared<RdmaAggregator>(rx_read_requests));
	reporter->addAggregator(std::make_shared<RdmaAggregator>("mlx5_1", rx_write_requests));
	reporter->addAggregator(std::make_shared<RdmaAggregator>("mlx5_1", rx_read_requests));
	reporter->addOutput(std::make_shared<StdOut_Output>());
	if(!m_logfile.empty()) {
		reporter->addOutput(std::make_shared<Csv_Output>(m_logfile));
	}

	m_dServer = new RDMAServer<ExReliableRDMA>("test", m_serverPort);

  class RDMAServerConnAgg : public Aggregator {
    public:
      long read(){ return m_dServer->getNumQPs(); }
      std::string getName() { return "qps"; }
      std::string getUnit() { return ""; }
  };
  reporter->addAggregator(std::make_shared<RDMAServerConnAgg>());
  reporter->activate();

	m_dServer->startServer();
  auto t = std::thread([this](){
      std::string s;
      while(true) {
        std::cin >> s;
        if(s.find("stop") != std::string::npos) {
          m_dServer->stopServer();
          return;
        }
      }
    });
	while (m_dServer->isRunning()) {
		usleep(Config::RDMA_SLEEP_INTERVAL);
	}
  t.join();
  reporter->deactivate();
}

void rdma::XRC_RemoteMemoryPerf::runClient() {
	//start all client threads
	for (size_t i = 0; i < m_numThreads; i++) {
		XRC_RemoteMemoryPerfThread* perfThread = new XRC_RemoteMemoryPerfThread(m_conns,
				m_size, m_iter, m_logfile);
		perfThread->start();
		if (!perfThread->ready()) {
			usleep(Config::RDMA_SLEEP_INTERVAL);
		}
		m_threads.push_back(perfThread);
	}

	//wait for user input
	waitForUser();

	//send signal to run benchmark
	XRC_RemoteMemoryPerf::signaled = false;
	unique_lock < mutex > lck(XRC_RemoteMemoryPerf::waitLock);
	XRC_RemoteMemoryPerf::waitCv.notify_all();
	XRC_RemoteMemoryPerf::signaled = true;
	lck.unlock();
	for (size_t i = 0; i < m_threads.size(); i++) {
		m_threads[i]->join();
	}
}

double rdma::XRC_RemoteMemoryPerf::time() {
	uint128_t totalTime = 0;
	for (size_t i = 0; i < m_threads.size(); i++) {
		totalTime += m_threads[i]->time();
	}
	return ((double) totalTime) / m_threads.size();
}

