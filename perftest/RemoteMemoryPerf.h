/*
 * RemoteMemory_BW.h
 *
 *  Created on: 26.11.2015
 *      Author: cbinnig
 */

#ifndef RemoteMemory_BW_H_
#define RemoteMemory_BW_H_

#include "../utils/Config.h"
#include "../utils/StringHelper.h"
#include "../utils/Network.h"
#include "../message/MessageErrors.h"
#include "../thread/Thread.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"
#include "PerfTest.h"

// perf-counter
#include "reporter.hpp"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace rdma {


class RemoteMemoryPerfThread: public Thread {
public:
	RemoteMemoryPerfThread(vector<string>& conns, size_t size, size_t iter, std::string logfile = "");
	~RemoteMemoryPerfThread();
	void run();
	bool ready() {
		return m_ready;
	}

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> m_client;
	void* m_data;
	size_t m_size;
	size_t m_iter;
	vector<string> m_conns;
	vector<NodeID> m_addr;
	std::shared_ptr<Reporter> reporter = std::make_shared<Reporter>();
	size_t* m_remOffsets;
};

class RemoteMemoryPerf: public PerfTest {
public:
	RemoteMemoryPerf(config_t config, bool isClient);

	RemoteMemoryPerf(string& region, size_t serverPort, size_t size,
			size_t iter, size_t threads);

	~RemoteMemoryPerf();

	void printHeader() {
		std::cout << "Size\tIter\tBW\tmopsPerS" << std::endl;
	}

	void printResults() {
		double time = (this->time()) / (1e9);
		size_t bw = (((double) m_size * m_iter * m_numThreads ) / (1024 * 1024)) / time;
		double mops = (((double) m_iter * m_numThreads) / time) / (1e6);

		std::cout << m_size << "\t" << m_iter << "\t" << bw << "\t" << mops << std::endl;

		std::cout <<  time  << "time" << std::endl;
	}

	void printUsage() {
		std::cout << "perf_test ... -s #servers ";
		if (this->isClient())
			std::cout << "(-d #dataSize -p #serverPort -t #threadNum)";
		else
			std::cout << "(-p #serverPort)";
		std::cout << std::endl;
	}

	void runClient();
	void runServer();
	double time();

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;

private:
	NodeIDSequencer *m_nodeIDSequencer;
	vector<string> m_conns;
	size_t m_serverPort;
	size_t m_size;
	size_t m_iter;
	size_t m_numThreads;
	std::string m_logfile;

	vector<RemoteMemoryPerfThread*> m_threads;

	RDMAServer<ReliableRDMA>* m_dServer;
};

}

#endif /* RemoteMemory_BW_H_ */
