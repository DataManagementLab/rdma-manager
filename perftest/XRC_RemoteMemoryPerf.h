/*
 * XRC_RemoteMemoryPerf.h
 *
 *  Created on: 06.09.2020
 *  Original: RemoteMemoryPerf.h
 *  OrigAuthor: cbinnig
 *  Author: dfailing
 */

#ifndef XRC_RemoteMemoryPerf_H_
#define XRC_RemoteMemoryPerf_H_

#include "../utils/Config.h"
#include "../utils/StringHelper.h"
#include "../utils/Network.h"
#include "../message/MessageErrors.h"
#include "../thread/Thread.h"
#include "../rdma/ExReliableRDMA.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"
#include "PerfTest.h"

// perf-counter
#include "reporter.hpp"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <memory>

namespace rdma {


class XRC_RemoteMemoryPerfThread: public Thread {
public:
	XRC_RemoteMemoryPerfThread(vector<string>& conns, size_t size, size_t iter);
	~XRC_RemoteMemoryPerfThread();
	void run();
	bool ready() {
		return m_ready;
	}

private:
	bool m_ready = false;
	RDMAClient<ExReliableRDMA> m_client;
	void* m_data;
	size_t m_size;
	size_t m_iter;
	vector<string> m_conns;
	vector<NodeID> m_addr;
	std::shared_ptr<Reporter> reporter = std::make_shared<Reporter>();
	size_t* m_remOffsets;
};

class XRC_RemoteMemoryPerf: public PerfTest {
public:
	XRC_RemoteMemoryPerf(config_t config, bool isClient);

	XRC_RemoteMemoryPerf(string& region, size_t serverPort, size_t size,
			size_t iter, size_t threads);

	~XRC_RemoteMemoryPerf();

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

	vector<XRC_RemoteMemoryPerfThread*> m_threads;

	RDMAServer<ExReliableRDMA>* m_dServer;
};

}

#endif /* XRC_RemoteMemoryPerf_H_ */
