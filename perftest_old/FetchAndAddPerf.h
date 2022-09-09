/*
 * FetchAndAddPerf.h
 *
 *  Created on: 01.03.2020
 *      Author: lthostrup
 */

#ifndef FetchAndAddPerf_H_
#define FetchAndAddPerf_H_

#include "../utils/Config.h"
#include "../utils/StringHelper.h"
#include "../utils/Network.h"
#include "../message/MessageErrors.h"
#include "../thread/Thread.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"
#include "PerfTest.h"


#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace rdma {


class FetchAndAddPerfThread: public Thread {
public:
	FetchAndAddPerfThread(std::string& conns, size_t iter);
	~FetchAndAddPerfThread();
	void run();
	bool ready() {
		return m_ready;
	}

private:
	bool m_ready = false;
	RDMAClient<ReliableRDMA> m_client;
	size_t* m_localCounter;
	size_t m_iter;
	std::string m_conns;
	size_t m_remOffset;
	NodeID m_serverID;
};

class FetchAndAddPerf: public PerfTest {
public:
	FetchAndAddPerf(config_t config, bool isClient);

	FetchAndAddPerf(string& region, size_t serverPort, size_t iter, size_t threads);

	~FetchAndAddPerf();

	void printHeader() {
		cout << "Size\tIter\tBW\tmopsPerS" << endl;
	}

	void printResults() {
		double time = (this->time()) / (1e9);
		size_t bw = (((double) sizeof(size_t) * m_iter * m_numThreads ) / (1024 * 1024)) / time;
		double mops = (((double) m_iter * m_numThreads) / time) / (1e6);

		std::cout << sizeof(size_t) << "\t" << m_iter << "\t" << bw << "\t" << mops << std::endl;

		std::cout <<  time  << "time" << std::endl;
	}

	void printUsage() {
	}

	void runClient();
	void runServer();
	double time();

	static std::mutex waitLock;
	static std::condition_variable waitCv;
	static bool signaled;

private:
	NodeIDSequencer *m_nodeIDSequencer;
	std::string m_conn;
	size_t m_serverPort;
	size_t m_iter;
	size_t m_numThreads;

	std::vector<FetchAndAddPerfThread*> m_threads;

	RDMAServer<ReliableRDMA>* m_dServer;
	inline void* cache_align( size_t size, void*& ptr, size_t& space) {
		intptr_t int_ptr = reinterpret_cast<intptr_t>(ptr),
				offset = (-int_ptr) & (64 - 1);
		if ((space -= offset) < size) {
			space += offset;
			return nullptr;
		}
		return reinterpret_cast<void*>(int_ptr + offset);
	}
};

}

#endif /* FetchAndAddPerf_H_ */
