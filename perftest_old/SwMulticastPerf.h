/*
 * MulticastPerf.h
 *
 *  Created on: Dec 3, 2016
 *      Author: cbinnig
 */

#ifndef SWMULTICASTPERF_H_
#define SWMULTICASTPERF_H_

#include "../utils/Config.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"
#include "PerfTest.h"


#include <vector>


namespace istore2 {
    using namespace rdma;

class SWMCClientPerfThread: public Thread {
public:
	SWMCClientPerfThread(config_t config, bool isClient);
	SWMCClientPerfThread(size_t size, size_t iter, size_t budget);

	~SWMCClientPerfThread();

	void run();

	bool ready() {
		return m_ready;
	}
 
uint128_t resultedTime;
private:
	bool m_ready = false;
	void* m_data;
	int* m_signal;

	size_t m_size;
	size_t m_iter;
	size_t m_budget;
	RDMAClient* m_client;
	vector<ib_addr_t> m_serverConns;
};

class SWMCServerPerfThread: public Thread {
public:
	SWMCServerPerfThread(size_t serverPort, size_t size, size_t iter,
			size_t budget, size_t numThreads);
	~SWMCServerPerfThread();
	void run();

	bool ready() {
		return m_ready;
	}

private:
	bool m_ready = false;
	void* m_data;
	int* m_signal;

	size_t m_size;
	size_t m_iter;
	size_t m_budget;
	RDMAServer* m_server;
	size_t m_numThreads;
};

class SWMulticastPerf: public PerfTest {
public:
	SWMulticastPerf(config_t config, bool isClient);
	SWMulticastPerf(size_t serverPort, size_t size, size_t iter,
			size_t threads);
	~SWMulticastPerf();

	void runClient();
	void runServer();
	double time();

	void printHeader() {
		cout << "Size\tIter\tBW\tTime\tMrps" << endl;
	}

	void printResults() {
		double time = (this->time()) / (1e9);
		size_t bw = (((double) m_size * m_iter * m_budget * m_numThreads)
				/ (1024 * 1024)) / time;

		double mrps = m_budget/((1e6)*time); // million request persecond per machine

		cout << m_size << "\t" << m_iter * m_budget << "\t" << bw << "\t" <<time<<"\t"
				<< mrps << endl;
	}

	void printUsage() {
		cout << "istore2_perftest ... -s #mcastGrp ";
		cout << "(-d #dataSize -p #serverPort -t #threadNum)?" << endl;
	}

private:
	size_t m_serverPort;
	size_t m_size;
	size_t m_iter;
	bool m_isClient;
	size_t m_budget;
	size_t m_numThreads;

	RDMAClient* m_client;
	RDMAServer* m_server;
	vector<ib_addr_t> m_serverConns;
	vector<SWMCClientPerfThread*> m_cthreads;
	SWMCServerPerfThread* m_sthread;
};

}

#endif /* MULTICASTPERF_H_ */
