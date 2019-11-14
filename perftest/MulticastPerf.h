/*
 * MulticastPerf.h
 *
 *  Created on: Dec 3, 2016
 *      Author: cbinnig
 */

#ifndef MULTICASTPERF_H_
#define MULTICASTPERF_H_

#include "../utils/Config.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"
#include "PerfTest.h"

#include <vector>
#include <mutex>
#include <condition_variable>

namespace istore2 {
    using namespace rdma;

class MCClientPerfThread: public Thread {
public:
	MCClientPerfThread(config_t config, bool isClient);

	MCClientPerfThread(string mcastGroup, size_t size, size_t iter,
			size_t budget);

	~MCClientPerfThread();

	void run();

	bool ready() {
		return m_ready;
	}

uint128_t resultedTime;

private:
	bool m_ready = false;
	void* m_data;
	int* m_signal;

	ib_addr_t m_mcastConn;
	size_t m_size;
	size_t m_iter;
	size_t m_budget;
	RDMAClient* m_client;
	vector<ib_addr_t> m_serverConns;
};

class MCServerPerfThread: public Thread {
public:
	MCServerPerfThread(string mcastGroup, size_t serverPort, size_t size,
			size_t iter, size_t budget, size_t numThreads);
	~MCServerPerfThread();
	void run();

	bool ready() {
		return m_ready;
	}

private:
	bool m_ready = false;
	void* m_data;
	int* m_signal;

	ib_addr_t m_mcastConn;
	size_t m_size;
	size_t m_iter;
	size_t m_budget;
	RDMAServer* m_server;
	size_t m_numThreads;
};

class MulticastPerf: public PerfTest {
public:
	MulticastPerf(config_t config, bool isClient);
	MulticastPerf(string& group, size_t serverPort, size_t size, size_t iter,
			size_t threads);
	~MulticastPerf();

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

//cout<<"time "<<time<<endl;	
	double mrps = m_iter*m_budget/((1e6)*time); // million request persecond per machine


		cout << m_size << "\t" << m_iter * m_budget << "\t" << bw << "\t" <<time<<"\t"<<mrps<<endl;
	}

	void printUsage() {
		cout << "istore2_perftest ... -s #mcastGrp ";
		cout << "(-d #dataSize -p #serverPort -t #threadNum)?" << endl;
	}

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;

private:
	string m_group;
	size_t m_serverPort;
	size_t m_size;
	size_t m_iter;
	bool m_isClient;
	size_t m_budget;
	size_t m_numThreads;

	RDMAClient* m_client;
	RDMAServer* m_server;

	vector<ib_addr_t> m_serverConns;
	vector<MCClientPerfThread*> m_cthreads;
	MCServerPerfThread* m_sthread;
};

}

#endif /* MULTICASTPERF_H_ */
