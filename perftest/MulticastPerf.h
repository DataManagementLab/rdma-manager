/*
 * MulticastPerf.h
 *
 *  Created on: Dec 3, 2016
 *      Author: cbinnig
 */

#ifndef MULTICASTPERF_H_
#define MULTICASTPERF_H_

#include "../utils/Config.h"
#include "../utils/StringHelper.h"
#include "../utils/Filehelper.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"
#include "../rdma/UnreliableRDMA.h"
#include "PerfTest.h"

#include <vector>
#include <mutex>
#include <condition_variable>

namespace rdma {

class MCClientPerfThread: public Thread {
public:
	MCClientPerfThread(int threadid, string mcastGroup, vector<string> servers, size_t size, size_t iter,
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
	int m_threadid;
	NodeID m_mcastConn;
	size_t m_size;
	size_t m_iter;
	size_t m_budget;
	RDMAClient<UnreliableRDMA>* m_client;
	vector<NodeID> m_serverConns;
	
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

	NodeID m_mcastConn;
	size_t m_size;
	size_t m_iter;
	size_t m_budget;
	RDMAServer<UnreliableRDMA>* m_server;
	size_t m_numThreads;
	std::unique_ptr<NodeIDSequencer> m_nodeIDSequencer;
};

class MulticastPerf: public PerfTest {
public:
	MulticastPerf(config_t config, bool isClient);
	MulticastPerf(string& servers, size_t serverPort, size_t size, size_t iter,
			size_t threads);
	~MulticastPerf();

	void runClient();
	void runServer();
	double time();

	void printHeader() {

		if (!m_logfile.empty() && !Filehelper::fileExists(m_logfile))
		{
		std::ofstream log(m_logfile, std::ios_base::app | std::ios_base::out);
		log << "Size,Iter,BW,mopsPerS,Time,threads,servers\n";
		log.flush();
		}
		cout << "Size\tIter\tBW\tmopsPerS\tTime\tthreads\tservers" << endl;
	}

	void printResults() {
		double time = (this->time()) / (1e9);
		size_t bw = (((double) m_size * m_iter * m_budget * m_numThreads) / (1024 * 1024)) / time;

//cout<<"time "<<time<<endl;	
	double mrps = m_iter*m_budget/((1e6)*time); // million request persecond per machine


		cout << m_size << "\t" << m_iter * m_budget << "\t" << bw << "\t" <<mrps<<"\t"<<time<<"\t"<<m_numThreads<<"\t"<<m_servers.size()<<endl;
		if (!m_logfile.empty())
		{
			std::ofstream log(m_logfile, std::ios_base::app | std::ios_base::out);
			log << m_size << "," << m_iter * m_budget << "," << bw << "," <<mrps<<","<<time<<","<<m_numThreads<<","<<m_servers.size()<<'\n';
			log.flush();
			log.close();
		}
	}

	void printUsage() {
		cout << "perf_test ... ";
		cout << "-s #servers -d #dataSize -p #serverPort -t #threadNum" << endl;
	}

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;

private:
	string m_group;
	std::vector<std::string> m_servers;
	size_t m_serverPort;
	size_t m_size;
	size_t m_iter;
	bool m_isClient;
	size_t m_budget;
	size_t m_numThreads;
	std::string m_logfile;

	RDMAClient<UnreliableRDMA>* m_client;
	RDMAServer<UnreliableRDMA>* m_server;

	vector<NodeID> m_serverConns;
	vector<MCClientPerfThread*> m_cthreads;
	MCServerPerfThread* m_sthread;
};

}

#endif /* MULTICASTPERF_H_ */
