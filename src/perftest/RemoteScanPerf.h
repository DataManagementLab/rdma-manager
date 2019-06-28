/*
 * ScanMemory_BW.h
 *
 *  Created on: 26.11.2015
 *      Author: cbinnig
 */

#ifndef ScanMemory_BW_H_
#define ScanMemory_BW_H_

#include "../utils/Config.h"
#include "../utils/StringHelper.h"
#include "../thread/Thread.h"
#include "../net/rdma/RDMAClient.h"
#include "../net/rdma/RDMAServer.h"
#include "PerfTest.h"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace istore2 {

class RemoteScanPerfThread: public Thread {
public:
	RemoteScanPerfThread(string& conn, size_t size, size_t threadId,
			size_t numThreads);
	~RemoteScanPerfThread();
	void run();
	bool ready() {
		return m_ready;
	}

private:
	inline bool prefetch() {
		//prefetch
		while (m_writeIdx - m_readIdx < m_prefetchCount && m_writeIdx < m_endIdx) {

			void* localWritePtr = (void*) (m_data
					+ ((m_writeIdx % m_prefetchCount) * m_pageSize));
			size_t remReadOffset = m_remReadIdx * m_pageSize;

			if (!m_client.requestRead(m_addr, remReadOffset, localWritePtr,
					m_pageSize)) {
				return false;
			}

			m_writeIdx++;
			m_remReadIdx++;
		}

		return true;
	}

	bool m_ready = false;
	RDMAClient m_client;

	size_t m_pageSize;
	size_t m_tuplesPerPage;
	size_t m_prefetchCount;

	size_t m_startIdx;
	size_t m_endIdx;
	size_t m_readIdx;
	size_t m_remReadIdx;
	size_t m_writeIdx;

	char* m_data;
	ib_addr_t m_addr;
};

class RemoteScanPerf: public PerfTest {
public:
	RemoteScanPerf(config_t config, bool isClient);

	RemoteScanPerf(string& conn, size_t serverPort, size_t size, size_t iter,
			size_t threads);

	~RemoteScanPerf();

	void printHeader() {
		cout << "Size\tTuples\tBW\tmopsPerS" << endl;
	}

	void printResults() {
		uint64_t value = 0;
		size_t memSize = Config::RDMA_MEMSIZE;
		size_t tupleCount = memSize / sizeof(value);

		double time = (this->time()) / (1e9);
		size_t bw = ( memSize / (1024 * 1024)) / time;
		double mops = (((double) tupleCount ) / time) / (1e6);

		cout << m_size << "\t" << tupleCount << "\t" << bw << "\t" << mops << endl;
	}

	void printUsage() {
		cout << "istore2_perftest ... -s #server ";
		cout << "(-d #pageSize -p #serverPort -t #threadNum)?" << endl;
	}

	void runClient();
	void runServer();
	double time();

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;

private:
	string m_conn;
	size_t m_serverPort;
	size_t m_size;
	size_t m_iter;
	size_t m_numThreads;

	vector<RemoteScanPerfThread*> m_threads;

	RDMAServer* m_dServer;
};

}

#endif /* ScanMemory_BW_H_ */
