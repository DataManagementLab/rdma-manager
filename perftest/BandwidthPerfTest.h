#ifndef PerfThread_H
#define PerfThread_H

#include "PerfTest.h"
#include "../rdma/RDMAClient.h"
#include "../rdma/RDMAServer.h"

#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace rdma {

class BandwidthPerfThread : public Thread {
public:
	BandwidthPerfThread(std::vector<std::string>& conns, size_t size, size_t iter);
	~BandwidthPerfThread();
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
	std::vector<std::string> m_conns;
	std::vector<NodeID> m_addr;
	size_t* m_remOffsets;
};



class BandwidthPerfTest : PerfTest {
public:
	BandwidthPerfTest();
	~BandwidthPerfTest();
	void setupTest();
	std::string getTestParameters();
	void runTest();

	static mutex waitLock;
	static condition_variable waitCv;
	static bool signaled;

private:
	std::vector<std::string> m_conns;
	std::vector<BandwidthPerfThread*> m_threads;

	RDMAServer<ReliableRDMA>* m_dServer;
};



}
#endif