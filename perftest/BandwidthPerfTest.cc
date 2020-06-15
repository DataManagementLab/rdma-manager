#include "BandwidthPerfTest.h"

#include "../src/memory/BaseMemory.h"
#include "../src/memory/MainMemory.h"
#include "../src/memory/CudaMemory.h"
#include "PerfEvent.hpp"


mutex rdma::BandwidthPerfTest::waitLock;
condition_variable rdma::BandwidthPerfTest::waitCv;
bool rdma::BandwidthPerfTest::signaled;

rdma::BandwidthPerfThread::BandwidthPerfThread(std::vector<std::string>& conns, size_t size, size_t iter) {
	m_size = size;
	m_iter = iter;
	m_conns = conns;
	m_remOffsets = new size_t[m_conns.size()];

	for (size_t i = 0; i < m_conns.size(); ++i) {
	    NodeID  nodeId = 0;
		//ib_addr_t ibAddr;
		string conn = m_conns[i];
		if (!m_client.connect(conn, nodeId)) {
			throw invalid_argument("RemoteMemoryPerfThread connection failed");
		}
		m_addr.push_back(nodeId);
		m_client.remoteAlloc(conn, m_size, m_remOffsets[i]);
	}

	m_data = m_client.localAlloc(m_size);
	memset(m_data, 1, m_size);
}

rdma::BandwidthPerfThread::~BandwidthPerfThread() {

	m_client.localFree(m_data);

	for (size_t i = 0; i < m_conns.size(); ++i) {
		string conn = m_conns[i];
		m_client.remoteFree(conn, m_remOffsets[i], m_size);
	}
    delete m_remOffsets;

}

void rdma::BandwidthPerfThread::run() {
	unique_lock < mutex > lck(BandwidthPerfTest::waitLock);
	if (!BandwidthPerfTest::signaled) {
		m_ready = true;
		BandwidthPerfTest::waitCv.wait(lck);
	}
	lck.unlock();

	{
		std::cout << "Perf Measurement enabled" << std::endl;
		PerfEventBlock pb(m_iter);
		startTimer();
		for (size_t i = 0; i < m_iter; ++i) {
			size_t connIdx = i % m_conns.size();
			bool signaled = (i == (m_iter - 1));
			m_client.write(m_addr[connIdx],m_remOffsets[connIdx],m_data,m_size,signaled);


		}
		endTimer();
	}	

}


rdma::BandwidthPerfTest::~BandwidthPerfTest() : PerfTest(){}
rdma::BandwidthPerfTest::~BandwidthPerfTest(){
	for (size_t i = 0; i < m_threads.size(); i++) {
		delete m_threads[i];
	}
	m_threads.clear();
}

void rdma::BandwidthPerfTest::setupTest(){
	printf("TEST SETUP\n"); // TODO REMOVE

	// TODO

}

std::string rdma::BandwidthPerfTest::getTestParameters(){

	// TODO

	return "PARAMETERS"; // TODO REMOVE
}

void rdma::BandwidthPerfTest::runTest(){
	printf("TEST RUNNING\n"); // TODO REMOVE

	// TODO

}