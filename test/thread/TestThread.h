/**
 * @file TestThread.h
 * @author cbinnig, lthostrup, tziegler
 * @date 2018-08-17
 */



#ifndef SRC_TEST_THREAD_TESTTHREAD_H_
#define SRC_TEST_THREAD_TESTTHREAD_H_

#include "../../src/utils/Config.h"
#include "../../src/thread/Thread.h"

using namespace rdma;

class TestThread : public CppUnit::TestFixture {
RDMA_UNIT_TEST_SUITE(TestThread);
  RDMA_UNIT_TEST(testRun);
  RDMA_UNIT_TEST_SUITE_END();
 private:
  class TestingThread : public Thread {
   public:
    bool runned = false;
    void run() {
      runned = true;
    }

  };
  TestingThread* m_testingThread;

 public:
  void setUp();
  void tearDown();
  void testRun();
};

#endif /* SRC_TEST_THREAD_TESTTHREAD_H_ */
