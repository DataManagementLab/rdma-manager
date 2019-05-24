

#include "TestThread.h"

void TestThread::setUp() {
  m_testingThread = new TestingThread();
}

void TestThread::tearDown() {
  delete m_testingThread;
}

void TestThread::testRun() {

  CPPUNIT_ASSERT_EQUAL(m_testingThread->runned, false);
  m_testingThread->start();
  m_testingThread->join();
  CPPUNIT_ASSERT_EQUAL(m_testingThread->runned, true);
  m_testingThread->stop();
}
