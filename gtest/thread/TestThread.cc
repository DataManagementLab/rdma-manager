

#include "TestThread.h"

void TestThread::SetUp() { m_testingThread = new TestingThread(); }

void TestThread::TearDown() { delete m_testingThread; }

TEST_F(TestThread, testRun){
  ASSERT_FALSE(m_testingThread->runned);
  m_testingThread->start();
  m_testingThread->join();
  ASSERT_TRUE(m_testingThread->runned);
  m_testingThread->stop();
}
