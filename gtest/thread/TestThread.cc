

#include "TestThread.h"

void TestThread::SetUp() { m_testingThread = std::make_unique<TestingThread>(); }

TEST_F(TestThread, testRun){
  ASSERT_FALSE(m_testingThread->runned);
  m_testingThread->start();
  m_testingThread->join();
  ASSERT_TRUE(m_testingThread->runned);
  m_testingThread->stop();
}
