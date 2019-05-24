

#include "Thread.h"
#include "../utils/Timer.h"

#include <sched.h>

using namespace rdma;

Thread::Thread() {
  m_thread = nullptr;
  m_kill = false;
  m_running = false;
}

Thread::~Thread() {
  if (m_thread != nullptr) {
    delete m_thread;
    m_thread = nullptr;
  }
}

void Thread::start() {
  m_thread = new thread(Thread::execute, this);
}

void Thread::join() {
  if(m_thread == nullptr){
      return;
  }
  if(m_thread->joinable()){
      m_thread->join();
  }
}

void Thread::stop() {
  m_kill = true;
}

void Thread::execute(void* arg) {
  //set affinity
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int& cpu : Config::THREAD_CPUS) {
    CPU_SET(cpu, &cpuset);
  }
  sched_setaffinity(0, sizeof(cpuset), &cpuset);

  //run thread
  Thread* thread = (Thread*) arg;
  thread->m_running = true;
  thread->run();
}

void Thread::startTimer() {
  m_startTime = Timer::timestamp();
  m_endTime = m_startTime;
}

void Thread::endTimer() {
  m_endTime = Timer::timestamp();
}

bool Thread::running() {
  return m_running;
}

bool Thread::killed() {
  return m_kill;
}
