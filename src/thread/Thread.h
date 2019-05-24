/**
 * @file Thread.h
 * @author cbinnig, tziegler
 * @date 2018-08-17
 */



#ifndef THREAD_HPP_
#define THREAD_HPP_

#include "../utils/Config.h"
#include <thread>

namespace rdma {

class Thread {
 public:
  Thread();

  virtual ~Thread();

  void start();

  void join();

  void stop();

  bool running();

  bool killed();

  virtual void run() = 0;

  uint128_t time() {
    return m_endTime - m_startTime;
  }

  void startTimer();
  void endTimer();

  void waitForUser() {
    //wait for user input
    cout << "Press Enter to run Benchmark!" << flush << endl;
    char temp;
    cin.get(temp);
  }
 private:
  static void execute(void* arg);

  uint128_t m_startTime;
  uint128_t m_endTime;
  thread* m_thread;
  bool volatile m_kill;
  bool m_running;
};

}  // end namespace dpi

#endif /* THREAD_HPP_ */
