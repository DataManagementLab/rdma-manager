set(TEST_THREAD_SRC
  TestThread.cc
  TestThread.h
  ) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/

add_library(test_thread ${TEST_THREAD_SRC})

target_include_directories(test_thread PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_thread thread)