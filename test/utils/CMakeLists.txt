set(TEST_UTILS_SRC
  TestConfig.cc
  TestConfig.h
  ) # Adding headers required for portability reasons http://voices.canonical.com/jussi.pakkanen/2013/03/26/a-list-of-common-cmake-antipatterns/

add_library(test_utils ${TEST_UTILS_SRC})

target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_utils thread)