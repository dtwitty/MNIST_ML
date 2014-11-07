cmake_minimum_required(VERSION 2.8)
project( EyeSore )
find_package( OpenCV REQUIRED )
file(GLOB EyeSore_SRC
  "*.h"
  "*.c"
  "*.hpp"
  "*.cpp"
  )
add_executable( EyeSore ${EyeSore_SRC} )
list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -g -O2")
target_link_libraries( EyeSore ${OpenCV_LIBS} )
