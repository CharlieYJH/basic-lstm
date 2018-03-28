cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-O3 -std=c++14 -fopenmp -DNDEBUG")

project(basic-lstm)

include_directories("./eigen3")
include_directories("./lstm")

add_executable(
	out
	main.cc
	./lstm/lstm.cc
)
