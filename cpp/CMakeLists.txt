cmake_minimum_required(VERSION 3.15)
project(asc_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(asc_cpp
    src/par.cpp
    src/prefetch.cpp
    src/benchmark.cpp
)

target_include_directories(asc_cpp PRIVATE include)
target_link_libraries(asc_cpp PRIVATE pybind11::module)

install(TARGETS asc_cpp DESTINATION .)
