cmake_minimum_required(VERSION 3.18)
project(libmath LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(libmath libmath.cpp)

target_compile_definitions(libmath PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)
