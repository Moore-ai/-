cmake_minimum_required(VERSION 3.2)

project(learn CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB MAIN_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

message(STATUS "头文件 = ${MAIN_HEAD}")

message(STATUS "源文件 = ${MAIN_SRC}")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(run ${MAIN_HEAD} ${MAIN_SRC})

target_link_libraries(run ${OpenCV_LIBS})
