cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(rk3588-yolo11-thread-video VERSION 0.0.1 LANGUAGES CXX)

message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RGA_DIR}/include
)

add_library(nn_process SHARED
    src/process/preprocess.cpp
    src/process/postprocess.cpp
)
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

add_library(rknn_engine SHARED
    src/engine/rknn_engine.cpp
)
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)

add_library(yolov11_lib SHARED
    src/task/yolov11_custom.cpp
)
target_link_libraries(yolov11_lib
    rknn_engine
    nn_process
)

add_library(draw_lib SHARED
    src/draw/cv_draw.cpp
)
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)

add_executable(img_demo
    src/img_demo.cpp
)
target_link_libraries(img_demo
    draw_lib
    yolov11_lib
)

add_executable(thread_pool_demo
    src/thread_pool_demo.cpp
    src/task/yolov11_thread_pool.cpp
)
target_link_libraries(thread_pool_demo
    draw_lib
    yolov11_lib
)
