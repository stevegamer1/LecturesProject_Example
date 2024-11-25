cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ConvNetServer LANGUAGES CXX CUDA)

# Enable C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
find_package(CUDA REQUIRED)

# OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(cpprestsdk REQUIRED)
find_package(OpenSSL REQUIRED)

set(SRC_FILES
    src/server.cpp
)

# Add the executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Add subdirectory for src
add_subdirectory(src)

# Link CUDA runtime library and OpenCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE cpprestsdk::cpprest OpenSSL::SSL OpenSSL::Crypto)

# Include the necessary CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)