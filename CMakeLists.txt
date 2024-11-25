cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(ConvNetServer LANGUAGES CXX CUDA)

# Enable C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS_ONLY "Build only tests" OFF)

if(NOT BUILD_TESTS_ONLY)
  # Enable CUDA
  find_package(CUDA REQUIRED)
endif()

# OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(cpprestsdk REQUIRED)
find_package(OpenSSL REQUIRED)

# Intel TBB package
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

# Google Test package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

if(NOT BUILD_TESTS_ONLY)
# Define source files for backend code
set(SRC_FILES
    src/backend.cpp)

# Add the main executable
add_executable(${PROJECT_NAME} ${SRC_FILES})
endif()

# Define source files for preprocessing tests
set(TEST_FILES
    src/Preprocessing.cpp
    src/PreprocessingTest.cpp)

# Add preprocessing tests
add_executable(PreprocessingTest ${TEST_FILES})

# Add subdirectory for src
add_subdirectory(src)

if(NOT BUILD_TESTS_ONLY)
# Link libraries to the main executable
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS} 
    cpprestsdk::cpprest 
    OpenSSL::SSL 
    OpenSSL::Crypto 
    TBB::tbb
    pthread
)
endif()

# Link libraries to the test executable
target_link_libraries(PreprocessingTest PRIVATE 
    ${OpenCV_LIBS} 
    ${GTEST_LIBRARIES} 
    TBB::tbb  # Add TBB for preprocessing
    gtest_main 
    pthread
)

# Include necessary directories
if(NOT BUILD_TESTS_ONLY)
include_directories(${CUDA_INCLUDE_DIRS})
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Enable testing
enable_testing()

# Add test target
add_test(
    NAME PreprocessingTests
    COMMAND PreprocessingTest
)
