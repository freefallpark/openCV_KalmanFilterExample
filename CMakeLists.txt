cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(chatGPTKalmanFilter VERSION 1.0)

# Use C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the Python interpreter and required packages
find_package(PythonLibs REQUIRED)
#find_package(Python3 COMPONENTS Interpreter NumPy)
include_directories(${PYTHON_INCLUDE_DIRS})
#opencv Stuff:
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(chatGPTKalmanFilter main.cpp)

# Link to the Python libraries
target_link_libraries(chatGPTKalmanFilter ${PYTHON_LIBRARIES} ${OpenCV_LIBS})