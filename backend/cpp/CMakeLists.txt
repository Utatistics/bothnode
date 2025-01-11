cmake_minimum_required(VERSION 3.10)  # Minimum required CMake version
project(cpps) # General project name, suitable for multiple modules

# ---------------------------------------------------------------
# C++ Standard Configuration
# ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)

# ---------------------------------------------------------------
# Pybind11 Configuration
# ---------------------------------------------------------------
set(pybind11_DIR "/home/vagrant/.venv/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)  # Find the pybind11 package

# ---------------------------------------------------------------
# Directory Configuration
# ---------------------------------------------------------------
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib) # shared librariesset
message(STATUS "INCLUDE_DIR: ${INCLUDE_DIR}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")


# ---------------------------------------------------------------
# Adding Modules
# ---------------------------------------------------------------
add_library(randomwalk MODULE src/pybind.cpp src/randomwalk.cpp)
set_target_properties(randomwalk PROPERTIES PREFIX "")
target_link_libraries(randomwalk PRIVATE pybind11::module) # Link pybind11 to the module
target_include_directories(randomwalk PRIVATE ${INCLUDE_DIR}) # Link pybind11 to the module
