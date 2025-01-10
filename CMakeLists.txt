cmake_minimum_required(VERSION 3.10)
project(node2vec)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Specify the location of pybind11
set(pybind11_DIR "/home/vagrant/.venv/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Create the shared library from your C++ source
add_library(node2vec MODULE backend/cpp/node2vec.cpp)

# Remove the 'lib' prefix and specify the output directory
set_target_properties(node2vec PROPERTIES
    PREFIX ""  # Remove 'lib' prefix from the library name
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)

# Link pybind11 to your project
target_link_libraries(node2vec PRIVATE pybind11::module)
