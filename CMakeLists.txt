cmake_minimum_required(VERSION 3.4.3)

project(main_project CXX)

find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-DDATA_FOLDER="${PROJECT_SOURCE_DIR}/data")

file(GLOB MAIN_SOURCES "src/*.cpp")
add_executable(gaze-estimator ${MAIN_SOURCES})
target_link_libraries(gaze-estimator PUBLIC ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})
target_include_directories(gaze-estimator PUBLIC "include/")

# Plot library
add_library(module-plot STATIC "src/plot/plot.cpp")
target_include_directories(module-plot PUBLIC "include/plot/")
target_link_libraries(module-plot PUBLIC ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})

# Link our libraries with our main executable
target_link_libraries(gaze-estimator PUBLIC module-plot)