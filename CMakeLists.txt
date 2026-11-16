cmake_minimum_required(VERSION 3.20)
project(ppe_edge_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)   # timings below are only meaningful for an optimized build
endif()

# Homebrew puts both libraries under /opt/homebrew; point CMake there when it is not already on the path.
if(APPLE AND EXISTS /opt/homebrew)
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/opencv /opt/homebrew/opt/onnxruntime)
endif()

# Both packages ship CMake config files: .../lib/cmake/opencv5/OpenCVConfig.cmake and
# .../lib/cmake/onnxruntime/onnxruntimeConfig.cmake (target: onnxruntime::onnxruntime).
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
message(STATUS "OpenCV ${OpenCV_VERSION} (${OpenCV_DIR})")

# Optional: link another libonnxruntime build instead of Homebrew's, e.g. the one inside the Python wheel, to compare
# runtimes without changing the language:  cmake -S . -B build-wheelort -DORT_LIBRARY=$PWD/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.32.0.dylib
# Headers still come from Homebrew, so the versions must match (1.32.0).
set(ORT_LIBRARY "" CACHE FILEPATH "Use this libonnxruntime instead of the Homebrew one")
if(ORT_LIBRARY)
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h PATHS /opt/homebrew/opt/onnxruntime/include/onnxruntime REQUIRED)
  # The wheel's dylib has install name @rpath/libonnxruntime.1.dylib; give dyld a directory where that name exists.
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ortlib)
  file(CREATE_LINK ${ORT_LIBRARY} ${CMAKE_BINARY_DIR}/ortlib/libonnxruntime.1.dylib SYMBOLIC)
  add_library(ort_override SHARED IMPORTED)
  set_target_properties(ort_override PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/ortlib/libonnxruntime.1.dylib
                        INTERFACE_INCLUDE_DIRECTORIES ${ORT_INCLUDE_DIR})
  set(ORT_TARGET ort_override)
  set(CMAKE_BUILD_RPATH ${CMAKE_BINARY_DIR}/ortlib)
  message(STATUS "ONNX Runtime: overridden, using ${ORT_LIBRARY}")
else()
  find_package(onnxruntime REQUIRED CONFIG)
  set(ORT_TARGET onnxruntime::onnxruntime)
  message(STATUS "ONNX Runtime ${onnxruntime_VERSION} (${onnxruntime_DIR})")
endif()

add_library(runner STATIC src/yolo.cpp src/detector.cpp src/pipeline.cpp)
target_include_directories(runner PUBLIC src)
target_link_libraries(runner PUBLIC ${OpenCV_LIBS} ${ORT_TARGET})
target_compile_options(runner PRIVATE -Wall -Wextra)

add_executable(ppe_runner src/main.cpp)
target_link_libraries(ppe_runner PRIVATE runner)
target_compile_options(ppe_runner PRIVATE -Wall -Wextra)
target_compile_definitions(ppe_runner PRIVATE CV_VERSION_STR="${OpenCV_VERSION}")

# Unit test for the pure C++ pieces (letterbox / NMS); the cross-check against Python is tests/run_correctness.sh.
enable_testing()
add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE runner)
add_test(NAME unit_tests COMMAND unit_tests)
add_test(NAME cpp_vs_python_reference
         COMMAND ${CMAKE_SOURCE_DIR}/tests/run_correctness.sh $<TARGET_FILE:ppe_runner>
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
