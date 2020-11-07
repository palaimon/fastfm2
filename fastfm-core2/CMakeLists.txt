cmake_minimum_required(VERSION 3.12)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(fastfm)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(3rdparty/eigen)

option(FASTFM_BUILD_WITH_GPERFTOOLS "Build with 'gperftools'" OFF)

string(COMPARE EQUAL "${CMAKE_TOOLCHAIN_FILE}" "" no_toolchain)

if(NOT MSVC AND no_toolchain)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

add_compile_definitions(EXTERNAL_RELEASE=${EXTERNAL_RELEASE})

include_directories(fastfm)
add_subdirectory(fastfm)

if(NOT EXTERNAL_RELEASE)
    include_directories(fastfm_app)
    add_subdirectory(fastfm_app)
endif()

enable_testing()
add_subdirectory(fastfm_tests)
include_directories(fastfm_tests)

