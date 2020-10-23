cmake_minimum_required(VERSION 3.12)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(fastfm)

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
    include_directories(fastFM_app)
    add_subdirectory(fastFM_app)
endif()

enable_testing()
add_subdirectory(fastFM_tests_catch)
