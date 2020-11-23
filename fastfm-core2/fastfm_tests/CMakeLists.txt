include_directories(helpers)
add_subdirectory(helpers)

add_subdirectory(api_tests)

if(NOT EXTERNAL_RELEASE)
    add_subdirectory(internal_tests)
    add_subdirectory(solver_tests)
endif()