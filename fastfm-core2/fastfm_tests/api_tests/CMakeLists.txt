set(SOURCE_FILES
    tests-main.cpp
    cd_test.cpp
    ext_api_data_test.cpp
    fixture.h
    )

if(NOT EXTERNAL_RELEASE)
    list(APPEND SOURCE_FILES
                ext_api_function_test.cpp
                icd_test.cpp
                sgd_test.cpp
                bpr_test.cpp
        )
endif()

include_directories(${fm-lib_SOURCE_DIR})

add_executable(runApiTestsCatch ${SOURCE_FILES})

target_link_libraries(runApiTestsCatch fastfm solvers helpers)

add_test(NAME RunApiTestsCatch COMMAND runApiTestsCatch)
