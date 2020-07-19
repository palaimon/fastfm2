project(fm-lib)

set(HEADER_FILES
        fastfm.h
        fastfm_impl.h
        datasets.h
    )

if(NOT EXTERNAL_RELEASE)
    list(APPEND HEADER_FILES
            sgd_impl.h
            icd_impl.h
            sgd_impl.h
            sgd.h
            ranking.h
            bpr_impl.h
            bpr.h
        )
endif()

set(SOURCE_FILES
        fastfm.cpp
        datasets.cpp
   )

if(NOT EXTERNAL_RELEASE)
    list(APPEND SOURCE_FILES
        sgd_impl.cpp
        icd_impl.cpp
        icd.cpp
        sgd_impl.cpp
        sgd.cpp
        sgd_helper_impl.h
        sgd_helper_impl.cpp
        ranking.cpp
        bpr_impl.cpp
        bpr.cpp
        )
endif()

add_library(fastFM ${SOURCE_FILES} ${HEADER_FILES})

find_package(Threads)
target_link_libraries(fastFM ${CMAKE_THREAD_LIBS_INIT}) # For pthreads
target_link_libraries(fastFM dl) # For ldl

# solvers
add_subdirectory(solvers)
include_directories(solvers)
target_link_libraries(fastFM solvers)
