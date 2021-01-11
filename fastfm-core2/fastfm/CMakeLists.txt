project(fm-lib)

set(HEADER_FILES
        fastfm.h
        fastfm_impl.h
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

add_library(fastfm ${SOURCE_FILES} ${HEADER_FILES})

if(MSVC)
    set_target_properties(fastfm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<0:>)
endif(MSVC)

if(UNIX)
    find_package(Threads)
    target_link_libraries(fastfm ${CMAKE_THREAD_LIBS_INIT}) # For pthreads
    target_link_libraries(fastfm dl) # For ldl
endif(UNIX)

# solvers
add_subdirectory(solvers)
include_directories(solvers)
target_link_libraries(fastfm solvers)
