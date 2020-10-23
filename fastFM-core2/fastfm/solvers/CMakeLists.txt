SET(solvers_SRC
        solvers.h
        cd.cpp
        cd_impl.h
        cd_impl.cpp
        )

if(NOT EXTERNAL_RELEASE)
    list(APPEND solvers_SRC
            mcmc.h
            fm_utils.h
            fm_utils.cpp
            )
endif()

add_library(solvers ${solvers_SRC})
