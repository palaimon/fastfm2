SET(helpers_SRC
        fastfm_helpers.h
        datasets.h
        datasets.cpp
        )

add_library(helpers ${helpers_SRC})

if(MSVC)
    set_target_properties(helpers PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<0:>)
endif(MSVC)
