###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(UTESTS_TBFMM CXX)

ADD_DEFINITIONS(${TBFMM_CXX_FLAGS})

# Find all code files (no GLOB_RECURSE)
file(
    GLOB	
    source_tests_files
    ./*.cpp
    )


# Adding the project sources dir as an include dir
INCLUDE_DIRECTORIES(
     ${TBFMM_BINARY_DIR}/src    
     ${TBFMM_SOURCE_DIR}/src    
     ${UTESTS_TBFMM_SOURCE_DIR}
)


# Add execs - 1 cpp = 1 exec
foreach(exec ${source_tests_files})
    get_filename_component( execname ${exec} NAME_WE)
    set(compile_exec TRUE)
    	
    foreach(tbf_use_key ${TBF_USE_LIST_AVAILABLE})
        file(STRINGS "${exec}" lines_tbf_use REGEX "@TBF_USE_${tbf_use_key}")
        if(lines_tbf_use AND NOT ${tbf_use_key} IN_LIST TBF_USE_LIST)
            message( STATUS "UTests -- ${execname} needs ${tbf_use_key}" )
            set(compile_exec FALSE)
        endif()
    endforeach()

    # Dependency are OK
    if( compile_exec )
        if($ENV{VERBOSE})
            MESSAGE(STATUS "UTests -- Add ${execname}")
        endif()

        add_executable(${execname} ${exec})
        
        if(TBFMM_USE_CUDA)
            set_source_files_properties(${execname} PROPERTIES LANGUAGE CUDA)
        endif()

        target_link_libraries(${execname} ${TBFMM_LIBRARIES})
        add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
    else()
        MESSAGE(STATUS "UTests -- ${execname} cannot be compiled due to missing libs (${exec})")
    endif()
endforeach(exec)
