###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2016
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
    get_filename_component(
	    execname ${exec}
	    NAME_WE
    )
    if($ENV{VERBOSE})
        MESSAGE(STATUS "UTests -- Add ${execname}")
    endif()

    add_executable(
		    ${execname}
		    ${exec}
	    )

    target_link_libraries(
        ${execname}
        ${TBFMM_LIBRARIES}
        )
    add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
endforeach(exec)
