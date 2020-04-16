project(TBFMM CXX)
cmake_minimum_required(VERSION 3.15)

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0057 NEW)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/deps/CMakeModules/")

#################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#################################################

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(TBFMM_EXTRA_FLAG "-m64 -march=native -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual  -Wall -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wno-error" CACHE STRING "Additional flags"  )
else()
    set(TBFMM_EXTRA_FLAG "-m64 -march=native -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wconversion  -Wall -Wno-sign-conversion -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wconversion -Wno-error" CACHE STRING "Additional flags"  )
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBFMM_EXTRA_FLAG}")

#################################################

set(TBF_USE_LIST "")
set(TBF_USE_LIST_AVAILABLE "")

#################################################

option(USE_COVERAGE "Should be ON to enable code coverage" OFF)

if( USE_COVERAGE )
    message(STATUS "Use code coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
    list(APPEND TBFMM_LIBRARIES gcov)
    # Then one can use these commande:
    # lcov --directory CMakeFiles/ -c -o coverage-rapport.info
    # genhtml -o ./coverage -t "Coverage example" coverage-rapport.info
    list(APPEND TBF_USE_LIST COVERAGE)
endif()

list(APPEND TBF_USE_LIST_AVAILABLE COVERAGE)

#################################################

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/spetabaru/CMakeLists.txt")
    message(STATUS "SPETABARU Found")
    set(SPETABARU_JUST_LIB TRUE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/deps/spetabaru")
    INCLUDE_DIRECTORIES(
             ${SPETABARU_BINARY_DIR}/Src
             ${SPETABARU_SOURCE_DIR}/Src
             ${SPETABARU_INCLUDE_DIR}
        )
    link_directories("${SPETABARU_DIR}/lib/")
    list(APPEND TBFMM_LIBRARIES ${SPETABARU_LIBRARIES} spetabaru)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPETABARU_CXX_FLAGS}")

    list(APPEND TBF_USE_LIST SPETABARU)
else()
    message(STATUS "SPETABARU Cannot be found in ${CMAKE_CURRENT_SOURCE_DIR}/deps/spetabaru (please use git submodule init && git submodule update)")
endif()

list(APPEND TBF_USE_LIST_AVAILABLE SPETABARU)

#################################################

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/inastemp/CMakeLists.txt")
    message(STATUS "INASTEMP Found")
    set(INASTEMP_JUST_LIB TRUE)
    set(INASTEMP_DO_INSTALL TRUE)
    # Create target
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/deps/inastemp")
    INCLUDE_DIRECTORIES(${INASTEMP_BINARY_DIR}/Src ${INASTEMP_SOURCE_DIR}/Src ${INASTEMP_INCLUDE_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${INASTEMP_CXX_FLAGS}")
        
    list(APPEND TBF_USE_LIST INASTEMP)
else()
    message(STATUS "Inastemp Cannot be found in ${CMAKE_CURRENT_SOURCE_DIR}/deps/inastemp (please use git submodule init && git submodule update)")
endif()

list(APPEND TBF_USE_LIST_AVAILABLE INASTEMP)

#################################################

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    list(APPEND TBF_USE_LIST OPENMP)
endif()

list(APPEND TBF_USE_LIST_AVAILABLE OPENMP)

#################################################

find_package(FFTW)
if(FFTW_FOUND)
    message(STATUS "FFTW Found") 
    message(STATUS "FFTW_INCLUDES = ${FFTW_INCLUDES}")   
    message(STATUS "FFTW_LIBRARIES = ${FFTW_LIBRARIES}")   

    INCLUDE_DIRECTORIES("${FFTW_INCLUDES}")
    link_directories("${FFTW_LIBRARIES}")
    list(APPEND TBFMM_LIBRARIES ${FFTW_LIBRARIES})
    list(APPEND TBF_USE_LIST FFTW)
else()
    message(STATUS "FFTW Cannot be found, try by setting -DFFTW_ROOT=... or env FFTW_ROOT")
endif()

list(APPEND TBF_USE_LIST_AVAILABLE FFTW)

#################################################

foreach(tbf_use_key ${TBF_USE_LIST})
    add_definitions(-DTBF_USE_${tbf_use_key})
endforeach()

message(STATUS "Available compilation keys are: ${TBF_USE_LIST}")

#################################################

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE cpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(cpp_for_lib)
    file(GLOB_RECURSE hpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
    add_library(TBFMM STATIC ${cpp_for_lib} ${hpp_for_lib})
    list(APPEND TBFMM_LIBRARIES TBFMM)
endif()


#################################################


file(GLOB_RECURSE cpp_for_examples ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
foreach(an_example ${cpp_for_examples})
    set(compile_exec TRUE)
    get_filename_component(execname ${an_example} NAME_WE)
    	
    foreach(tbf_use_key ${TBF_USE_LIST_AVAILABLE})
        file(STRINGS "${an_example}" lines_tbf_use REGEX "@TBF_USE_${tbf_use_key}")
        if(lines_tbf_use AND NOT ${tbf_use_key} IN_LIST TBF_USE_LIST)
            message( STATUS "Examples -- ${execname} needs ${tbf_use_key}" )
            set(compile_exec FALSE)
        endif()
    endforeach()

    # Dependency are OK
    if( compile_exec )
        
        message(STATUS "Add example ${execname}")

        add_executable(${execname} ${an_example})
        if(TBFMM_LIBRARIES)
            target_link_libraries(${execname} ${TBFMM_LIBRARIES}) 
        endif()
    else()
        MESSAGE(STATUS "Examples -- ${execname} cannot be compiled due to missing libs (${an_example})")
   endif()
endforeach()

#################################################

option(BUILD_TESTS "Should be ON to enable unit tests" OFF)

if( BUILD_TESTS )
    list(APPEND TBF_USE_LIST_AVAILABLE IMPOSSIBLE_LIB)

    MESSAGE(STATUS "Testing is enable")
    set(TBFMM_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    enable_testing()
    add_subdirectory(unit-tests)
endif()


