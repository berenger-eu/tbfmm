project(TBFMM CXX)
cmake_minimum_required(VERSION 3.0)

#################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#################################################

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

set(TBFMM_EXTRA_FLAG "-m64 -march=native -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wconversion  -Wall -Wno-sign-conversion -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wconversion -Wno-error" CACHE STRING "Additional flags"  )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBFMM_EXTRA_FLAG}")

#################################################

option(USE_COVERAGE "Should be ON to enable code coverage" OFF)

if( USE_COVERAGE )
    message(STATUS "Use code coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
    list(APPEND TBFMM_LIBRARIES gcov)
    # Then one can use these commande:
    # lcov --directory CMakeFiles/ -c -o coverage-rapport.info
    # genhtml -o ./coverage -t "Coverage example" coverage-rapport.info
endif()

#################################################

set(SPETABARU_JUST_LIB TRUE)
add_subdirectory(deps/spetabaru)
INCLUDE_DIRECTORIES(
         ${SPETABARU_BINARY_DIR}/Src
         ${SPETABARU_SOURCE_DIR}/Src
         ${SPETABARU_INCLUDE_DIR}
    )
link_directories("${SPETABARU_DIR}/lib/")
list(APPEND TBFMM_LIBRARIES ${SPETABARU_LIBRARIES} spetabaru)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPETABARU_CXX_FLAGS}")

#################################################

list(APPEND TBFMM_LIBRARIES fftw3)


#################################################

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE cpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(cpp_for_lib)
    file(GLOB_RECURSE hpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
    add_library(TBFMM STATIC ${cpp_for_lib} ${hpp_for_lib})
    list(APPEND TBFMM_LIBRARIES TBFMM)
endif()


#################################################


file(GLOB_RECURSE cpp_for_examples ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(an_example ${cpp_for_examples})
    get_filename_component(execname ${an_example} NAME_WE)
    
    message(STATUS "Add example ${execname} for ${an_example}")

    add_executable(${execname} ${an_example})
    if(TBFMM_LIBRARIES)
        target_link_libraries(${execname} ${TBFMM_LIBRARIES}) 
    endif()
endforeach()

#################################################

option(BUILD_TESTS "Should be ON to enable unit tests" OFF)

if( BUILD_TESTS )
    MESSAGE(STATUS "Testing is enable")
    set(TBFMM_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    enable_testing()
    add_subdirectory(unit-tests)
endif()


