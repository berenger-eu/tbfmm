cmake_minimum_required(VERSION 3.15)
project(TBFMM CXX)

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0057 NEW)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/deps/CMakeModules/")

#################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif(APPLE)

try_compile(COMPILE_RESULT  ${CMAKE_CURRENT_BINARY_DIR}
      "${PROJECT_SOURCE_DIR}/deps/CMakeModules/cppversion.cpp"
      COMPILE_DEFINITIONS "${CMAKE_CXX_FLAGS}"
      OUTPUT_VARIABLE COMPILE_OUTPUT)

if(NOT ${COMPILE_RESULT})
    message(STATUS "Cannot compile C++17, output when compiling simple example is :")
    message(STATUS "${COMPILE_OUTPUT}")
    message(STATUS "Will continue anyway...")
endif()

#################################################

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(TBFMM_EXTRA_FLAG "-march=native -Wdangling-reference -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual  -Wall -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wno-error -fopenmp=libomp" CACHE STRING "Additional flags"  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
# suppress warnings from getenv
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    set(TBFMM_EXTRA_FLAG "-march=native -Wdangling-reference -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wconversion  -Wall -Wno-sign-conversion -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wconversion -Wno-error" CACHE STRING "Additional flags"  )
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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/specx/CMakeLists.txt")
    message(STATUS "SPECX Found")
    if(NOT ${TBFMM_ENABLE_SPECX})
        message(STATUS "SPECX Disabled")   
    else() 
    	OPTION(TBFMM_USE_CUDA "Set to on to compile tasks with CUDA callables" ${CMAKE_CUDA_COMPILER})
    	set(SPECX_COMPILE_WITH_CUDA ${TBFMM_USE_CUDA})
	    
	    set(SPECX_JUST_LIB TRUE)
	    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/deps/specx")
	    INCLUDE_DIRECTORIES(
		     ${SPECX_BINARY_DIR}/Src
		     ${SPECX_SOURCE_DIR}/Src
		     ${SPECX_INCLUDE_DIR}
		)
		
        message(STATUS "SPECX_DIR = ${SPECX_DIR}")
        message(STATUS "SPECX_LIBRARIES = ${SPECX_LIBRARIES}")
        message(STATUS "SPECX_CXX_FLAGS = ${SPECX_CXX_FLAGS}")
		
	    link_directories("${SPECX_DIR}/lib/")
	    list(APPEND TBFMM_LIBRARIES ${SPECX_LIBRARIES} specx)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPECX_CXX_FLAGS}")

	    list(APPEND TBF_USE_LIST SPECX)
	    if(TBFMM_USE_CUDA)
	        enable_language(CUDA)
	        set(CUDA_SEPARABLE_COMPILATION ON)
		            
            if(NOT CMAKE_CUDA_ARCHITECTURES)
              set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING
                  "Set CUDA target architecture" FORCE)
            endif()
	        list(APPEND TBF_USE_LIST CUDA)	   
	        
	        add_definitions("--expt-relaxed-constexpr") 
	    endif()
    endif()
else()
    message(STATUS "SPECX Cannot be found in ${CMAKE_CURRENT_SOURCE_DIR}/deps/specx (please use git submodule init && git submodule update)")
endif()

list(APPEND TBF_USE_LIST_AVAILABLE SPECX)

#################################################

if(DEFINED ENV{STARPU_DIR} AND NOT DEFINED TBFMM_USE_STARPU)
    set(TBFMM_USE_STARPU $ENV{STARPU_DIR})
endif()

OPTION(TBFMM_USE_STARPU "The path to StarPU (default is env variable STARPU_DIR), empty will disable StarPU")


if(TBFMM_USE_STARPU)
    message(STATUS "STARPU Found")
    if(NOT ${TBFMM_ENABLE_STARPU})
        message(STATUS "STARPU Disabled (but env variable found)")   
    else()	    
    	OPTION(TBFMM_USE_CUDA "Set to on to compile tasks with CUDA callables" ${CMAKE_CUDA_COMPILER})
    	set(SPECX_COMPILE_WITH_CUDA ${TBFMM_USE_CUDA})
    			
        message(STATUS "TBFMM_USE_STARPU = ${TBFMM_USE_STARPU}")
        
        set(TBFMM_STARPU_VERSION "1.4" CACHE STRING "Set to starpu version to look at the inalude dir and set the lib name"  )
        message(STATUS "TBFMM_STARPU_VERSION = ${TBFMM_STARPU_VERSION}")
		
	    INCLUDE_DIRECTORIES(${TBFMM_USE_STARPU}/include/starpu/${TBFMM_STARPU_VERSION}/)
	    link_directories("${TBFMM_USE_STARPU}/lib/")
	    list(APPEND TBFMM_LIBRARIES starpu-${TBFMM_STARPU_VERSION})

	    list(APPEND TBF_USE_LIST STARPU)
	    if(TBFMM_USE_CUDA)
	        enable_language(CUDA)
	        set(CUDA_SEPARABLE_COMPILATION ON)
		            
            if(NOT CMAKE_CUDA_ARCHITECTURES)
              set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING
                  "Set CUDA target architecture" FORCE)
            endif()
	        list(APPEND TBF_USE_LIST CUDA)	   
	        
	        add_definitions("--expt-relaxed-constexpr") 
	    endif()
    endif()
else()
    message(STATUS "STARPU Cannot be found, set env variable STARPU_DIR")
endif()

list(APPEND TBF_USE_LIST_AVAILABLE STARPU)

#################################################

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/inastemp/CMakeLists.txt")
    if(NOT ${TBFMM_ENABLE_INASTEMP})
        message(STATUS "INASTEMP Disabled")   
    else() 
        message(STATUS "INASTEMP Found")
        set(INASTEMP_JUST_LIB TRUE)
        set(INASTEMP_DO_INSTALL TRUE)
        # Create target
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/deps/inastemp")
        INCLUDE_DIRECTORIES(${INASTEMP_BINARY_DIR}/Src ${INASTEMP_SOURCE_DIR}/Src ${INASTEMP_INCLUDE_DIR})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${INASTEMP_CXX_FLAGS}")
            
        list(APPEND TBF_USE_LIST INASTEMP)
        
        #if(TBFMM_USE_CUDA)
            #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${INASTEMP_CXX_FLAGS}")
            #message(STATUS "CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}")   
        #endif()
    endif()
else()
    message(STATUS "Inastemp Cannot be found in ${CMAKE_CURRENT_SOURCE_DIR}/deps/inastemp (please use git submodule init && git submodule update)")
endif()

list(APPEND TBF_USE_LIST_AVAILABLE INASTEMP)

#################################################

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    message(STATUS "MSVC detected")
    if(NOT ${TBFMM_ENABLE_OPENMP})
        message(STATUS "OpenMP Disabled")   
        message(STATUS "OpenMP enabled with  -openmp:llvm -openmp:experimental")   
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:llvm -openmp:experimental")
        list(APPEND TBF_USE_LIST OPENMP)
    endif()
else()   
    find_package(OpenMP)
    if (OPENMP_FOUND)
        message(STATUS "OpenMP Found") 
        if(NOT ${TBFMM_ENABLE_OPENMP})
            message(STATUS "OpenMP Disabled")   
        else() 
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
            list(APPEND TBF_USE_LIST OPENMP)
        endif()
    endif()
endif()
list(APPEND TBF_USE_LIST_AVAILABLE OPENMP)

#################################################

find_package(FFTW)
if(FFTW_FOUND)
    message(STATUS "FFTW Found") 
    if(NOT ${TBFMM_ENABLE_FFTW})
        message(STATUS "FFTW Disabled")   
    else() 
        message(STATUS "FFTW_INCLUDES = ${FFTW_INCLUDES}")   
        message(STATUS "FFTW_LIBRARIES = ${FFTW_LIBRARIES}")   

        INCLUDE_DIRECTORIES("${FFTW_INCLUDES}")
        link_directories("${FFTW_LIBRARIES}")
        list(APPEND TBFMM_LIBRARIES ${FFTW_LIBRARIES})
        list(APPEND TBF_USE_LIST FFTW)
    endif()
else()
    message(STATUS "FFTW Cannot be found, try by setting -DFFTW_ROOT=... or env FFTW_ROOT")
endif()

list(APPEND TBF_USE_LIST_AVAILABLE FFTW)

#################################################

foreach(tbf_use_key ${TBF_USE_LIST})
    add_definitions(-DTBF_USE_${tbf_use_key})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTBF_USE_${tbf_use_key}")
endforeach()

message(STATUS "Available compilation keys are: ${TBF_USE_LIST}")

#################################################

string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPERCASE)

if(TBFMM_USE_CUDA AND (CMAKE_BUILD_TYPE_UPPERCASE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE_UPPERCASE STREQUAL "RELWITHDEBINFO"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -G -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
    message(STATUS "CMAKE_CUDA_FLAGS = ${CMAKE_CUDA_FLAGS}")
endif()
if(TBFMM_USE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CXX_FLAGS}")
endif()

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE cpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(cpp_for_lib)
    file(GLOB_RECURSE hpp_for_lib ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
    if(TBFMM_USE_CUDA)
        foreach(cppfile ${cpp_for_lib})
            set_source_files_properties(${cppfile} PROPERTIES LANGUAGE CUDA)
        endforeach()
    endif()
    add_library(TBFMM STATIC ${cpp_for_lib} ${hpp_for_lib})
    if(TBFMM_USE_CUDA)
        set_source_files_properties(TBFMM PROPERTIES LANGUAGE CUDA)
    endif()
    list(APPEND TBFMM_LIBRARIES TBFMM)
endif()

message(STATUS "TBFMM_LIBRARIES = ${TBFMM_LIBRARIES}")

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
        if(TBFMM_USE_CUDA)
            set_source_files_properties(${an_example} PROPERTIES LANGUAGE CUDA)
        endif()

        add_executable(${execname} ${an_example})
        
        if(TBFMM_USE_CUDA)
            set_source_files_properties(${execname} PROPERTIES LANGUAGE CUDA)
        endif()
        
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


