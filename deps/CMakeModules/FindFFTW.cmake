# - Find FFTW
# Find the native FFTW includes and library
#
#  FFTW_INCLUDES    - where to find fftw3.h
#  FFTW_LIBRARIES   - List of libraries when using FFTW.
#  FFTW_FOUND       - True if FFTW found.


if( NOT FFTW_ROOT AND DEFINED ENV{FFTWDIR} )
  set( FFTW_ROOT $ENV{FFTWDIR})
endif()
if( NOT FFTW_ROOT AND DEFINED ENV{FFTW_DIR} )
  set( FFTW_ROOT $ENV{FFTW_DIR})
endif()
if( NOT FFTW_ROOT AND DEFINED ENV{FFTW_ROOT} )
  set( FFTW_ROOT $ENV{FFTW_ROOT})
endif()
if( NOT FFTW_ROOT AND DEFINED ENV{FFTW_HOME} )
  set( FFTW_ROOT $ENV{FFTW_HOME})
endif()

message(STATUS "Consider FFTW_ROOT = ${FFTW_ROOT}")

if( PKG_CONFIG_FOUND AND NOT FFTW_ROOT )
  pkg_check_modules( PKG_FFTW QUIET "fftw3" )
endif()

find_path (FFTW_INCLUDES fftw3.h)

find_library (FFTW_LIB NAMES fftw3 PATHS ${FFTW_ROOT} PATH_SUFFIXES "lib" "lib64")
find_library (FFTWF_LIB NAMES fftw3f PATHS ${FFTW_ROOT} PATH_SUFFIXES "lib" "lib64")
#
set(FFTW_LIBRARIES ${FFTWF_LIB} ${FFTW_LIB})

# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDES)

mark_as_advanced (FFTW_LIBRARIES FFTW_INCLUDES )

