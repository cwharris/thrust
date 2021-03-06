cmake_minimum_required(VERSION 3.15)

project(Thrust NONE)

include(cmake/AppendOptionIfAvailable.cmake)

include(cmake/ThrustBuildCompilerTargets.cmake)
include(cmake/ThrustBuildTargetList.cmake)
include(cmake/ThrustMultiConfig.cmake)
include(cmake/ThrustUtilities.cmake)

# Add cache string options for CMAKE_BUILD_TYPE and default to RelWithDebInfo.
if ("" STREQUAL "${CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
  )
endif ()

# Disable compiler extensions:
set(CMAKE_CXX_EXTENSIONS OFF)

# Where to put the things we build:
set(THRUST_LIBRARY_OUTPUT_DIR "${Thrust_BINARY_DIR}/lib")
set(THRUST_EXECUTABLE_OUTPUT_DIR "${Thrust_BINARY_DIR}/bin")

# Temporary hacks to make Feta work; this requires you to define
# `CMAKE_CUDA_COMPILER_ID=Feta` and `CMAKE_CUDA_COMPILER_FORCED`.
if ("Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  # If using Feta, don't set CXX compiler
  if (NOT "${CMAKE_CXX_COMPILER}" STREQUAL "")
    unset(CMAKE_CXX_COMPILER CACHE)
    message(FATAL_ERROR "You are using Feta as your CUDA C++ compiler, but have"
      " specified a different ISO C++ compiler; Feta acts as both, so please"
      " unset the CMAKE_CXX_COMPILER variable.")
  endif ()

  # We don't set CMAKE_CUDA_HOST_COMPILER for Feta; if we do, CMake tries to
  # pass `-ccbin ${CMAKE_CUDA_HOST_COMPILER}` to Feta, which it doesn't
  # understand.
  if (NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "")
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    message(FATAL_ERROR "You are using Feta as your CUDA C++ compiler, but have"
      " specified a different host ISO C++ compiler; Feta acts as both, so"
      " please unset the CMAKE_CUDA_HOST_COMPILER variable.")
  endif ()

  set(CMAKE_CXX_COMPILER "${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -stdpar")
  set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_LINK_EXECUTABLE
      "<CMAKE_CUDA_HOST_LINK_LAUNCHER> ${CMAKE_CUDA_FLAGS} <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif ()

# This must appear after any changes to CMAKE_CXX_COMPILER or else CMake will
# delete the cache and reconfigure from scratch.
enable_language(CXX)

# We don't set CMAKE_CUDA_HOST_COMPILER for Feta; if we do, CMake tries to
# pass `-ccbin ${CMAKE_CUDA_HOST_COMPILER}` to Feta, which it doesn't
# understand.
if (NOT "Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  if (NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "")
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    message(FATAL_ERROR "Thrust tests and examples require the C++ compiler"
      " and the CUDA host compiler to be the same; to set this compiler, please"
      " use the CMAKE_CXX_COMPILER variable, not the CMAKE_CUDA_HOST_COMPILER"
      " variable.")
  endif ()
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif ()

thrust_configure_multiconfig()
thrust_build_target_list()

thrust_update_system_found_flags()
message(STATUS "CPP system found?  ${THRUST_CPP_FOUND}")
message(STATUS "CUDA system found? ${THRUST_CUDA_FOUND}")
message(STATUS "TBB system found?  ${THRUST_TBB_FOUND}")
message(STATUS "OMP system found?  ${THRUST_OMP_FOUND}")

if (THRUST_CUDA_FOUND)
  include(cmake/ThrustCUDAConfig.cmake)
endif()

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.00)
    message(FATAL_ERROR "This version of MSVC no longer supported.")
  endif ()
endif ()

if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.4)
    message(FATAL_ERROR "This version of GCC no longer supported.")
  endif ()
endif ()

option(THRUST_ENABLE_HEADER_TESTING "Test that all public headers compile." "ON")
option(THRUST_ENABLE_TESTING "Build Thrust testing suite." "ON")
option(THRUST_ENABLE_EXAMPLES "Build Thrust examples." "ON")

if (THRUST_ENABLE_HEADER_TESTING)
  include(cmake/ThrustHeaderTesting.cmake)
endif()

# Both testing and examples use ctest
if (THRUST_ENABLE_TESTING OR THRUST_ENABLE_EXAMPLES)
  include(CTest)
  enable_testing()
endif()

if (THRUST_ENABLE_TESTING)
  add_subdirectory(testing)
endif()

if (THRUST_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
