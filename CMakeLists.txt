cmake_minimum_required (VERSION 3.8)
project (math_tests)

set (CMAKE_CXX_STANDARD 17)

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

add_library(project_warnings INTERFACE)
include(cmake_modules/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

add_executable(math_tests tests/tests_main.cpp
                          tests/tests_images.cpp
                          tests/tests_general.cpp
                          tests/tests_float4.cpp
                          tests/tests_float3.cpp
                          tests/tests_float2.cpp
                          tests/tests_uint4.cpp
                          tests/tests_uint3.cpp
                          tests/tests_uint2.cpp
                          tests/tests_int4.cpp
                          tests/tests_int3.cpp
                          tests/tests_int2.cpp
                          Image2d.cpp
                          )

set_target_properties(math_tests PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(math_tests PRIVATE project_warnings)

option(CODE_COVERAGE "Enable coverage reporting" OFF)
option(BUILD_TESTING "Build the testing tree." OFF)
option(USE_STB_IMAGE "Use StbImage for 'png'" OFF)

if(USE_STB_IMAGE)
  add_definitions(-DUSE_STB_IMAGE)
  include_directories("${CMAKE_SOURCE_DIR}/external")
  if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-duplicated-branches -Wno-duplicated-cond ")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter ")

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND BUILD_TESTING)
  message("CTEST is ENABLED")
  enable_testing()
  add_test(AllTests math_tests)
  set_tests_properties(AllTests PROPERTIES
                       FAIL_REGULAR_EXPRESSION "FAILED!")
endif()

#uncomment this to add code coverage
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_COMPILER_IS_GNUCXX AND CODE_COVERAGE)
  LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")
  include(CodeCoverage)
  APPEND_COVERAGE_COMPILER_FLAGS()
  #target_link_libraries(math_tests gcov)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fkeep-inline-functions -fno-inline -fno-inline-small-functions -fno-default-inline")
  #set(COVERAGE_EXCLUDES "dir1/*")
  set(COVERAGE_EXCLUDES "${CMAKE_SOURCE_DIR}/external/*")
  setup_target_for_coverage_gcovr_html(NAME            coverage
                                       EXECUTABLE      math_tests
                                       BASE_DIRECTORY  "${PROJECT_SOURCE_DIR}"
                                       EXCLUDE         "${PROJECT_SOURCE_DIR}/tests/*")  
endif()