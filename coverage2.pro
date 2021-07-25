SOURCES += tests/tests_main.cpp tests/tests_general.cpp tests/tests_float2.cpp tests/tests_float3.cpp tests/tests_float4.cpp tests/tests_uint2.cpp tests/tests_uint3.cpp tests/tests_uint4.cpp tests/tests_int2.cpp tests/tests_int3.cpp tests/tests_int4.cpp

QMAKE_CXXFLAGS += -Wall -Wextra -Weffc++ # -Werror

# gcov
QMAKE_CXXFLAGS += --coverage -fkeep-inline-functions -fno-inline -fno-inline-small-functions -fno-default-inline
LIBS += -lgcov

# C++11
QMAKE_CXX = g++
QMAKE_LINK = g++
QMAKE_CC = gcc
QMAKE_CXXFLAGS += -std=c++11
