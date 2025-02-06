cmake_minimum_required(VERSION 3.13)


if(NOT TARGET LiteMath)
    project(LiteMath CXX)

    add_library(LiteMath INTERFACE)

    target_sources(LiteMath PUBLIC Image2d.cpp)

    target_compile_features(LiteMath INTERFACE cxx_std_17)
    target_include_directories(LiteMath INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(LiteMath INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external)
endif()