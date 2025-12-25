cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(shmem_allocator LANGUAGES CXX)

option(BUILD_TESTS  "build test or not" OFF)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

add_compile_options(-hno-unused-parameter -lno-unused-function -Wunused-value -Wcast-align)
add_compile_options(-Wcast-qual -Winvalid-pch -Wwrite-strings -Wsign-compare -Wextra)

if (${CMAKE_BUILD_TYPE} MATCHES "RELEASE")
    add_compile_options(-O3)
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    add_compile_options(-fstack-protector-strong)
    add_compile_options(-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -fPIE -fPIC -ftrapv -s)
    message(STATUS "build type set to RELEASE")
else ()
    add_compile_options(-g -rdynamic)
endif ()

if(DEFINED ENV{DEBUG_MODE})
    if("$ENV{DEBUG_MODE}" STREQUAL "ON")
        add_compile_definitions(DEBUG_MODE)
        message(STATUS "Debug logging enabled from environment")
    endif()
endif()

set(PROJECT_OP_SRC_BASE ${PROJECT_SOURCE_DIR}/csrc)
set(PROJECT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(PROJECT_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/output)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

set(ALLOCATOR_LIB_NAME shmem_allocator)

set(SOC_VERSION Ascend910_9382)  # to make sgl-kernel config_envs happy
include(../../cmake/config_git_last_commit.cmake)
include(../../cmake/config_envs.cmake)
include(../../cmake/config_ascend.cmake)

file(GLOB ALLOCATOR_SOURCE_FILES
        "${PROJECT_OP_SRC_BASE}/*.cpp"
)
add_library(${ALLOCATOR_LIB_NAME} SHARED
        ${ALLOCATOR_SOURCE_FILES}
)

# python
if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, try using default python3")
    set(Python3_EXECUTABLE "/usr/local/python3.11.13/bin/python3.11")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()
find_package(Python3 COMPONENTS Interpreter Development)
list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")

# shmem
if(NOT DEFINED SHMEM_HOME_PATH AND NOT DEFINED ENV{SHMEM_HOME_PATH})
    message(FATAL_ERROR "Cannot find SHMEM_HOME_PATH, please run set_env.sh.")
elseif(NOT DEFINED SHMEM_HOME_PATH)
    set(SHMEM_HOME_PATH $ENV{SHMEM_HOME_PATH})
endif()

target_link_libraries(${ALLOCATOR_LIB_NAME} PRIVATE
        torch
        torch_npu
        shmem
)

target_link_directories(${ALLOCATOR_LIB_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
        ${SHMEM_HOME_PATH}/shmem/lib
)

target_include_directories(${ALLOCATOR_LIB_NAME} PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${ASCEND_HOME_PATH}/include
        ${TORCH_NPU_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${SHMEM_HOME_PATH}/shmem/include
        ${PROJECT_OP_SRC_BASE}
        ${PROJECT_SOURCE_DIR}
)
