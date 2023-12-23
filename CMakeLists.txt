cmake_minimum_required (VERSION 3.0)

project(KhuzdulGraph)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Numa REQUIRED)

# setup compiler

set (CMAKE_C_COMPILER "mpicc")
set (CMAKE_CXX_COMPILER "mpicxx")

# cxx flags
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-O3")
set (CMAKE_CXX_FLGAS "-rpath=/spack/apps/gcc/8.3.0/lib64")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed -lprofiler -Wl,--as-needed") # used for profiling, please delete it TODO
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D PRINT_DEBUG_MSG") # delete this line to prevent printing debug msg

# include directories
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(NUMA_INCLUDE_DIR)

# libraries
add_library(dwarves_debug src/dwarves_debug.cc)
add_library(graph src/graph.cc)       
add_library(utilities src/utilities.cc)
add_library(graph_loader src/graph_loader.cc)
add_library(shared_memory_sys src/shared_memory_sys.cc)
add_library(distributed_sys src/distributed_sys.cc)
add_library(timer src/timer.cc)
add_library(memory_pool src/memory_pool.cc)
add_library(vertex_set src/vertex_set.cc)
add_library(types src/types.cc)
add_library(distributed_core src/engine.cc src/distributed_application.cc)

# dependencies
set (dependencies "")
set (dependencies ${dependencies} distributed_core)
set (dependencies ${dependencies} graph_loader)    
set (dependencies ${dependencies} graph)
set (dependencies ${dependencies} vertex_set)
set (dependencies ${dependencies} memory_pool)
set (dependencies ${dependencies} shared_memory_sys)
set (dependencies ${dependencies} distributed_sys)
set (dependencies ${dependencies} utilities)
set (dependencies ${dependencies} timer)
set (dependencies ${dependencies} dwarves_debug)
set (dependencies ${dependencies} types)
set (dependencies ${dependencies} ${NUMA_LIBRARY})      

add_subdirectory(toolkits)
add_subdirectory(distributed_applications)

