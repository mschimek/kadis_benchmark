cmake_minimum_required(VERSION 3.23)
project(KadisBenchmark)

include(FetchContent)

FetchContent_Declare(
  kaval
  GIT_REPOSITORY https://github.com/niklas-uhl/kaval.git
  GIT_TAG 1600bda
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/kaval)

FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG 1effe84
  SYSTEM)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.3.2
  SYSTEM)

FetchContent_Declare(magic-enum
  GIT_REPOSITORY  https://github.com/Neargye/magic_enum.git
  GIT_TAG         v0.9.7
  SYSTEM
)

FetchContent_Declare(tlx
  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         2dd63ab
  SYSTEM
)
FetchContent_MakeAvailable(tlx)

FetchContent_Declare(
  kadis
  GIT_REPOSITORY https://github.com/MichaelAxtmann/KaDiS.git
  GIT_TAG ac9c575
  SYSTEM)

FetchContent_MakeAvailable(kaval kamping cli11 magic-enum kadis)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_NINJA_PARALLEL_PROCESSING ON)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

# -Werror stops compiliation because of some files in KaDiS
list(APPEND
  DSSS_WARNING_FLAGS
  # "-Werror"
  "-Wall"
  "-Wextra"
  "-Wundef"
  "-Wunreachable-code"
  "-Wno-unused-parameter"
  "-Wpedantic"
  "-Wnull-dereference"
  "-Wimplicit-fallthrough"
  "-Wno-vla"
  "-Wno-pessimizing-move"
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND DSSS_WARNING_FLAGS "-Wno-gnu-zero-variadic-macro-arguments")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DDEBUG -march=native -ggdb")


# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

find_package(MPI REQUIRED)

add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark PRIVATE kamping::kamping)
target_link_libraries(benchmark PRIVATE MPI::MPI_CXX)
target_link_libraries(benchmark PRIVATE tlx)
target_link_libraries(benchmark PUBLIC kadis)
target_link_libraries(benchmark PRIVATE magic_enum::magic_enum)
target_link_libraries(benchmark PRIVATE CLI11::CLI11)
