# OpenBEM - Copyright (C) 2026 Shashwat Sharma

# This file is part of OpenBEM.

# OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.

# You should have received a copy of the GNU General Public License along with OpenBEM.
# If not, see <https://www.gnu.org/licenses/>.


cmake_minimum_required(VERSION 3.16.0)

project(OpenBEM)


# ====== Settings ======

set(PRECISION "double" CACHE STRING "Floating point precision. Should be one of 'single', 'double' (default), or 'extended'.")
set_property(CACHE PRECISION PROPERTY STRINGS single double extended)

set(HEADER_ONLY "no" CACHE STRING "Header-only mode. Should be one of 'yes' or no' (default).")
set_property(CACHE HEADER_ONLY PROPERTY STRINGS "yes" "no")

set(DEBUG_LEVEL "release" CACHE STRING "Debugging level. Should be one of 'release' (default), 'debug', or 'extra_debug'.")
set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS release debug extra_debug)

set(BUILD_TESTS "no" CACHE STRING "Build all tests. Should be 'yes' or 'no' (default).")
set_property(CACHE BUILD_TESTS PROPERTY STRINGS "yes" "no")

set(BUILD_EXAMPLES "no" CACHE STRING "Build all examples. Should be 'yes' or 'no' (default).")
set_property(CACHE BUILD_EXAMPLES PROPERTY STRINGS "yes" "no")

set(OPTIMIZE_NATIVE_ARCH "yes" CACHE STRING "Optimize builds for native architecture. Should be one of 'yes' (default) or 'no'.")
set_property(CACHE OPTIMIZE_NATIVE_ARCH PROPERTY STRINGS "yes" "no")


# ====== Compile settings ======

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

ENABLE_LANGUAGE(C)
ENABLE_LANGUAGE(CXX)
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.3.1")
        message(FATAL_ERROR "OpenBEM requires at least GCC version 7.3.1")
    endif ()
endif ()


string(TOLOWER "${HEADER_ONLY}" HEADER_ONLY)
if (HEADER_ONLY STREQUAL "no")

    message("Compile mode: linked")
    add_compile_definitions(BEM_LINKED)

endif ()


string(TOLOWER "${PRECISION}" PRECISION)
if (PRECISION STREQUAL "single")

    message("Precision: 'single'")
    add_compile_definitions(USE_SINGLE_PRECISION)

elseif(PRECISION STREQUAL "double")

    message("Precision: 'double'")
    add_compile_definitions(USE_DOUBLE_PRECISION)

elseif(PRECISION STREQUAL "extended")

    message("Precision: 'extended'")
    add_compile_definitions(USE_EXTENDED_PRECISION)

else()

    message(FATAL_ERROR "'PRECISION' must be one of 'single', 'double', or 'extended'. Exiting.")

endif()


string(TOLOWER "${DEBUG_LEVEL}" DEBUG_LEVEL)
if (DEBUG_LEVEL STREQUAL "debug")

    message("Debug level: 'debug'")
    set(CMAKE_BUILD_TYPE Debug)
    add_compile_definitions(DEBUG)
    add_compile_definitions(EIGEN_NO_DEBUG)
    add_compile_options(-O0 -g)

elseif (DEBUG_LEVEL STREQUAL "extra_debug")

    message("Debug level: 'extra_debug'")
    set(CMAKE_BUILD_TYPE Debug)
    add_compile_definitions(EXTRA_DEBUG)
    add_compile_options(-O0 -g)

elseif (DEBUG_LEVEL STREQUAL "release")

    message("Debug level: 'release'")
    set(CMAKE_BUILD_TYPE Release)
    add_compile_definitions(NO_DEBUG)
    add_compile_definitions(EIGEN_NO_DEBUG)
    add_compile_options(-O3)

else ()

    message(FATAL_ERROR "'DEBUG_LEVEL' must be one of 'release' (default), 'debug', or 'extra_debug'. Exiting.")

endif ()

string(TOLOWER "${OPTIMIZE_NATIVE_ARCH}" OPTIMIZE_NATIVE_ARCH)
if (OPTIMIZE_NATIVE_ARCH STREQUAL "yes")

    message("Optimized for native architecture")
    add_compile_options(-march=native)

endif ()


add_compile_options(
    -Wall
    -Wno-sign-compare # not ideal to disable this; done because Eigen reports sizes as signed ints
    # -ffast-math
)

if (NOT CMAKE_BUILD_TYPE)
    message("build type not set")
    set(CMAKE_BUILD_TYPE Release)
endif ()

message(CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")


# ====== OpenMP ======

find_package(OpenMP)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(OPENMP)
    set(OPENMP_LINK OpenMP::OpenMP_CXX)
    message (STATUS "OpenMP found")
else ()
    message (STATUS "OpenMP not found")
    set(OPENMP_LINK "")
endif ()


# ====== Source ======

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)

set(INC_DIR ${SRC_DIR} ${EXT_DIR})

set(SRC_FILES
    ${SRC_DIR}/geometry/operations.cpp
    ${SRC_DIR}/geometry/point_cloud.cpp
    ${SRC_DIR}/geometry/primitives/edge.cpp
    ${SRC_DIR}/geometry/primitives/triangle.cpp
    ${SRC_DIR}/geometry/mesh/triangle_mesh.cpp
    ${SRC_DIR}/geometry/mesh/mesh_transfer.cpp
    ${SRC_DIR}/quadrature/utility.cpp
    ${SRC_DIR}/quadrature/triangle/gauss.cpp
    ${SRC_DIR}/quadrature/triangle/adaptive.cpp
    ${SRC_DIR}/quadrature/triangle/iterative_gauss.cpp
    ${SRC_DIR}/quadrature/line/gauss.cpp
    ${SRC_DIR}/quadrature/line/iterative_gauss.cpp
    ${SRC_DIR}/quadrature/line/trapz.cpp
    ${SRC_DIR}/quadrature/line/iterative_trapz.cpp
    ${SRC_DIR}/kernels/hgf.cpp
    ${SRC_DIR}/rwg/function_space.cpp
    ${SRC_DIR}/rwg/assemblers/index_generator.cpp
    ${SRC_DIR}/rwg/assemblers/operator_matrix.cpp
    ${SRC_DIR}/rwg/assemblers/excitation_matrix.cpp
    ${SRC_DIR}/rwg/assemblers/projector_matrix.cpp
    ${SRC_DIR}/rwg/excitations/inf_gap.cpp
    ${SRC_DIR}/rwg/excitations/plane_wave.cpp
)

if (HEADER_ONLY STREQUAL "no")

    add_library(openbem SHARED ${SRC_FILES})
    target_include_directories(openbem PRIVATE ${INC_DIR})
    target_link_libraries(openbem ${OPENMP_LINK})

else ()

    add_library(openbem INTERFACE)
    target_include_directories(openbem INTERFACE ${INC_DIR})

endif ()

set(LINK_LIBS
    openbem
    ${OPENMP_LINK}
)


# ====== Examples ======

if (BUILD_EXAMPLES STREQUAL "yes")

    set(EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)

    add_executable(ex01 ${EXAMPLE_DIR}/ex01_start_here.cpp)
    target_include_directories(ex01 PRIVATE ${INC_DIR})
    target_link_libraries(ex01 ${LINK_LIBS})

    add_executable(ex02 ${EXAMPLE_DIR}/ex02_assemblers.cpp)
    target_include_directories(ex02 PRIVATE ${INC_DIR})
    target_link_libraries(ex02 ${LINK_LIBS})

endif ()


# ====== Tests ======

if (BUILD_TESTS STREQUAL "yes")

    set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

    add_executable(test_quadrature ${TEST_DIR}/test_quadrature.cpp)
    target_include_directories(test_quadrature PRIVATE ${INC_DIR})
    target_link_libraries(test_quadrature ${LINK_LIBS})

    add_executable(test_src_int ${TEST_DIR}/test_src_int.cpp)
    target_include_directories(test_src_int PRIVATE ${INC_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_src_int ${LINK_LIBS})

    add_executable(test_rwg_operators ${TEST_DIR}/test_rwg_operators.cpp)
    target_include_directories(test_rwg_operators PRIVATE ${INC_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_rwg_operators ${LINK_LIBS})

    add_executable(test_operator_matrices ${TEST_DIR}/test_operator_matrices.cpp)
    target_include_directories(test_operator_matrices PRIVATE ${INC_DIR})
    target_link_libraries(test_operator_matrices ${LINK_LIBS})

    add_executable(test_triangle_mesh ${TEST_DIR}/test_triangle_mesh.cpp)
    target_include_directories(test_triangle_mesh PRIVATE ${INC_DIR})
    target_link_libraries(test_triangle_mesh ${LINK_LIBS})

    add_executable(test_materials ${TEST_DIR}/test_materials.cpp)
    target_include_directories(test_materials PRIVATE ${INC_DIR})
    target_link_libraries(test_materials ${LINK_LIBS})

    add_executable(test_excitations ${TEST_DIR}/test_excitations.cpp)
    target_include_directories(test_excitations PRIVATE ${INC_DIR})
    target_link_libraries(test_excitations ${LINK_LIBS})

    add_executable(test_point_cloud ${TEST_DIR}/test_point_cloud.cpp)
    target_include_directories(test_point_cloud PRIVATE ${INC_DIR})
    target_link_libraries(test_point_cloud ${LINK_LIBS})

    add_executable(test_projectors ${TEST_DIR}/test_projectors.cpp)
    target_include_directories(test_projectors PRIVATE ${INC_DIR})
    target_link_libraries(test_projectors ${LINK_LIBS})

    add_executable(test_ops_sphere_rcs ${TEST_DIR}/test_ops_sphere_rcs.cpp)
    target_include_directories(test_ops_sphere_rcs PRIVATE ${INC_DIR})
    target_link_libraries(test_ops_sphere_rcs ${LINK_LIBS})

    add_executable(test_op_set_sphere_rcs ${TEST_DIR}/test_op_set_sphere_rcs.cpp)
    target_include_directories(test_op_set_sphere_rcs PRIVATE ${INC_DIR})
    target_link_libraries(test_op_set_sphere_rcs ${LINK_LIBS})

    add_executable(test_lineint_sphere_rcs ${TEST_DIR}/test_lineint_sphere_rcs.cpp)
    target_include_directories(test_lineint_sphere_rcs PRIVATE ${INC_DIR})
    target_link_libraries(test_lineint_sphere_rcs ${LINK_LIBS})

    add_executable(test_compile ${TEST_DIR}/test_compile.cpp)
    target_include_directories(test_compile PRIVATE ${INC_DIR})
    target_link_libraries(test_compile ${LINK_LIBS})

endif ()
