#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR=${SCRIPT_DIR}/../build

cmake -DDEBUG_LEVEL=release -DPRECISION=double -DHEADER_ONLY=no -DBUILD_TESTS=yes -DBUILD_EXAMPLES=no -B ${BUILD_DIR} -S ${SCRIPT_DIR}/../

time make -j 4 -C ${BUILD_DIR}

export OMP_NUM_THREADS=1

time ${BUILD_DIR}/test_compile
time ${BUILD_DIR}/test_materials
time ${BUILD_DIR}/test_point_cloud
time ${BUILD_DIR}/test_triangle_mesh
time ${BUILD_DIR}/test_excitations
time ${BUILD_DIR}/test_projectors
time ${BUILD_DIR}/test_quadrature
time ${BUILD_DIR}/test_src_int
time ${BUILD_DIR}/test_rwg_operators
time ${BUILD_DIR}/test_operator_matrices
time ${BUILD_DIR}/test_ops_sphere_rcs
time ${BUILD_DIR}/test_lineint_sphere_rcs
# time ${BUILD_DIR}/test_vector_ops_sphere_rcs

