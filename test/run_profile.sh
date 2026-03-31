#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BUILD_DIR=${SCRIPT_DIR}/../build

cmake -DDEBUG_LEVEL=release -DPRECISION=double -DHEADER_ONLY=no -DBUILD_TESTS=no -DBUILD_EXAMPLES=yes -B ${BUILD_DIR} -S ${SCRIPT_DIR}/../

time make -j 4 -C ${BUILD_DIR}

export OMP_NUM_THREADS=1

valgrind --tool=callgrind --log-file=${BUILD_DIR}/ex01_memcheck.txt ${BUILD_DIR}/ex01 &

