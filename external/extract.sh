#!/bin/bash

eigen_str=eigen-5.0.1
tar -zxvf ${eigen_str}.tar.gz
cp -r ${eigen_str}/Eigen ../source/external/
cp -r ${eigen_str}/unsupported ../source/external/
mv ../source/external/unsupported ../source/external/EigenUnsupported
cp ${eigen_str}/COPYING* ../source/external/Eigen/

