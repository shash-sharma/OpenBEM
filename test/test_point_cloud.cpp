// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>
#include <vector>
#include <array>

#include "types.hpp"
#include "constants.hpp"

#include "geometry/point_cloud.hpp"


using namespace bem;


const Float LAMBDA = 1;


void test_set_points()
{

    Float tol = float_eps * 10;

    EigMatMN<Float, 3, 8> points;
    points.col(0) = EigColVecN<Float, 3> ({ 0, 1, -0.2 });
    points.col(1) = EigColVecN<Float, 3> ({ 1, 1.5, -0.3 });
    points.col(2) = EigColVecN<Float, 3> ({ 2, 2, -0.4 });
    points.col(3) = EigColVecN<Float, 3> ({ 3, 2.5, -0.5 });
    points.col(4) = EigColVecN<Float, 3> ({ 4, 3, -0.6 });
    points.col(5) = EigColVecN<Float, 3> ({ 5, 3.5, -0.7 });
    points.col(6) = EigColVecN<Float, 3> ({ 6, 4, -0.8 });
    points.col(7) = EigColVecN<Float, 3> ({ 7, 4.5, -0.9 });

    EigColVecN<Float, 3> start, stop;
    start << 0, 1, -0.2;
    stop << 7, 4.5, -0.9;

    PointCloud<3> cloud;
    cloud.set_line_data(start, stop, 8);
    if (!(Eigen::abs((cloud.points() - points).array()) < tol).all())
        std::cout << "FAIL: cloud.set_line_data()" << std::endl;

    EigColVecN<Index, 3> num_pts;
    num_pts << 2, 5, 8;
    cloud.set_block_data(start, stop, num_pts);

    EigColVecN<Float, 3> step;
    step << 7, 0.875, -0.1;

    for (Index ii = 0; ii < num_pts[0]; ++ii)
    {
        for (Index jj = 0; jj < num_pts[1]; ++jj)
        {
            for (Index kk = 0; kk < num_pts[2]; ++kk)
            {
                EigColVecN<Float, 3> point;
                point << start[0] + step[0] * ii, start[1] + step[1] * jj, start[2] + step[2] * kk;
                EigColVecN<Float, 3> diff;
                diff = point - cloud.points(kk + num_pts[2] * jj + num_pts[1] * num_pts[2] * ii);
                if (!(Eigen::abs(diff.array()) < tol).all())
                    std::cout << "FAIL: cloud.set_block_data(): " << ii << ", " << jj << ", " << kk << std::endl;
            }
        }
    }

    start << 2, 0, quarter_pi;
    stop << 2, two_pi, half_pi;
    num_pts << 1, 9, 2;
    EigColVecN<Float, 3> center;
    center << 0, 0, 0;
    cloud.set_polar_data(start, stop, center, num_pts);
    if (!(Eigen::abs(cloud.points().colwise().norm().array() - 2) < tol).all())
        std::cout << "FAIL: cloud.set_polar_data()" << std::endl;

    return;

}


int main(int argc, char** argv)
{
    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_point_cloud.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    test_set_points();
    return 0;
}
