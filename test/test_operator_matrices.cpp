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
#include <tuple>

#include "test_operator_matrices.hpp"

#include "types.hpp"
#include "quadrature/line/gauss.hpp"

#include "rwg/assemblers/operator_assembler.hpp"

#include "rwg/integrators/src/strategic.hpp"
#include "rwg/integrators/obs/quadrature.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"

#include "matrix/eigen_matrix.hpp"

#include "geometry/mesh/triangle_mesh.hpp"


using namespace bem;
using namespace bem::rwg;


const Float LAMBDA = 1;


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_operator_matrices.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    Float f = c0 / LAMBDA;
    Complex k = two * pi * f * std::sqrt(eps0 * (one - (Float)0.1 * J) * mu0);

    GaussTriangleQuadrature<2> src_tri_quad (4);
    GaussTriangleQuadrature<3> obs_tri_quad (4);
    GaussLineQuadrature<1> line_quad (10);

    SrcIntegrationSettings settings;
    SrcStrategic src_int (settings);
    ObsQuadrature obs_int (obs_tri_quad, src_int);

    auto ops = std::make_tuple(
        VectorSingleLayerOp (obs_int),
        RotVectorSingleLayerOp (obs_int),
        ScalarSingleLayerOp (obs_int),
        RotGradScalarSingleLayerOp (obs_int),
        VectorHypersingularOp (obs_int),
        RotVectorHypersingularOp (obs_int),
        VectorDoubleLayerPvOp (obs_int),
        RotVectorDoubleLayerPvOp (obs_int),
        VectorIdentityOp (),
        RotVectorIdentityOp ()
        );

    std::apply(
        [&] (auto&&... args)
        {
            (test_unit_cube(args, k), ...);
        },
        ops
        );

    return 0;

}
