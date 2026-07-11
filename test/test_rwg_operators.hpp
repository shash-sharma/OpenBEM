// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#ifndef TEST_RWG_OPERATORS_H
#define TEST_RWG_OPERATORS_H

#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>
#include <vector>
#include <array>
#include <typeinfo>

#include "types.hpp"

#include "geometry/operations.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "quadrature/line/gauss.hpp"

#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"
#include "rwg/integrators/src/singularity.hpp"
#include "rwg/integrators/src/line.hpp"
#include "rwg/integrators/src/strategic.hpp"

#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/quadrature.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"


using namespace bem;
using namespace bem::rwg;


Triangle<3> make_triangle(Float a, Float b, Float theta, EigColVecN<Float, 3> offset = EigColVecN<Float, 3>::Zero(3, 1), Float angf = 1.0)
{
    EigMatMN<Float, 3, 3> v;
    v.col(0) = EigColVecN<Float, 3> (0.0, 0.0, 0.0) + offset;
    v.col(1) = EigColVecN<Float, 3> (a, 0.0, -a / 3 * angf) + offset;
    v.col(2) = EigColVecN<Float, 3> (b * std::cos(theta), b * std::sin(theta), a / 4 * angf) + offset;
    return Triangle<3> (v);
}


template <typename ObsIntegratorType>
auto make_ops_tuple(ObsIntegratorType& obs_int)
{
    return std::make_tuple(
        VectorSingleLayerOp (obs_int),
        RotVectorSingleLayerOp (obs_int),
        ScalarSingleLayerOp (obs_int),
        // RotGradScalarSingleLayerOp (obs_int),
        VectorHypersingularOp (obs_int),
        // RotVectorHypersingularOp (obs_int),
        VectorDoubleLayerPvOp (obs_int),
        RotVectorDoubleLayerPvOp (obs_int),
        VectorIdentityOp (),
        RotVectorIdentityOp ()
        );
}


template <typename ObsIntegratorType, typename ObsIntegratorRefType>
void compare_ops(
    Complex k,
    Triangle<3> &src_tri,
    Triangle<3> &obs_tri,
    const ObsIntegratorType &obs_int,
    const ObsIntegratorRefType &obs_int_ref,
    Float tol = 1e-3,
    bool print_anyway = false,
    std::string message = ""
    )
{

    auto ops = make_ops_tuple(obs_int);
    auto ops_ref = make_ops_tuple(obs_int_ref);

    std::apply(
        [&] (auto&&... op)
        {
            std::apply(
                [&] (auto&&... op_ref)
                {([&]
                {
                    EigMat<Complex> result, result_ref;
                    result = op.compute(k, obs_tri, src_tri);
                    result_ref = op_ref.compute(k, obs_tri, src_tri);

                    EigMat<Float> error = (
                        result_ref - result
                        ).array().abs() / result_ref.array().abs().maxCoeff();

                    if (result_ref.array().abs().maxCoeff() == 0.0)
                        error = (result_ref - result).array().abs();

                    bool pass = (error.array() <= tol).all();

                    if (!pass || print_anyway)
                    {
                        std::cout << "====== operator " << typeid(op).name()
                                  << " ======" << std::endl;

                        std::cout << "--- " << message
                                  << ", k = " << k
                                  << " rad/m ---" << std::endl;

                        std::cout << "--- FAIL ---\n  op_name: " << typeid(op).name()
                                  << "\n  max error: " << error.maxCoeff()
                                  << "\n  errors:\n" << error
                            // << "\n  src_tri:\n" << src_tri.v()
                            // << "\n  obs_tri:\n" << obs_tri.v()
                                  << "\n  ref vals:\n" << result_ref
                                  << "\n  vals:\n" << result << std::endl;
                    }

                }(), ...); },
                ops_ref
                );
        },
        ops
        );

    return;

}


template <typename ScalarKernelType>
void compare_quad_and_line(
    Complex k,
    ScalarKernelType &kernel,
    Triangle<3> &src_tri,
    Triangle<3> &obs_tri,
    uint8_t src_quad_order = 4,
    uint8_t obs_quad_order = 4,
    uint8_t src_line_order = 10,
    Float tol = 1e-3,
    bool print_anyway = false,
    bool singularity = true,
    std::string message = ""
    )
{
    GaussTriangleQuadrature<2> src_tri_quad (src_quad_order);
    GaussTriangleQuadrature<3> obs_tri_quad (obs_quad_order);
    GaussLineQuadrature<1> src_line_quad (src_line_order);

    SrcLineIntegrator src_int_line (src_line_quad);
    ObsQuadrature obs_int_line (obs_tri_quad, src_int_line);

    if (singularity)
    {
        SrcSingularity src_int_quad (src_tri_quad, kernel);
        ObsQuadrature obs_int_quad (obs_tri_quad, src_int_quad);

        compare_ops(k, src_tri, obs_tri, obs_int_quad, obs_int_line, tol, print_anyway, message);
    }
    else
    {
        SrcQuadrature src_int_quad (src_tri_quad, kernel);
        ObsQuadrature obs_int_quad (obs_tri_quad, src_int_quad);

        compare_ops(k, src_tri, obs_tri, obs_int_quad, obs_int_line, tol, print_anyway, message);
    }

    return;
}

#endif

