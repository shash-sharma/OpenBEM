// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Functionality for computing plane wave excitation vector coefficients for RWG-based BEM systems.
*/

#include "rwg/excitations/plane_wave.hpp"

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

EigMatNX<Complex, 3> RwgPlaneWave::compute(const Complex k, const Triangle<3>& obs_tri)
{
    Index rhs = 0;

    auto eval_lambda = [&] (ConstEigRef<EigMatNX<Float, 3>> r_obs) -> EigMatNX<Complex, 3>
        { return eval(k, r_obs, rhs); };

    EigMatNX<Complex, 3> result = EigMatNX<Complex, 3>::Zero(3, pos_.cols());

    for (rhs = 0; rhs < pos_.cols(); ++rhs)
        result.col(rhs) = Rwg::test_field(obs_tri, eval_lambda, *tri_quad_);

    return result;
};


EigMatNX<Complex, 3> NxRwgPlaneWave::compute(const Complex k, const Triangle<3>& obs_tri)
{
    Index rhs = 0;

    auto eval_lambda = [&] (ConstEigRef<EigMatNX<Float, 3>> r_obs) -> EigMatNX<Complex, 3>
        { return eval(k, r_obs, rhs); };

    EigMatNX<Complex, 3> result = EigMatNX<Complex, 3>::Zero(3, pos_.cols());

    for (rhs = 0; rhs < pos_.cols(); ++rhs)
        result.col(rhs) = Rwg::test_field(obs_tri, eval_lambda, *tri_quad_, true);

    return result;
};

}

