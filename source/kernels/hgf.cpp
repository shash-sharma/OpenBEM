// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Scalar Green's function kernels for homogeneous, linear, and isotropic materials.
*/

#include "kernels/hgf.hpp"

#include <cassert>
#include <limits>
#include <cmath>

#include "types.hpp"
#include "constants.hpp"


namespace bem
{

Complex HGF::kernel(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigColVecN<Float, 3>> r_src,
    const Complex k
    ) const
{
    Float r = (r_obs - r_src).norm();
    assert(r > 0 && "HGF::kernel(): Distance must be greater than 0.");
    return std::exp(-J * k * r) / r / four_pi;
}

EigColVecN<Complex, 3> HGF::grad_kernel(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigColVecN<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigColVecN<Float, 3> r_diff = r_obs - r_src;
    Float r = r_diff.norm();
    assert(r > 0 && "HGF::grad_kernel(): Distance must be greater than 0.");

    const Complex jkr = J * k * r;
    const Complex scalar_term = -std::exp(-jkr) * (one + jkr) / (Float)std::pow(r, 3) / four_pi;

    return r_diff * scalar_term;
}


Complex SingularitySubtractedHGF::kernel(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigColVecN<Float, 3>> r_src,
    const Complex k
    ) const
{
    Float r = (r_obs - r_src).norm();
    assert(r > 0 && "HGF::kernel(): Distance must be greater than 0.");
    return (std::exp(-J * k * r) - one) / r / four_pi;
}

EigColVecN<Complex, 3> SingularitySubtractedHGF::grad_kernel(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigColVecN<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigColVecN<Float, 3> r_diff = r_obs - r_src;
    Float r = r_diff.norm();
    assert(r > 0 && "HGF::grad_kernel(): Distance must be greater than 0.");

    const Complex jkr = J * k * r;
    const Float r_sq = std::pow(r, 2);
    const Float r_cu = r_sq * r;

    Complex scalar_term = -(
        std::exp(-jkr) * (one + jkr) - (one + half * k * k * r_sq)
        ) / r_cu / four_pi;
    return r_diff * scalar_term;
}


Complex SingularitySubtractedTaylorHGF::kernel(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigColVecN<Float, 3>> r_src,
    const Complex k
    ) const
{
    Float r = (r_obs - r_src).norm();

    if (std::abs(k) == 0.0)
        return (Complex)0.0;

    const Float tol = KERNEL_DEFAULT_TOL;
    const Complex jk = J * k;
    const Complex jkr = jk * r;

    Complex multiplier = -jk;
    Complex val = multiplier;

    for (uint32_t jj = 2; true; jj++)
    {
        multiplier *= -jkr / (Float)jj;
        val += multiplier;
        if (std::abs(multiplier) <= tol * std::abs(val))
            break;
    }
    val /= four_pi;

    return val;
}

EigColVecN<Complex, 3> SingularitySubtractedTaylorHGF::grad_kernel(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigColVecN<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigColVecN<Float, 3> r_diff = r_obs - r_src;
    Float r = r_diff.norm();

    if (std::abs(k) == 0.0)
        return EigColVecN<Complex, 3>::Zero(3, 1);

    const Float tol = KERNEL_DEFAULT_TOL;
    const Complex jk = J * k;
    const Complex jkc = jk * k * k;
    const Complex jkr = jk * r;

    Complex multiplier = jkc * (one + jkr) / (Float)6.0;
    Complex scalar_term = -jkc / two + multiplier;
    for (uint32_t jj = 4; true; jj++)
    {
        multiplier *= -jkr / (Float)jj;
        scalar_term += multiplier;
        if (std::abs(multiplier) <= tol * std::abs(scalar_term))
            break;
    }
    scalar_term *= -1.0 / four_pi;

    return r_diff * scalar_term;
}

}
