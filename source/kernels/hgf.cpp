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

EigRowVec<Complex> HGF::compute(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigRowVec<Float> r = (r_src.colwise() - r_obs).colwise().norm();
    assert(r.array().all() > 0 && "HGF::compute(): Distance must be greater than 0.");
    return Eigen::exp(-J * k * r.array()) / r.array() / four_pi;
}


EigMatNX<Complex, 3> HGF::compute_grad(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigMatNX<Float, 3> r_diff = -(r_src.colwise() - r_obs);
    EigRowVec<Float> r = r_diff.colwise().norm();
    assert(r.array().all() > 0 && "HGF::compute_grad(): Distance must be greater than 0.");

    const EigRowVec<Complex> jkr = J * k * r.array();
    const EigRowVec<Complex> scalar_term = -Eigen::exp(-jkr.array()) * (one + jkr.array()) /
        (r.array() * r.array() * r.array()) / four_pi;

    return r_diff.array().rowwise() * scalar_term.array();
}


EigRowVec<Complex> SingularitySubtractedHGF::compute(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigRowVec<Float> r = (r_src.colwise() - r_obs).colwise().norm();
    assert(r.array().all() > 0 && "SingularitySubtractedHGF::compute(): Distance must be greater than 0.");
    return (Eigen::exp(-J * k * r.array()) - one) / r.array() / four_pi;
}


EigMatNX<Complex, 3> SingularitySubtractedHGF::compute_grad(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigMatNX<Float, 3> r_diff = -(r_src.colwise() - r_obs);
    EigRowVec<Float> r = r_diff.colwise().norm();
    assert(r.array().all() > 0 && "SingularitySubtractedHGF::compute_grad(): Distance must be greater than 0.");

    const EigRowVec<Complex> jkr = J * k * r.array();
    const EigRowVec<Float> r_sq = r.array() * r.array();
    const EigRowVec<Float> r_cu = r_sq.array() * r.array();

    EigRowVec<Complex> scalar_term = -(
        Eigen::exp(-jkr.array()) * (one + jkr.array()) - (one + half * k * k * r_sq.array())
        ) / r_cu.array() / four_pi;

    return r_diff.array().rowwise() * scalar_term.array();
}


EigRowVec<Complex> SingularitySubtractedTaylorHGF::compute(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigRowVec<Float> r = (r_src.colwise() - r_obs).colwise().norm();

    if (std::real(k) == 0.0)
        return EigRowVec<Complex>::Zero(1, r.cols());

    const Float tol = KERNEL_DEFAULT_TOL;
    const Complex jk = J * k;
    const EigRowVec<Complex> jkr = jk * r;

    EigRowVec<Complex> multiplier = EigRowVec<Complex>::Constant(1, r.cols(), -jk);
    EigRowVec<Complex> val = multiplier;

    for (uint32_t jj = 2; true; jj++)
    {
        multiplier.array() *= -jkr.array() / (Float)jj;
        val += multiplier;
        if ((multiplier.cwiseAbs2().array() <= tol * tol * val.cwiseAbs2().array()).all())
            break;
    }
    val.array() /= four_pi;

    return val;
}


EigMatNX<Complex, 3> SingularitySubtractedTaylorHGF::compute_grad(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    ) const
{
    EigMatNX<Float, 3> r_diff = -(r_src.colwise() - r_obs);
    EigRowVec<Float> r = r_diff.colwise().norm();

    if (std::real(k) == 0.0)
        return EigMatNX<Complex, 3>::Zero(3, r.cols());

    const Float tol = KERNEL_DEFAULT_TOL;
    const Complex jk = J * k;
    const Complex jkc = jk * k * k;
    const EigRowVec<Complex> jkr = jk * r;

    EigRowVec<Complex> multiplier = jkc * (one + jkr.array()) / (Float)6.0;
    EigRowVec<Complex> scalar_term = -jkc / two + multiplier.array();
    for (uint32_t jj = 4; true; jj++)
    {
        multiplier.array() *= -jkr.array() / (Float)jj;
        scalar_term += multiplier;
        if ((multiplier.cwiseAbs2().array() <= tol * tol * scalar_term.cwiseAbs2().array()).all())
            break;
    }
    scalar_term *= -1.0 / four_pi;

    return r_diff.array().rowwise() * scalar_term.array();
}

}
