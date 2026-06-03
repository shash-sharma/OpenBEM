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
    )
{
    r_ = (r_src.colwise() - r_obs).colwise().norm();
    assert(r_.array().all() > 0 && "HGF::compute(): Distance must be greater than 0.");
    return Eigen::exp(-J * k * r_.array()) / r_.array() / four_pi;
}


EigMatNX<Complex, 3> HGF::compute_grad(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    )
{
    r_diff_ = -(r_src.colwise() - r_obs);
    r_ = r_diff_.colwise().norm();
    assert(r_.array().all() > 0 && "HGF::compute_grad(): Distance must be greater than 0.");

    jkr_ = J * k * r_.array();
    val_ = -Eigen::exp(-jkr_.array()) * (one + jkr_.array()) /
        (r_.array() * r_.array() * r_.array()) / four_pi;

    return r_diff_.array().rowwise() * val_.array();
}


EigRowVec<Complex> SingularitySubtractedHGF::compute(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    )
{
    r_ = (r_src.colwise() - r_obs).colwise().norm();
    assert(r_.array().all() > 0 && "SingularitySubtractedHGF::compute(): Distance must be greater than 0.");
    return (Eigen::exp(-J * k * r_.array()) - one) / r_.array() / four_pi;
}


EigMatNX<Complex, 3> SingularitySubtractedHGF::compute_grad(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    )
{
    r_diff_ = -(r_src.colwise() - r_obs);
    r_ = r_diff_.colwise().norm();
    assert(r_.array().all() > 0 && "SingularitySubtractedHGF::compute_grad(): Distance must be greater than 0.");

    jkr_ = J * k * r_.array();
    r_sq_ = r_.array() * r_.array();
    r_cu_ = r_sq_.array() * r_.array();

    val_ = -(
        Eigen::exp(-jkr_.array()) * (one + jkr_.array()) - (one + half * k * k * r_sq_.array())
        ) / r_cu_.array() / four_pi;

    return r_diff_.array().rowwise() * val_.array();
}


EigRowVec<Complex> SingularitySubtractedTaylorHGF::compute(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    )
{
    r_ = (r_src.colwise() - r_obs).colwise().norm();

    if (std::real(k) == 0.0)
        return EigRowVec<Complex>::Zero(1, r_.cols());

    const Float tol = KERNEL_DEFAULT_TOL;
    const Complex jk = J * k;
    jkr_ = jk * r_;

    multiplier_ = EigRowVec<Complex>::Constant(1, r_.cols(), -jk);
    val_ = multiplier_;

    for (uint32_t jj = 2; true; jj++)
    {
        multiplier_.array() *= -jkr_.array() / (Float)jj;
        val_ += multiplier_;
        if ((multiplier_.cwiseAbs2().array() <= tol * tol * val_.cwiseAbs2().array()).all())
            break;
    }
    val_.array() /= four_pi;

    return val_;
}


EigMatNX<Complex, 3> SingularitySubtractedTaylorHGF::compute_grad(
    ConstEigRef<EigColVecN<Float, 3>> r_obs,
    ConstEigRef<EigMatNX<Float, 3>> r_src,
    const Complex k
    )
{
    r_diff_ = -(r_src.colwise() - r_obs);
    r_ = r_diff_.colwise().norm();

    if (std::real(k) == 0.0)
        return EigMatNX<Complex, 3>::Zero(3, r_.cols());

    const Float tol = KERNEL_DEFAULT_TOL;
    const Complex jk = J * k;
    const Complex jkc = jk * k * k;
    jkr_ = jk * r_;

    multiplier_ = jkc * (one + jkr_.array()) / (Float)6.0;
    val_ = -jkc / two + multiplier_.array();
    for (uint32_t jj = 4; true; jj++)
    {
        multiplier_.array() *= -jkr_.array() / (Float)jj;
        val_ += multiplier_;
        if ((multiplier_.cwiseAbs2().array() <= tol * tol * val_.cwiseAbs2().array()).all())
            break;
    }
    val_ *= -1.0 / four_pi;

    return r_diff_.array().rowwise() * val_.array();
}

}
