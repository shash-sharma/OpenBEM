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

#ifndef BEM_RWG_EXC_PLANE_WAVE_H
#define BEM_RWG_EXC_PLANE_WAVE_H

#include <stdexcept>
#include <memory>

#include "types.hpp"
#include "quadrature/triangle/gauss.hpp"
#include "rwg/function_space.hpp"
#include "rwg/excitations/base.hpp"


namespace bem
{
// Forward declarations
template <uint8_t dim> class Triangle;
}


namespace bem::rwg
{

/**
* \addtogroup rwgexc
* @{
*/

/**
* @brief Class for computing plane wave excitation coefficients for RWG-based BEM systems.
*/
class PlaneWaveBase: public ExcitationBase
{
public:

    /**
    * @brief Constructs a `PlaneWaveBase` object with given plane wave parameters.
    * @param[in] dir - Direction of propagation each plane wave; will be normalized automatically.
    * @param[in] pol - Polarization vector of the field associated with each plane wave;
    * will be normalized automatically.
    * @param[in] pos - Point where each plane wave is originating from, used as a phase reference only. This
    * can be any point on a plane perpendicular to the direction of propagation, which will be used as the
    * phase reference plane.
    * @param[in] amp - Complex amplitudes of the field associated with each plane wave.
    * @param[in] quadrature_order - Quadrature order for integration over the observation triangle (optional).
    * @details
    * The `PlaneWaveBase` object is agnostic to which field (electric or magnetic) is being considered.
    * This allows the same object to be used for generating different excitation vector coefficients for
    * different BEM systems (EFIE, MFIE, etc.). To generate an excitation vector coefficient associated with
    * an electric field excitation, `pol` should be set to the electric field polarization vector, and for a
    * magnetic field excitation, `pol` should be set to the magnetic field polarization vector.
    * For multiple right-hand-sides, e.g., multiple polarizations or directions of propagation, the
    * `dir`, `pol`, `pos`, and `amp` parameters should be provided as matrices with each column
    * corresponding to a different right-hand-side. The number of columns in these matrices must match.
    */
    PlaneWaveBase(
        ConstEigRef<EigMatNX<Float, 3>> dir,
        ConstEigRef<EigMatNX<Float, 3>> pol,
        ConstEigRef<EigMatNX<Float, 3>> pos,
        ConstEigRef<EigRowVec<Complex>> amp,
        const uint8_t quadrature_order = 4
        ):
            dir_(dir.colwise().normalized()),
            pol_(pol.colwise().normalized()),
            pos_(pos),
            amp_(amp),
            tri_quad_(std::make_shared<GaussTriangleQuadrature<3>> (
                GaussTriangleQuadrature<3> (quadrature_order)
                ))
    {
        if (
            dir_.cols() != pol_.cols() ||
            dir_.cols() != pos_.cols() ||
            dir_.cols() != amp.size()
            )
        {
            throw std::invalid_argument(
                "PlaneWaveBase: `dir`, `pol`, `pos`, and `amp` must have the same number of columns."
                );
        }

        return;
    };


    /**
    * @brief Returns the number of excitations (right-hand sides) to be generated.
    * @return Number of excitations (right-hand sides).
    */
    Index num_excitations() const override { return dir_.cols(); };


protected:

    /**
    * @brief Evaluates the plane wave vector field value at given observation points.
    * @param[in] k - Complex wavenumber.
    * @param[in] r_obs - Observation points at which to evaluate the plane wave field values.
    * @param[in] idx - Index of the right-hand side for which to evaluate the field
    * (optional when there's only one right-hand side).
    * @return Plane wave vector field values sampled at `r_obs`, where each column corresponds to
    * each excitation when there is more than one excitation (i.e., more than one right-hand side).
    */
    EigMatNX<Complex, 3> eval(const Complex k, ConstEigRef<EigMatNX<Float, 3>> r_obs, const Index idx = 0)
    {
        EigMatNX<Float, 3> r_diff = -pos_.col(idx).replicate(1, r_obs.cols()) + r_obs;
        EigRowVec<Float> r_dir = dir_.col(idx).transpose() * r_diff;
        return amp_[idx] * pol_.col(idx) * Eigen::exp(-J * k * r_dir.array()).matrix();
    };


    const EigMatNX<Float, 3> dir_, pol_, pos_;
    const EigRowVec<Complex> amp_;
    std::shared_ptr<GaussTriangleQuadrature<3>> tri_quad_;

};


/**
* @brief Class for computing plane wave excitation vector coefficients when tested with RWG functions.
*/
class RwgPlaneWave: public PlaneWaveBase
{
public:

    using PlaneWaveBase::PlaneWaveBase;

    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    DofSpace obs_dof() const override { return DofSpace::EDGE; };


    /**
    * @brief Computes the plane wave excitation coefficients when the field is tested with RWG functions.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the local coordinate system of `src_tri`.
    * @return Excitation coefficient matrix, where each row corresponds to each degree of freedom
    * associated with `obs_tri`, and each column corresponds to each excitation when there is more
    * than one excitation (i.e., more than one right-hand side).
    */
    EigMat<Complex> compute(const Complex k, const Triangle<3>& obs_tri) override;

};


/**
* @brief Class for computing plane wave excitation vector coefficients when tested with rotated RWG functions.
*/
class NxRwgPlaneWave: public PlaneWaveBase
{
public:

    using PlaneWaveBase::PlaneWaveBase;

    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    DofSpace obs_dof() const override { return DofSpace::EDGE; };


    /**
    * @brief Computes the plane wave excitation coefficients when the field is tested with rotated RWG functions.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the local coordinate system of `src_tri`.
    * @return Excitation coefficient matrix, where each row corresponds to each degree of freedom
    * associated with `obs_tri`, and each column corresponds to each excitation when there is more
    * than one excitation (i.e., more than one right-hand side).
    */
    EigMat<Complex> compute(const Complex k, const Triangle<3>& obs_tri) override;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/excitations/plane_wave.cpp"
#endif

#endif
