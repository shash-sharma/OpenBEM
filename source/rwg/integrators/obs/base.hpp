// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for integration over the observation triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_OBS_BASE_H
#define BEM_RWG_OPINT_OBS_BASE_H

#include <memory>
#include <stdexcept>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintobs
* @{
*/

/**
* @brief Data structure to hold the results of integration over the observation triangle.
*/
struct ObsResult
{
    Complex g = 0;
    EigRowVecN<Complex, 12> rs_g = EigRowVecN<Complex, 12>::Zero();
    EigRowVecN<Complex, 9> grad_g = EigRowVecN<Complex, 9>::Zero();
    EigRowVecN<Complex, 15> rot_grad_g = EigRowVecN<Complex, 15>::Zero();
};


/**
* @brief Base class for integration over the observation triangle for RWG-based BEM operators.
*/
class ObsIntegratorBase
{
public:

    /**
    * @brief Computes the integral over the observation triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle in the local coordinate system of `src_tri`.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] g_term - Whether to compute the scalar kernel term (optional).
    * @param[in] rs_g_terms - Whether to compute the vector kernel termss (optional).
    * @param[in] grad_g_terms - Whether to compute the gradient kernel terms (optional).
    * @param[in] rot_grad_g_terms - Whether to compute the rotated gradient kernel terms (optional).
    * @return Integration result.
    * @details
    * If `g_term` is true, computes
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `rs_g_terms` is true, computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{r}\,'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `grad_g_terms` is true, computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * If `rot_grad_g_terms` is true, computes terms related to
    * \f[
    * \int_{\mathrm{obs\_tri}} d\mathcal{S}\,\hat{n}\times\vec{r}\cdot
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,')\times\vec{r}\,'
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    */
    virtual ObsResult integrate(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<2>& src_tri,
        const bool g_term = true,
        const bool rs_g_terms = true,
        const bool grad_g_terms = true,
        const bool rot_grad_g_terms = true
        ) = 0;


    /**
    * @brief Returns a unique pointer to a newly constructed object of the derived type.
    * @return Unique pointer to the new object.
    */
    virtual std::unique_ptr<ObsIntegratorBase> clone() const
    { throw std::runtime_error("ObsIntegratorBase::clone(): Not implemented."); };


    /**
    * @brief Virtual destructor.
    */
    virtual ~ObsIntegratorBase() = default;

};

/**
* @}
*/

}

#endif
