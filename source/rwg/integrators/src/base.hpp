// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for integration over the source triangle for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPINT_SRC_BASE_H
#define BEM_RWG_OPINT_SRC_BASE_H

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgopintsrc
* @{
*/

/**
* @brief Data structure to hold the results of integration over the source triangle.
*/
struct SrcResult
{
    EigRowVec<Complex> g;
    EigMatNX<Complex, 2> rs_g;
    EigMatNX<Complex, 3> grad_g;
    // EigMatNX<Complex, 3> xs_grad_g, ys_grad_g;
    void resize(Index n) {
        g.resize(1, n); rs_g.resize(2, n); grad_g.resize(3, n);
        g.setZero(); rs_g.setZero(); grad_g.setZero();
        return;
    };
};


/**
* @brief Base class for integration over the source triangle for RWG-based BEM operators.
*/
class SrcIntegratorBase
{
public:

    /**
    * @brief Computes the integral over the source triangle.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @return Integration result.
    */
    virtual SrcResult integrate(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs
        ) = 0;


    /**
    * @brief Sets flags defining which terms to compute during integration.
    * @param[in] compute_g_terms - If true, computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * and
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\vec{r}\,'\,G(k, \vec{r}, \vec{r}\,')
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$.
    * @param[in] compute_grad_g_terms - If true, computes
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,x'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * and
    * \f[
    * \int_{\mathrm{src\_tri}} d\mathcal{S}'\,y'\,\nabla G(k, \vec{r}, \vec{r}\,'),
    * \f]
    * for the scalar kernel \f$ G(k, \vec{r}, \vec{r}\,') \f$, and for local source
    * triangle coordinates \f$\vec{r}\,' = [x', y']^T\f$.
    */
    virtual void set_compute_terms(
        bool compute_g_terms,
        bool compute_grad_g_terms
        )
    {
        compute_g_terms_ = compute_g_terms;
        compute_grad_g_terms_ = compute_grad_g_terms;
        return;
    };


    /**
    * @brief Returns a result of zeros of the appropriate size.
    * @param[in] k - Complex wavenumber.
    * @param[in] src_tri - Source triangle in 2D space.
    * @param[in] r_obs - Observation points in the local coordinate system of `src_tri`.
    * @return Integration result.
    */
    virtual SrcResult zeros(
        const Complex k,
        const Triangle<2>& src_tri,
        ConstEigRef<EigMatNX<Float, 3>> r_obs
        ) { SrcResult result; result.resize(r_obs.cols()); return result; };


    /**
    * @brief Virtual destructor.
    */
    virtual ~SrcIntegratorBase() = default;


protected:

    bool compute_g_terms_ = true;
    bool compute_grad_g_terms_ = true;

};

/**
* @}
*/

}

#endif
