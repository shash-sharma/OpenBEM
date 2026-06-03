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

#ifndef BEM_KERNEL_HGF_H
#define BEM_KERNEL_HGF_H

#include "types.hpp"
#include "kernels/base.hpp"


namespace bem
{

/**
* \addtogroup kern
* @{
*/

/**
* @brief Class for computing the scalar Green's function for homogeneous, linear, and isotropic materials.
*/
class HGF: public ScalarKernelBase<3>
{
public:

    /**
    * @brief Computes the kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Kernel value.
    * @details
    * Computes
    * \f[
    * \frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|}
    * \f]
    * where \f$ \vec{r} \f$ is the observer position vector, \f$ \vec{r}\,' \f$ is the source position vector,
    * and \f$ k \f$ is the complex wavenumber.
    */
    EigRowVec<Complex> compute(
        ConstEigRef<EigColVecN<Float, 3>> r_obs,
        ConstEigRef<EigMatNX<Float, 3>> r_src,
        const Complex k
        ) override;


    /**
    * @brief Computes the gradient of the kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Components of the gradient of the kernel.
    * @details
    * Computes
    * \f[
    * \nabla\left(\frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|}\right) =
    * -(\vec{r} - \vec{r}\,')\left(1 + jk|\vec{r} - \vec{r}\,'|\right)
    * \frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|^3}
    * \f]
    * where \f$ \vec{r} \f$ is the observer position vector, \f$ \vec{r}\,' \f$ is the source position vector,
    * and \f$ k \f$ is the complex wavenumber.
    */
    EigMatNX<Complex, 3> compute_grad(
        ConstEigRef<EigColVecN<Float, 3>> r_obs,
        ConstEigRef<EigMatNX<Float, 3>> r_src,
        const Complex k
        ) override;


protected:

    // class-wide containers to avoid repeated memory (de)allocation
    EigRowVec<Float> r_;
    EigMatNX<Float, 3> r_diff_;
    EigRowVec<Complex> jkr_;
    EigRowVec<Complex> val_;

};


/**
* @brief Class for computing the scalar Green's function for homogeneous, linear, and isotropic materials,
* with the singular term(s) explicitly subtracted out.
*/
class SingularitySubtractedHGF: public ScalarKernelBase<3>
{
public:

    /**
    * @brief Computes the kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Kernel value.
    * @details
    * Computes
    * \f[
    * \frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|} - \frac{1}{4\pi|\vec{r} - \vec{r}\,'|}
    * \f]
    * where \f$ \vec{r} \f$ is the observer position vector, \f$ \vec{r}\,' \f$ is the source position vector,
    * and \f$ k \f$ is the complex wavenumber.
    */
    EigRowVec<Complex> compute(
        ConstEigRef<EigColVecN<Float, 3>> r_obs,
        ConstEigRef<EigMatNX<Float, 3>> r_src,
        const Complex k
        ) override;


    /**
    * @brief Computes the gradient of the kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Components of the gradient of the kernel.
    * @details
    * Computes
    * \f{align*}{
    * \nabla\left(\frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|}
    * - \frac{1}{4\pi|\vec{r} - \vec{r}\,'|}\right) &=\\
    * -(\vec{r} - \vec{r}\,')&\left[\left(1 + jk|\vec{r} - \vec{r}\,'|\right)
    * \frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|^3} -
    * \frac{1}{4\pi|\vec{r} - \vec{r}\,'|^3} - \left(\frac{1}{2}\right)
    * \frac{k^2}{4\pi|\vec{r} - \vec{r}\,'|}\right]
    * \f}
    * where \f$ \vec{r} \f$ is the observer position vector, \f$ \vec{r}\,' \f$ is the source position vector,
    * and \f$ k \f$ is the complex wavenumber.
    */
    EigMatNX<Complex, 3> compute_grad(
        ConstEigRef<EigColVecN<Float, 3>> r_obs,
        ConstEigRef<EigMatNX<Float, 3>> r_src,
        const Complex k
        ) override;


protected:

    // class-wide containers to avoid repeated memory (de)allocation
    EigRowVec<Float> r_;
    EigMatNX<Float, 3> r_diff_;
    EigRowVec<Float> r_sq_;
    EigRowVec<Float> r_cu_;
    EigRowVec<Complex> jkr_;
    EigRowVec<Complex> val_;

};


/**
* @brief Class for computing the scalar Green's function for homogeneous, linear, and isotropic materials,
* with the singular term(s) implicitly subtracted using the Taylor series expansion of the kernel.
*/
class SingularitySubtractedTaylorHGF: public ScalarKernelBase<3>
{
public:

    /**
    * @brief Computes the kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Kernel value.
    * @details
    * Computes
    * \f[
    * \frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|} - \frac{1}{4\pi|\vec{r} - \vec{r}\,'|}
    * \f]
    * where \f$ \vec{r} \f$ is the observer position vector, \f$ \vec{r}\,' \f$ is the source position vector,
    * and \f$ k \f$ is the complex wavenumber. The singular term is never actually computed; it is implicitly
    * subtracted out using the Taylor series expansion
    * \f[
    * \frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|} = \frac{1}{4\pi|\vec{r} - \vec{r}\,'|}
    * + \frac{(-jk)}{4\pi} + \frac{(-jk)^2}{4\pi\cdot 2!}|\vec{r} - \vec{r}\,'| + \ldots
    * \f]
    * truncated when the next term is smaller than the tolerance `KERNEL_DEFAULT_TOL` relative to the sum so far.
    */
    EigRowVec<Complex> compute(
        ConstEigRef<EigColVecN<Float, 3>> r_obs,
        ConstEigRef<EigMatNX<Float, 3>> r_src,
        const Complex k
        ) override;


    /**
    * @brief Computes the gradient of the kernel for given observation and source points.
    * @param[in] r_obs - Observer position vector.
    * @param[in] r_src - Set of source position vectors.
    * @param[in] k - Complex wavenumber.
    * @return Components of the gradient of the kernel.
    * @details
    * Computes
    * \f{align*}{
    * \nabla\left(\frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|}
    * - \frac{1}{4\pi|\vec{r} - \vec{r}\,'|}\right) &=\\
    * -(\vec{r} - \vec{r}\,')&\left[\left(1 + jk|\vec{r} - \vec{r}\,'|\right)
    * \frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|^3} -
    * \frac{1}{4\pi|\vec{r} - \vec{r}\,'|^3} - \left(\frac{1}{2}\right)
    * \frac{k^2}{4\pi|\vec{r} - \vec{r}\,'|}\right]
    * \f}
    * where \f$ \vec{r} \f$ is the observer position vector, \f$ \vec{r}\,' \f$ is the source position vector,
    * and \f$ k \f$ is the complex wavenumber. The singular term is never actually computed; it is implicitly
    * subtracted out using the Taylor series expansion
    * \f[
    * \nabla\left(\frac{e^{-jk|\vec{r} - \vec{r}\,'|}}{4\pi|\vec{r} - \vec{r}\,'|}\right) =
    * -\frac{\vec{r} - \vec{r}\,'}{4\pi|\vec{r} - \vec{r}\,'|}\left[
    * \frac{1}{|\vec{r} - \vec{r}\,'|^2} - \frac{(-jk)^2}{2!} - 2\frac{(-jk)^3}{3!}|\vec{r} - \vec{r}\,'|
     + \ldots\right]
    * \f]
    * truncated when the next term is smaller than the tolerance `KERNEL_DEFAULT_TOL` relative to the sum so far.
    */
    EigMatNX<Complex, 3> compute_grad(
        ConstEigRef<EigColVecN<Float, 3>> r_obs,
        ConstEigRef<EigMatNX<Float, 3>> r_src,
        const Complex k
        ) override;


protected:

    // class-wide containers to avoid repeated memory (de)allocation
    EigRowVec<Float> r_;
    EigMatNX<Float, 3> r_diff_;
    EigRowVec<Complex> jkr_;
    EigRowVec<Complex> multiplier_;
    EigRowVec<Complex> val_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "kernels/hgf.cpp"
#endif

#endif
