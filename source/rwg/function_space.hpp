// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes defining operations associated with the RWG and related function spaces.
*/

#ifndef BEM_RWG_FUNCSP_H
#define BEM_RWG_FUNCSP_H

#include <functional>

#include "types.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/triangle/base.hpp"


namespace bem
{
// Forward declarations
template <typename T> class MatrixBase;
template <uint8_t dim> class TriangleMesh;
}


namespace bem::rwg
{

/**
* \defgroup basis Basis Functions
* \ingroup rwg
* @brief Classes for operations associated with RWG and related function spaces.
* @{
*/

/**
* @brief Class for operations associated with RWG functions.
*/
class Rwg
{
public:

    /**
    * @brief Defines the normalization factor for each RWG function associated with a given triangle.
    * @param[in] tri - Triangle for whose edges the RWG function normalization is required.
    * @return Normalization factor for each RWG function associated with the edges of `tri`.
    */
    static EigRowVecN<Float, 3> normalization(const Triangle<3>& tri)
    { return tri.edge_polarities() / tri.area() / two; };


    /**
    * @brief Evaluates the value of the RWG function associated with a given edge of a given triangle
    * at a given set of points in that triangle.
    * @param[in] tri - Triangle for whose edges the RWG function is to be evaluated.
    * @param[in] edge - Edge index (0, 1, or 2) of the triangle for which the RWG function value is required.
    * @param[in] points - Points in the triangle at which to evaluate the RWG function.
    * @param[in] rotated - If true, evaluates the rotated RWG function (nxRWG).
    * @return Value of the RWG function associated with edge `edge` of `tri` at each of the `points`.
    */
    static EigMatNX<Float, 3> value(
        const Triangle<3>& tri,
        uint8_t edge,
        ConstEigRef<EigMatNX<Float, 3>> points,
        const bool rotated = false
        );


    /**
    * @brief Tests a field on the RWG functions associated with the edges of a given triangle.
    * @param[in] tri - Triangle for whose edges the RWG function is to be evaluated.
    * @param[in] field_eval - Function or class with `operator()` that evaluates the vector field
    * at given set of points in the triangle.
    * @param[in] tri_quad - Triangle quadrature object to use for integration over the triangle.
    * @param[in] rotated - If true, tests the field with rotated RWG functions (nxRWG).
    * @return Tested field on each RWG function associated with the edges of `tri`.
    * @details
    * Computes
    * \f[
    * \int_{\mathrm{tri}} d\mathcal{S}\,\vec{f_n}(\vec{r})\cdot \vec{g}(\vec{r})
    * \f]
    * for each triangle edge \f$ n \f$, where \f$ \vec{g}(\vec{r}) \f$ is a vector field, and
    * \f$ \vec{f_n}(\vec{r}) \f$ is the RWG function associated with edge \f$ n \f$ of `tri`.
    */
    static EigColVecN<Complex, 3> test_field(
        const Triangle<3>& tri,
        std::function<EigMatNX<Complex, 3> (ConstEigRef<EigMatNX<Float, 3>>)> field_eval,
        TriangleQuadratureBase<3>& tri_quad,
        const bool rotated = false
        );


    /**
    * @brief Reconstructs a vector field expressed with RWG functions on a given triangle mesh.
    * @param[in] mesh - Triangle mesh on which the field is defined.
    * @param[in] coeffs - Vector of RWG coefficients for each mesh edge.
    * @param[in] points - Points on which to sample the field.
    * @param[in] rotated - If true, the field is expressed with rotated RWG functions (nxRWG).
    * @return Vector field sampled at `points`.
    */
    static EigMatNX<Complex, 3> reconstruct_field(
        const TriangleMesh<3>& mesh,
        const MatrixBase<Complex>& coeffs,
        ConstEigRef<EigMatNX<Float, 3>> points,
        const bool rotated = false
        );

};


/**
* @brief Class for operations associated with pulse functions.
*/
class Pulse
{
public:

    /**
    * @brief Defines the normalization factor for the pulse function associated with a given triangle.
    * @param[in] tri - Triangle for which the pulse function normalization is required.
    * @return Normalization factor for the pulse function associated with `tri`.
    */
    static Float normalization(const Triangle<3>& tri)
    { return one / tri.area(); };


    /**
    * @brief Evaluates the value of the pulse function associated with a given triangle
    * at a given set of points in that triangle.
    * @param[in] tri - Triangle for which the pulse function is to be evaluated.
    * @param[in] points - Points in the triangle at which to evaluate the pulse function.
    * @return Value of the pulse function associated with `tri` at each of the `points`.
    */
    static EigRowVec<Float> value(const Triangle<3>& tri, ConstEigRef<EigMatNX<Float, 3>> points)
    {
        return EigRowVec<Float>::Constant(1, points.cols(), normalization(tri));
    }


    /**
    * @brief Tests a field on the pulse function associated with a given triangle.
    * @param[in] tri - Triangle for which the pulse function is to be evaluated.
    * @param[in] field_eval - Function or class with `operator()` that evaluates the scalar field
    * at given set of points in the triangle.
    * @param[in] tri_quad - Triangle quadrature object to use for integration over the triangle.
    * @return Tested field on the pulse function associated with `tri`.
    * @details
    * Computes
    * \f[
    * \int_{\mathrm{tri}} d\mathcal{S}\,h(\vec{r})\,g(\vec{r})
    * \f]
    * where \f$ g(\vec{r}) \f$ is a scalar field, and \f$ h(\vec{r}) \f$ is the pulse
    * function associated with `tri`.
    */
    static Complex test_field(
        const Triangle<3>& tri,
        std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, 3>>)> field_eval,
        TriangleQuadratureBase<3>& tri_quad
        );


    /**
    * @brief Reconstructs a scalar field expressed with pulse functions on a given triangle mesh.
    * @param[in] mesh - Triangle mesh on which the field is defined.
    * @param[in] coeffs - Vector of pulse coefficients for each mesh edge.
    * @param[in] points - Points on which to sample the field.
    * @return Scalar field sampled at `points`.
    */
    static EigRowVec<Complex> reconstruct_field(
        const TriangleMesh<3>& mesh,
        const MatrixBase<Complex>& coeffs,
        ConstEigRef<EigMatNX<Float, 3>> points
        );

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/function_space.cpp"
#endif

#endif
