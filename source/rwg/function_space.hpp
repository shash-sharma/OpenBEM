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

/**
* \defgroup basis Basis Functions
* \ingroup rwg
* @brief Classes for operations associated with RWG and related function spaces.
* @{
*/

/**
* @brief Base class for function spaces.
*/
template <typename Derived, uint8_t num_dof, uint8_t num_dim>
class FunctionSpaceBase
{
public:

    /**
    * @brief Number of degrees of freedom per mesh face.
    */
    static constexpr uint8_t dof = num_dof;


    /**
    * @brief Dimension of the function.
    */
    static constexpr uint8_t dim = num_dim;


    /**
    * @brief Returns the normalization factor for a given triangle.
    * @param[in] tri - Triangle to which normalization is associated.
    * @return Normalization factor for each degree of freedom.
    */
    static EigRowVecN<Float, dof> normalization(const Triangle<3>& tri)
    { return Derived::normalization(tri); };


    /**
    * @brief Returns the function evaluated at a set of triangle points for each degree of freedom.
    * @param[in] tri - Triangle in which to evaluate the function.
    * @param[in] points - Points in the triangle at which to evaluate the function.
    * @param[in] idx - Degree of freedom index for which the function value is required.
    * @return Values of the function at `points` associated with the `idx` degree of freedom.
    */
    static EigMatNX<Float, dim> value(
        const Triangle<3>& tri,
        ConstEigRef<EigMatNX<Float, 3>> points,
        uint8_t idx
        ) { return Derived::value(tri, points, idx); };


    /**
    * @brief Tests a field with the function space associated with a given triangle.
    * @param[in] tri - Triangle on which to test the field.
    * @param[in] field_eval - Function or class with `operator()` that evaluates the field
    * at given set of points in the triangle.
    * @param[in] tri_quad - Triangle quadrature object to use for integration over the triangle.
    * @return Tested field for each degree of freedom associated with `tri`.
    */
    static EigColVecN<Complex, dof> test_field(
        const Triangle<3>& tri,
        std::function<EigMatNX<Complex, dim> (ConstEigRef<EigMatNX<Float, 3>>)> field_eval,
        TriangleQuadratureBase<3>& tri_quad
        ) { return Derived::test_field(tri, field_eval, tri_quad); };


    /**
    * @brief Reconstructs a field expressed with the function space on a given triangle mesh.
    * @param[in] mesh - Triangle mesh on which the field is defined.
    * @param[in] coeffs - Vector of function space coefficients for each degree of freedom.
    * @param[in] points - Points on which to sample the field.
    * @return Field sampled at `points`.
    * @details
    * A point contained in more than one face (e.g. a mesh vertex) is averaged across all
    * faces containing it, rather than summed, so that reconstructing at face centroids
    * (where each point belongs to exactly one face) and at vertices are both meaningful.
    */
    static EigMatNX<Complex, dim> reconstruct_field(
        const TriangleMesh<3>& mesh,
        const MatrixBase<Complex>& coeffs,
        ConstEigRef<EigMatNX<Float, 3>> points
        ) { return Derived::reconstruct_field(mesh, coeffs, points); };

};


/**
* @brief Class for operations associated with RWG functions.
*/
class Rwg: public FunctionSpaceBase<Rwg, 3, 3>
{
public:

    /**
    * @brief Returns the normalization factor for a given triangle.
    * @param[in] tri - Triangle to which normalization is associated.
    * @return Normalization factor for each degree of freedom.
    */
    static EigRowVecN<Float, 3> normalization(const Triangle<3>& tri)
    { return tri.edge_polarities() / tri.area() / two; };


    /**
    * @brief Returns the function evaluated at a set of triangle points for each degree of freedom.
    * @param[in] tri - Triangle in which to evaluate the function.
    * @param[in] points - Points in the triangle at which to evaluate the function.
    * @param[in] idx - Degree of freedom index for which the function value is required.
    * @return Values of the function at `points` associated with the `idx` degree of freedom.
    */
    static EigMatNX<Float, 3> value(
        const Triangle<3>& tri,
        ConstEigRef<EigMatNX<Float, 3>> points,
        uint8_t idx
        );


    /**
    * @brief Tests a field with the function space associated with a given triangle.
    * @param[in] tri - Triangle on which to test the field.
    * @param[in] field_eval - Function or class with `operator()` that evaluates the field
    * at given set of points in the triangle.
    * @param[in] tri_quad - Triangle quadrature object to use for integration over the triangle.
    * @return Tested field for each degree of freedom associated with `tri`.
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
        TriangleQuadratureBase<3>& tri_quad
        );


    /**
    * @brief Reconstructs a field expressed with the function space on a given triangle mesh.
    * @param[in] mesh - Triangle mesh on which the field is defined.
    * @param[in] coeffs - Vector of function space coefficients for each degree of freedom.
    * @param[in] points - Points on which to sample the field.
    * @return Field sampled at `points`.
    * @details
    * A point contained in more than one face (e.g. a mesh vertex) is averaged across all
    * faces containing it, rather than summed, so that reconstructing at face centroids
    * (where each point belongs to exactly one face) and at vertices are both meaningful.
    */
    static EigMatNX<Complex, 3> reconstruct_field(
        const TriangleMesh<3>& mesh,
        const MatrixBase<Complex>& coeffs,
        ConstEigRef<EigMatNX<Float, 3>> points
        );

};


/**
* @brief Class for operations associated with rotated RWG functions.
*/
class NxRwg: public FunctionSpaceBase<NxRwg, 3, 3>
{
public:

    /**
    * @brief Returns the normalization factor for a given triangle.
    * @param[in] tri - Triangle to which normalization is associated.
    * @return Normalization factor for each degree of freedom.
    */
    static EigRowVecN<Float, 3> normalization(const Triangle<3>& tri)
    { return tri.edge_polarities() / tri.area() / two; };


    /**
    * @brief Returns the function evaluated at a set of triangle points for each degree of freedom.
    * @param[in] tri - Triangle in which to evaluate the function.
    * @param[in] points - Points in the triangle at which to evaluate the function.
    * @param[in] idx - Degree of freedom index for which the function value is required.
    * @return Values of the function at `points` associated with the `idx` degree of freedom.
    */
    static EigMatNX<Float, 3> value(
        const Triangle<3>& tri,
        ConstEigRef<EigMatNX<Float, 3>> points,
        uint8_t idx
        );


    /**
    * @brief Tests a field with the function space associated with a given triangle.
    * @param[in] tri - Triangle on which to test the field.
    * @param[in] field_eval - Function or class with `operator()` that evaluates the field
    * at given set of points in the triangle.
    * @param[in] tri_quad - Triangle quadrature object to use for integration over the triangle.
    * @return Tested field for each degree of freedom associated with `tri`.
    * @details
    * Computes
    * \f[
    * \int_{\mathrm{tri}} d\mathcal{S}\,\hat{n}\times\vec{f_n}(\vec{r})\cdot \vec{g}(\vec{r})
    * \f]
    * for each triangle edge \f$ n \f$, where \f$ \vec{g}(\vec{r}) \f$ is a vector field,
    * \f$ \vec{f_n}(\vec{r}) \f$ is the RWG function associated with edge \f$ n \f$ of `tri`,
    * and \f$ \hat{n} \f$ is the unit normal associated with `tri`.
    */
    static EigColVecN<Complex, 3> test_field(
        const Triangle<3>& tri,
        std::function<EigMatNX<Complex, 3> (ConstEigRef<EigMatNX<Float, 3>>)> field_eval,
        TriangleQuadratureBase<3>& tri_quad
        );


    /**
    * @brief Reconstructs a field expressed with the function space on a given triangle mesh.
    * @param[in] mesh - Triangle mesh on which the field is defined.
    * @param[in] coeffs - Vector of function space coefficients for each degree of freedom.
    * @param[in] points - Points on which to sample the field.
    * @return Field sampled at `points`.
    * @details
    * A point contained in more than one face (e.g. a mesh vertex) is averaged across all
    * faces containing it, rather than summed, so that reconstructing at face centroids
    * (where each point belongs to exactly one face) and at vertices are both meaningful.
    */
    static EigMatNX<Complex, 3> reconstruct_field(
        const TriangleMesh<3>& mesh,
        const MatrixBase<Complex>& coeffs,
        ConstEigRef<EigMatNX<Float, 3>> points
        );

};


/**
* @brief Class for operations associated with pulse functions.
*/
class Pulse: public FunctionSpaceBase<Pulse, 1, 1>
{
public:

    /**
    * @brief Returns the normalization factor for a given triangle.
    * @param[in] tri - Triangle to which normalization is associated.
    * @return Normalization factor for each degree of freedom.
    */
    static EigRowVecN<Float, 1> normalization(const Triangle<3>& tri)
    { return EigRowVecN<Float, 1>::Constant(1, 1, one / tri.area()); };


    /**
    * @brief Returns the function evaluated at a set of triangle points for each degree of freedom.
    * @param[in] tri - Triangle in which to evaluate the function.
    * @param[in] points - Points in the triangle at which to evaluate the function.
    * @param[in] idx - Degree of freedom index for which the function value is required.
    * @return Values of the function at `points` associated with the `idx` degree of freedom.
    */
    static EigMatNX<Float, 1> value(
        const Triangle<3>& tri,
        ConstEigRef<EigMatNX<Float, 3>> points,
        uint8_t idx = 0
        )
    { return EigMatNX<Float, 1>::Constant(dim, points.cols(), normalization(tri)[0]); }


    /**
    * @brief Tests a field with the function space associated with a given triangle.
    * @param[in] tri - Triangle on which to test the field.
    * @param[in] field_eval - Function or class with `operator()` that evaluates the field
    * at given set of points in the triangle.
    * @param[in] tri_quad - Triangle quadrature object to use for integration over the triangle.
    * @return Tested field for each degree of freedom associated with `tri`.
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
        std::function<EigMatNX<Complex, 1> (ConstEigRef<EigMatNX<Float, 3>>)> field_eval,
        TriangleQuadratureBase<3>& tri_quad
        );


    /**
    * @brief Reconstructs a field expressed with the function space on a given triangle mesh.
    * @param[in] mesh - Triangle mesh on which the field is defined.
    * @param[in] coeffs - Vector of function space coefficients for each degree of freedom.
    * @param[in] points - Points on which to sample the field.
    * @return Field sampled at `points`.
    * @details
    * A point contained in more than one face (e.g. a mesh vertex) is averaged across all
    * faces containing it, rather than summed, so that reconstructing at face centroids
    * (where each point belongs to exactly one face) and at vertices are both meaningful.
    */
    static EigMatNX<Complex, 1> reconstruct_field(
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
