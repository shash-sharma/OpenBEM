// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for assembling discrete operators for integral equations.
*/

#ifndef BEM_RWG_INTEQ_BASE_H
#define BEM_RWG_INTEQ_BASE_H

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_dense.hpp"

#include "rwg/assemblers/operator_matrix.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgie
* @{
*/

/**
* @brief Base class defining an RWG-based integral equation.
* @tparam MatrixType - Matrix type used for all operators, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenDenseMatrix<Complex>>
class IntegralEquationBase
{

static_assert(
    std::is_base_of<MatrixBase<Complex>, MatrixType>::value,
    "IntegralEquationBase: MatrixType must be derived from MatrixBase<Complex>."
    );

public:

    /**
    * @brief Constructs an `IntegralEquationBase` object.
    * @param[in] obs_mesh - Observation triangle mesh for which the operators are to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operators are to be assembled.
    * @param[in] elem_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operators are to be assembled.
    * @param[in] flip_normals - Whether to flip the surface normals so they point inwards (optional).
    */
    IntegralEquationBase(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh,
        ConstEigRef<EigMatNX<Index, 2>> elem_pairs,
        const bool flip_normals = false
        ):
            obs_mesh_(obs_mesh),
            src_mesh_(src_mesh),
            elem_pairs_(elem_pairs),
            flip_normals_(flip_normals) {};


    /**
    * @brief Constructs an `IntegralEquationBase` object.
    * @param[in] obs_mesh - Observation triangle mesh for which the operators are to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operators are to be assembled.
    * @param[in] flip_normals - Whether to flip the surface normals so they point inwards (optional).
    */
    IntegralEquationBase(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh,
        const bool flip_normals = false
        ):
            obs_mesh_(obs_mesh),
            src_mesh_(src_mesh),
            elem_pairs_(OperatorAssemblerBase<3, 3>::make_pairs(obs_mesh, src_mesh)),
            flip_normals_(flip_normals) {};


    /**
    * @brief Constructs an `IntegralEquationBase` object.
    * @param[in] mesh - Source and observation triangle mesh for which the operators are to be assembled.
    * @param[in] elem_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operators are to be assembled.
    * @param[in] flip_normals - Whether to flip the surface normals so they point inwards (optional).
    */
    IntegralEquationBase(
        const TriangleMesh<3>& mesh,
        ConstEigRef<EigMatNX<Index, 2>> elem_pairs,
        const bool flip_normals = false
        ):
            obs_mesh_(mesh),
            src_mesh_(mesh),
            elem_pairs_(elem_pairs),
            flip_normals_(flip_normals) {};


    /**
    * @brief Constructs an `IntegralEquationBase` object.
    * @param[in] mesh - Source and observation triangle mesh for which the operators are to be assembled.
    * @param[in] flip_normals - Whether to flip the surface normals so they point inwards (optional).
    */
    IntegralEquationBase(
        const TriangleMesh<3>& mesh,
        const bool flip_normals = false
        ):
            obs_mesh_(mesh),
            src_mesh_(mesh),
            elem_pairs_(OperatorAssemblerBase<3, 3>::make_pairs(mesh, mesh)),
            flip_normals_(flip_normals) {};


    /**
    * @brief Virtual destructor.
    */
    virtual ~IntegralEquationBase() = default;


protected:

    const TriangleMesh<3>& obs_mesh_;
    const TriangleMesh<3>& src_mesh_;
    const EigMatNX<Index, 2> elem_pairs_;
    const bool flip_normals_ = false;

};

/**
* @}
*/

}

#endif
