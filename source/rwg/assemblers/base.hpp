// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for assembling RWG-based BEM matrices.
*/

#ifndef BEM_RWG_ASSEMBLER_BASE_H
#define BEM_RWG_ASSEMBLER_BASE_H

#include "types.hpp"
#include "geometry/point_cloud.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/primitives/triangle.hpp"
#include "matrix/base.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/assemblers/index_generator.hpp"
#include "rwg/excitations/base.hpp"
#include "rwg/projectors/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup assm
* @{
*/

/**
* @brief Base class for generating RWG-based BEM operator matrices.
* @tparam obs_num_dof - Number of degrees of freedom associated with each observation triangle.
* @tparam src_num_dof - Number of degrees of freedom associated with each source triangle.
*/
template <uint8_t obs_num_dof, uint8_t src_num_dof>
class OperatorAssemblerBase
{

    static_assert((obs_num_dof > 0), "OperatorAssemblerBase: `obs_num_dof` must be greater than 0.");
    static_assert((src_num_dof > 0), "OperatorAssemblerBase: `src_num_dof` must be greater than 0.");

public:

    /**
    * @brief Constructs an `OperatorAssemblerBase` for given observation and source meshes.
    * @param[in] obs_mesh - Observation triangle mesh for which the operator matrix is to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operator matrix is to be assembled.
    * @param[in] elem_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operator matrix is to be assembled.
    */
    OperatorAssemblerBase(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh,
        ConstEigRef<EigMatNX<Index, 2>> elem_pairs
        ):
            obs_mesh_(obs_mesh),
            src_mesh_(src_mesh),
            elem_pairs_(elem_pairs) {};


    /**
    * @brief Constructs an `OperatorAssemblerBase` for given observation and source meshes.
    * @param[in] obs_mesh - Observation triangle mesh for which the operator matrix is to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operator matrix is to be assembled.
    */
    OperatorAssemblerBase(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh
        ):
            obs_mesh_(obs_mesh),
            src_mesh_(src_mesh),
            elem_pairs_(IndexGenerator::elem_pairs(obs_mesh, src_mesh)) {};


    /**
    * @brief Constructs an `OperatorAssemblerBase` for a given mesh.
    * @param[in] mesh - Triangle mesh for which the operator matrix is to be assembled.
    * @param[in] elem_pairs - Observation (first row) and source (second row) triangle index pairs
    * for which the operator matrix is to be assembled.
    */
    OperatorAssemblerBase(
        const TriangleMesh<3>& mesh,
        ConstEigRef<EigMatNX<Index, 2>> elem_pairs
        ):
        obs_mesh_(mesh),
        src_mesh_(mesh),
        elem_pairs_(elem_pairs) {};


    /**
     * @brief Constructs an `OperatorAssemblerBase` for a given mesh.
     * @param[in] mesh - Triangle mesh for which the operator matrix is to be assembled.
     */
    OperatorAssemblerBase(const TriangleMesh<3>& mesh):
        obs_mesh_(mesh),
        src_mesh_(mesh),
        elem_pairs_(IndexGenerator::elem_pairs(mesh, mesh)) {};


    /**
    * @brief Assembles the operator matrix for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`; must
    * inherit from `OperatorBase<obs_num_dof, src_num_dof>`.
    * @param[in] k - Complex wavenumber.
    */
    template <typename OperatorType>
    void assemble(
        MatrixBase<Complex>& mat,
        OperatorType op,
        const Complex k
        )
    {

        static_assert(
            std::is_base_of<OperatorBase<obs_num_dof, src_num_dof>, OperatorType>::value,
            "OperatorAssemblerBase::assemble(): `OperatorType` must derive from `OperatorBase`"
            );

        prep_matrix(mat);

#pragma omp parallel for firstprivate(op)
        for (Index ii = 0; ii < elem_pairs_.cols(); ++ii)
        {
            Triangle<3> obs_tri = obs_mesh_.elem_primitive(elem_pairs_(0, ii));
            Triangle<3> src_tri = src_mesh_.elem_primitive(elem_pairs_(1, ii));

            EigMatMN<Complex, obs_num_dof, src_num_dof> values = op.compute(k, obs_tri, src_tri);

#pragma omp critical
            fill_matrix(mat, elem_pairs_.col(ii), values);
        }

        mat.assemble();
        return;
    };


    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    */
    virtual void prep_matrix(MatrixBase<Complex>& mat) = 0;


    /**
    * @brief Fills operator values in the matrix based on source and observation meshes and degrees of freedom.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    virtual void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, obs_num_dof, src_num_dof>> values
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~OperatorAssemblerBase() = default;


protected:

    const TriangleMesh<3>& obs_mesh_;
    const TriangleMesh<3>& src_mesh_;
    const EigMatNX<Index, 2> elem_pairs_;

};


/**
* @brief Base class for generating excitation matrices for RWG-based BEM systems.
* @tparam obs_num_dof - Number of degrees of freedom associated with each observation triangle.
*/
template <uint8_t obs_num_dof>
class ExcitationAssemblerBase
{

    static_assert((obs_num_dof > 0), "ExcitationAssemblerBase: `obs_num_dof` must be greater than 0.");

public:

    /**
    * @brief Constructs an `ExcitationAssemblerBase` for a given mesh.
    * @param[in] mesh - Triangle mesh for which the excitation matrix is to be assembled.
    */
    ExcitationAssemblerBase(const TriangleMesh<3>& mesh):
        obs_mesh_(mesh),
        elems_(EigRowVec<Index>::LinSpaced(obs_mesh_.num_elems(), 0, obs_mesh_.num_elems() - 1)) {};


    /**
    * @brief Constructs an `ExcitationAssemblerBase` for a given mesh on given test elements.
    * @param[in] mesh - Triangle mesh for which the excitation matrix is to be assembled.
    * @param[in] elems - Triangle index pairs on which to test the incident field.
    */
    ExcitationAssemblerBase(const TriangleMesh<3>& mesh, ConstEigRef<EigRowVec<Index>> elems):
        obs_mesh_(mesh),
        elems_(elems) {};


    /**
    * @brief Assembles the excitation matrix for a given excitation object and observation triangle mesh.
    * @param[out] mat - Matrix to store the assembled excitation coefficients, with columns corresponding
    * to each right-hand side, and rows corresponding to observation mesh degrees of freedom.
    * @param[in] exc - Excitation object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    virtual void assemble(
        MatrixBase<Complex>& mat,
        ExcitationBase<obs_num_dof>& exc,
        const Complex k
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~ExcitationAssemblerBase() = default;


protected:

    const TriangleMesh<3>& obs_mesh_;
    const EigRowVec<Index> elems_;

};


/**
* @brief Base class for generating RWG-based BEM projector matrices.
* @tparam obs_dim - Dimension of the projected fields.
* @tparam src_num_dof - Number of degrees of freedom associated with each source triangle.
*/
template <uint8_t obs_dim, uint8_t src_num_dof>
class ProjectorAssemblerBase
{

    static_assert((obs_dim > 0), "`obs_dim` must be greater than 0.");
    static_assert((src_num_dof > 0), "`src_num_dof` must be greater than 0.");

public:

    /**
    * @brief Constructs a `ProjectorAssemblerBase` for given observation points and source mesh.
    * @param[in] obs_cloud - Observation point cloud on which to project fields.
    * @param[in] src_mesh - Source triangle mesh for which the projector matrix is to be assembled.
    * @param[in] elems - Source triangle indices for which the projector matrix is to be assembled.
    */
    ProjectorAssemblerBase(
        const PointCloud<3>& obs_cloud,
        const TriangleMesh<3>& src_mesh,
        ConstEigRef<EigRowVec<Index>> elems
        ):
            obs_cloud_(obs_cloud),
            src_mesh_(src_mesh),
            elems_(elems) {};


    /**
    * @brief Constructs a `ProjectorAssemblerBase` for given observation points and source mesh.
    * @param[in] obs_cloud - Observation point cloud on which to project fields.
    * @param[in] src_mesh - Source triangle mesh for which the projector matrix is to be assembled.
    */
    ProjectorAssemblerBase(
        const PointCloud<3>& obs_cloud,
        const TriangleMesh<3>& src_mesh
        ):
            obs_cloud_(obs_cloud),
            src_mesh_(src_mesh),
            elems_(EigRowVec<Index>::LinSpaced(src_mesh.num_elems(), 0, src_mesh.num_elems() - 1)) {};


    /**
    * @brief Assembles the projector matrix for a given projector object, source mesh, and observation points.
    * @param[out] mat - Matrix to store the assembled projector coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation points.
    * @param[in] op - Projector object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    virtual void assemble(
        MatrixBase<Complex>& mat,
        ProjectorBase<src_num_dof>& op,
        const Complex k
        ) = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~ProjectorAssemblerBase() = default;


protected:

    const PointCloud<3>& obs_cloud_;
    const TriangleMesh<3>& src_mesh_;
    const EigRowVec<Index> elems_;

};

/**
* @}
*/

}

#endif
