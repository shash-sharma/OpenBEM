// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for assembling discrete operators for the TEFIE.
*/

#ifndef BEM_RWG_INTEQ_TEFIE_H
#define BEM_RWG_INTEQ_TEFIE_H

#include "types.hpp"
#include "constants.hpp"
#include "materials.hpp"

#include "geometry/point_cloud.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"

#include "rwg/function_space.hpp"

#include "rwg/integral_equations/base.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"

#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "rwg/excitations/base.hpp"

#include "rwg/assemblers/operator_assembler.hpp"
#include "rwg/assemblers/excitation_assembler.hpp"
#include "rwg/assemblers/projector_assembler.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgie
* @{
*/

/**
* @brief Class defining the RWG-based TEFIE.
* @tparam MatrixType - Matrix type used for all operators, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
class Tefie: public IntegralEquationBase<MatrixType>
{

    using base = IntegralEquationBase<MatrixType>;
    using base::base;

public:

    /**
    * @brief Sets custom operators for the `Tefie` object.
    * @param[in] op_T - Object for the vector hypersingular operator.
    * @param[in] op_K - Object for the vector double-layer potential operator.
    */
    void set_operators(
        const VectorHypersingularOp& op_T,
        const VectorDoubleLayerPvOp& op_K
        )
    {
        op_T_ = op_T;
        op_K_ = op_K;
        return;
    };


    /**
    * @brief Returns the operator matrix associated with the electric surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType j_matrix(
        const Float f,
        const Material& material
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        MatrixType T;
        assembler_.assemble(T, op_T_, k);

        T.scale(-J * two_pi * f * mu);

        return T;
    };


    /**
    * @brief Returns the operator matrix associated with the magnetic surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType m_matrix(
        const Float f,
        const Material& material
        )
    {
        Complex k = material.k(f);

        MatrixType K;
        assembler_.assemble(K, op_K_, k);

        K.scale(-one);
        K.add_ax(id_matrix(), -half);

        return K;
    };


    /**
    * @brief Returns the identity operator matrix associated with the magnetic surface current density.
    * @return Operator matrix.
    */
    MatrixType id_matrix()
    {
        RotVectorIdentityOp op_Ir;
        MatrixType Ir;
        assembler_.assemble(Ir, op_Ir, 0);
        if (base::flip_normals_)
            Ir.scale(-one);
        return Ir;
    };


    /**
    * @brief Returns the excitation matrix for a given excitation operator.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] exc - Excitation operator.
    * @return Excitation matrix.
    */
    MatrixType exc_matrix(
        const Float f,
        const Material& material,
        ExcitationBase& exc
        )
    {
        std::vector<Index> obs_faces_vec (base::face_pairs_.cols());
        EigRowVec<Index>::Map(&obs_faces_vec[0], base::face_pairs_.row(0).size()) = base::face_pairs_.row(0);

        std::sort(obs_faces_vec.begin(), obs_faces_vec.end());
        obs_faces_vec.erase(std::unique(obs_faces_vec.begin(), obs_faces_vec.end()), obs_faces_vec.end());

        EigRowVec<Index> obs_faces = EigRowVec<Index>::Map(&obs_faces_vec[0], obs_faces_vec.size());
        ExcitationAssembler exc_assembler (base::obs_mesh_, obs_faces);

        Complex k = material.k(f);
        MatrixType inc (base::obs_mesh_.num_edges(), exc.num_excitations());
        exc_assembler.assemble(inc, exc, k);

        return inc;
    };


    /**
    * @brief Returns the projector matrix associated with the electric surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] points - Points at which to project fields.
    * @return Projector matrix.
    */
    MatrixType j_projector(
        const Float f,
        const Material& material,
        const PointCloud<3>& points
        )
    {
        Complex k = material.k(f);
        Complex mu = material.mu();

        ProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

        MatrixType T;
        proj_assembler.assemble(T, proj_T_, k);

        T.scale(-J * two_pi * f * mu);

        return T;
    };


    /**
    * @brief Returns the projector matrix associated with the magnetic surface current density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] points - Points at which to project fields.
    * @return Projector matrix.
    */
    MatrixType m_projector(
        const Float f,
        const Material& material,
        const PointCloud<3>& points
        )
    {
        Complex k = material.k(f);

        ProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

        MatrixType K;
        proj_assembler.assemble(K, proj_K_, k);

        K.scale(-one);

        return K;
    };


protected:

    VectorHypersingularOp op_T_;
    VectorDoubleLayerPvOp op_K_;

    VectorHypersingularProj proj_T_;
    VectorDoubleLayerProj proj_K_;

    OperatorAssembler assembler_ = OperatorAssembler (base::obs_mesh_, base::src_mesh_, base::face_pairs_);

};

/**
* @}
*/

}

#endif
