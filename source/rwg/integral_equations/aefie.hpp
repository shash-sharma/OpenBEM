// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for assembling discrete operators for the AEFIE.
*/

#ifndef BEM_RWG_INTEQ_AEFIE_H
#define BEM_RWG_INTEQ_AEFIE_H

#include "types.hpp"
#include "constants.hpp"
#include "materials.hpp"

#include "geometry/point_cloud.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"

#include "rwg/integral_equations/base.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"
#include "rwg/operators/incidence.hpp"

#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "rwg/excitations/base.hpp"

#include "rwg/assemblers/operator_matrix.hpp"
#include "rwg/assemblers/excitation_matrix.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgie
* @{
*/

/**
* @brief Class defining the RWG-based AEFIE.
* @tparam MatrixType - Matrix type used for all operators, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
class Aefie: public IntegralEquationBase<MatrixType>
{

    using base = IntegralEquationBase<MatrixType>;
    using base::base;

public:

    /**
    * @brief Sets custom operators for the `Aefie` object.
    * @param[in] op_La - Object for the vector single-layer potential operator.
    * @param[in] op_Lp - Object for the scalar single-layer potential operator.
    * @param[in] op_K - Object for the vector double-layer potential operator.
    */
    void set_operators(
        const VectorSingleLayerOp<>& op_La,
        const ScalarSingleLayerOp<>& op_Lp,
        const VectorDoubleLayerPvOp<>& op_K
        )
    {
        op_La_ = op_La;
        op_Lp_ = op_Lp;
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
        MatrixType La;
        assembler_.assemble(La, op_La_, material.k(f));
        La.scale(-J * two_pi * f * material.mu());
        return La;
    };


    /**
    * @brief Returns the operator matrix associated with the electric surface charge density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType rho_matrix(
        const Float f,
        const Material& material
        )
    {
        MatrixType Dt;
        Dt.set_transpose(div_matrix(f, material));

        MatrixType Lp = phi_matrix(f, material);
        MatrixType DtLp;
        DtLp.set_matmul(Dt, Lp);
        return DtLp;
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
        RotVectorIdentityOp<> op_Ir;
        MatrixType Ir;
        assembler_.assemble(Ir, op_Ir, 0);
        if (base::flip_normals_)
            Ir.scale(-one);
        return Ir;
    };


    /**
    * @brief Returns the intermediate operator matrix associated with the electric surface charge density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType phi_matrix(
        const Float f,
        const Material& material
        )
    {
        MatrixType Lp;
        assembler_.assemble(Lp, op_Lp_, material.k(f));
        Lp.scale(one / material.eps_eff(f));
        return Lp;
    };


    /**
    * @brief Returns the divergence matrix.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @return Operator matrix.
    */
    MatrixType div_matrix(
        const Float f,
        const Material& material
        )
    {
        DivergenceOp op_D;
        MatrixType D;
        assembler_.assemble(D, op_D, 0);
        return D;
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
        std::vector<Index> obs_elems_vec (base::elem_pairs_.cols());
        EigRowVec<Index>::Map(&obs_elems_vec[0], base::elem_pairs_.row(0).size()) = base::elem_pairs_.row(0);

        std::sort(obs_elems_vec.begin(), obs_elems_vec.end());
        obs_elems_vec.erase(std::unique(obs_elems_vec.begin(), obs_elems_vec.end()), obs_elems_vec.end());

        EigRowVec<Index> obs_elems = EigRowVec<Index>::Map(&obs_elems_vec[0], obs_elems_vec.size());
        ExcitationAssembler exc_assembler (base::obs_mesh_, obs_elems);

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

        ProjectorAssembler<3> edge_proj_assembler (points, base::src_mesh_);

        MatrixType La;
        edge_proj_assembler.assemble(La, proj_La_, k);
        La.scale(-J * two_pi * f * mu);

        return La;
    };


    /**
    * @brief Returns the projector matrix associated with the electric surface charge density.
    * @param[in] f - Frequency in Hz.
    * @param[in] material - Material for which the equation is defined.
    * @param[in] points - Points at which to project fields.
    * @return Projector matrix.
    */
    MatrixType rho_projector(
        const Float f,
        const Material& material,
        const PointCloud<3>& points
        )
    {
        Complex k = material.k(f);
        Complex eps_eff = material.eps_eff(f);

        ProjectorAssembler<3> face_proj_assembler (points, base::src_mesh_);

        MatrixType Lp;
        face_proj_assembler.assemble(Lp, proj_Lp_, k);
        Lp.scale(-one / eps_eff);

        return Lp;
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

    VectorSingleLayerOp<> op_La_;
    ScalarSingleLayerOp<> op_Lp_;
    VectorDoubleLayerPvOp<> op_K_;

    VectorSingleLayerProj<> proj_La_;
    GradScalarSingleLayerProj<> proj_Lp_;
    VectorDoubleLayerProj<> proj_K_;

    OperatorAssembler assembler_ = OperatorAssembler (base::obs_mesh_, base::src_mesh_, base::elem_pairs_);

};

/**
* @}
*/

}

#endif
