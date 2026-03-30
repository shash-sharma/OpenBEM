// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for assembling discrete operators for the NEFIE.
*/

#ifndef BEM_RWG_INTEQ_NEFIE_H
#define BEM_RWG_INTEQ_NEFIE_H

#include "types.hpp"
#include "constants.hpp"
#include "materials.hpp"

#include "geometry/point_cloud.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_dense.hpp"

#include "rwg/integral_equations/base.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"

#include "rwg/excitations/base.hpp"

#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "rwg/assemblers/operator_matrix.hpp"
#include "rwg/assemblers/excitation_matrix.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgie
* @{
*/

/**
* @brief Class defining the RWG-based NEFIE.
* @tparam MatrixType - Matrix type used for all operators, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenDenseMatrix<Complex>>
class Nefie: public IntegralEquationBase<MatrixType>
{

    using base = IntegralEquationBase<MatrixType>;
    using base::base;

public:

    /**
    * @brief Sets custom operators for the `Nefie` object.
    * @param[in] op_Tr - Object for the rotationally-tested vector hypersingular operator.
    * @param[in] op_Kr - Object for the rotationally-tested vector double-layer potential operator.
    */
    void set_operators(
        const RotVectorHypersingularOp<>& op_Tr,
        const RotVectorDoubleLayerPvOp<>& op_Kr
        )
    {
        op_Tr_ = op_Tr;
        op_Kr_ = op_Kr;
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

        MatrixType Tr;
        assembler_.assemble(Tr, op_Tr_, k);

        Tr.scale(J * two_pi * f * mu);

        if (base::flip_normals_)
            Tr.scale(-one);

        return Tr;
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

        MatrixType Kr;
        assembler_.assemble(Kr, op_Kr_, k);

        if (base::flip_normals_)
            Kr.scale(-one);

        Kr.add_ax(id_matrix(), -half);

        return Kr;
    };


    /**
    * @brief Returns the identity operator matrix associated with the magnetic surface current density.
    * @return Operator matrix.
    */
    MatrixType id_matrix()
    {
        RwgRwgOp<> op_I;
        MatrixType I;
        assembler_.assemble(I, op_I, 0);
        return I;
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
        ExcitationBase<3>& exc
        )
    {
        std::vector<Index> obs_elems_vec (base::elem_pairs_.cols());
        EigRowVec<Index>::Map(&obs_elems_vec[0], base::elem_pairs_.row(0).size()) = base::elem_pairs_.row(0);

        std::sort(obs_elems_vec.begin(), obs_elems_vec.end());
        obs_elems_vec.erase(std::unique(obs_elems_vec.begin(), obs_elems_vec.end()), obs_elems_vec.end());

        EigRowVec<Index> obs_elems = EigRowVec<Index>::Map(&obs_elems_vec[0], obs_elems_vec.size());
        EdgeExcitationAssembler exc_assembler (base::obs_mesh_, obs_elems);

        Complex k = material.k(f);
        MatrixType inc (base::obs_mesh_.num_edges(), exc.num_excitations());
        exc_assembler.assemble(inc, exc, k);

        if (!base::flip_normals_)
            inc.scale(-one);

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

        EdgeProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

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

        EdgeProjectorAssembler<3> proj_assembler (points, base::src_mesh_);

        MatrixType K;
        proj_assembler.assemble(K, proj_K_, k);

        K.scale(-one);

        return K;
    };


protected:

    RotVectorHypersingularOp<> op_Tr_;
    RotVectorDoubleLayerPvOp<> op_Kr_;

    VectorHypersingularProj<> proj_T_;
    VectorDoubleLayerProj<> proj_K_;

    EdgeOperatorAssembler assembler_ = EdgeOperatorAssembler(base::obs_mesh_, base::src_mesh_, base::elem_pairs_);

};

/**
* @}
*/

}

#endif
