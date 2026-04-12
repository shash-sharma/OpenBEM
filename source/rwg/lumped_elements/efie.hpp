// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Lumped element functionality for the TEFIE and NEFIE.
*/

#ifndef BEM_RWG_EFIE_LUMPED_ELEM_H
#define BEM_RWG_EFIE_LUMPED_ELEM_H

#include <vector>

#include "types.hpp"
#include "constants.hpp"

#include "geometry/mesh/base.hpp"
#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/incidence.hpp"
#include "rwg/assemblers/operator_matrix.hpp"
#include "rwg/lumped_elements/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgexc
* @{
*/

/**
* @brief Class providing lumped element functionality for the TEFIE.
* @tparam MatrixType - Matrix type, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
class TefieLumpedElement: public LumpedElementBase<MatrixType>
{

    using base = LumpedElementBase<MatrixType>;
    using base::base;

public:

    /**
    * @brief Returns the matrix the couples the element port current to the TEFIE.
    * @param[in] f - Frequency in Hz.
    * @return Coupling matrix.
    */
    MatrixType coupling_matrix(const Float f) const override
    {
        TriangleMesh<3> port_mesh = base::port_mesh_view().mesh();

        MatrixType D = divergence_matrix();
        MatrixType Dt;
        Dt.set_transpose(D);

        MatrixType Lp = phi_matrix(f, base::structure_.mesh(), port_mesh);
        MatrixType DtLp;
        DtLp.set_mat_mul(Dt, Lp);

        MatrixType current_map = base::current_mapping_matrix();
        MatrixType mat;
        mat.set_mat_mul(DtLp, current_map);

        return mat;
    };


    /**
    * @brief Returns the matrix associated with the potential differences between terminals.
    * @param[in] f - Frequency in Hz.
    * @return Voltage matrix.
    */
    MatrixType voltage_matrix(const Float f) const override
    {
        TriangleMesh<3> port_mesh = base::port_mesh_view().mesh();

        MatrixType voltage_map = base::voltage_mapping_matrix();
        MatrixType current_map = base::current_mapping_matrix();

        MatrixType D = divergence_matrix();
        MatrixType Lp_divj = phi_matrix(f, port_mesh, base::structure_.mesh());
        MatrixType Lp_jvol = phi_matrix(f, port_mesh, port_mesh);

        MatrixType Lp_divj_D;
        Lp_divj_D.set_mat_mul(Lp_divj, D);

        MatrixType Lp_current;
        Lp_current.set_mat_mul(Lp_jvol, current_map);

        MatrixType voltage_mat;
        voltage_mat.set_mat_mul(voltage_map, Lp_divj_D);

        MatrixType current_mat;
        current_mat.set_mat_mul(voltage_map, Lp_current);

        MatrixType mat (base::num_ports(), base::structure_.mesh().num_edges() + base::num_ports());
        mat.add_block(voltage_mat, 0, 0, -one);
        mat.add_block(current_mat, 0, base::structure_.mesh().num_edges(), -one);

        return mat;
    };


    /**
    * @brief Returns the matrix associated with terminal currents.
    * @param[in] f - Frequency in Hz
    * @return Current matrix.
    */
    MatrixType current_matrix(const Float f) const override
    {
        MatrixType mat (base::num_ports(), base::num_ports());
        for (Index ii = 0; ii < base::num_ports(); ++ii)
            mat.set_value(ii, ii, base::impedances()[ii]);
        mat.assemble();
        return mat;
    };


protected:

    /**
    * @brief Returns the matrix associated with the scalar potential on given source and observation meshes.
    * @param[in] f - Frequency in Hz.
    * @param[in] obs_mesh - Observation triangle mesh for which the operator matrix is to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operator matrix is to be assembled.
    * @return Scalar potential matrix.
    */
    MatrixType phi_matrix(
        const Float f,
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh
        ) const
    {
        Complex k = base::structure_.background_material().k(f);
        Complex eps_eff = base::structure_.background_material().eps_eff(f);

        ScalarSingleLayerOp op_Lp;
        FaceOperatorAssembler assm_face (obs_mesh, src_mesh);
        MatrixType Lp;
        assm_face.assemble(Lp, op_Lp, k);
        Lp.scale(-one / eps_eff / (J * two_pi * f));

        return Lp;
    };


    /**
    * @brief Returns the matrix that comptes the divergence matrix for `mesh` edges.
    * @return Divergence matrix.
    */
    virtual MatrixType divergence_matrix() const
    {
        DivRwgOp op_D;
        FaceEdgeOperatorAssembler assm_div (base::structure_.mesh(), base::structure_.mesh());
        MatrixType D;
        assm_div.assemble(D, op_D, 0);
        return D;
    };

};


/**
* @brief Class providing lumped element functionality for the NEFIE.
* @tparam MatrixType - Matrix type, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
class NefieLumpedElement: public LumpedElementBase<MatrixType>
{

    using base = LumpedElementBase<MatrixType>;
    using base::base;

public:

    /**
    * @brief Returns the matrix the couples the element port current to the NEFIE.
    * @param[in] f - Frequency in Hz.
    * @return Coupling matrix.
    */
    MatrixType coupling_matrix(const Float f) const override
    {
        TriangleMesh<3> port_mesh = base::port_mesh_view().mesh();
        MatrixType Lp = nxphi_matrix(f, base::structure_.mesh(), port_mesh);
        MatrixType current_map = base::current_mapping_matrix();
        MatrixType mat;
        mat.set_mat_mul(Lp, current_map);
        return mat;
    };


    /**
    * @brief Returns the matrix associated with the potential differences between terminals.
    * @param[in] f - Frequency in Hz.
    * @return Voltage matrix.
    */
    MatrixType voltage_matrix(const Float f) const override
    {
        TefieLumpedElement<MatrixType> tefie_elem (base::structure_, base::terminal_polygons_, base::ports_, base::impedances_);
        return tefie_elem.voltage_matrix(f);
    };


    /**
    * @brief Returns the matrix associated with terminal currents.
    * @param[in] f - Frequency in Hz
    * @return Current matrix.
    */
    MatrixType current_matrix(const Float f) const override
    {
        TefieLumpedElement<MatrixType> tefie_elem (base::structure_, base::terminal_polygons_, base::ports_, base::impedances_);
        return tefie_elem.current_matrix(f);
    };


protected:

    /**
    * @brief Returns the matrix associated with the scalar potential on given source and observation meshes.
    * @param[in] f - Frequency in Hz.
    * @param[in] obs_mesh - Observation triangle mesh for which the operator matrix is to be assembled.
    * @param[in] src_mesh - Source triangle mesh for which the operator matrix is to be assembled.
    * @return Scalar potential matrix.
    */
    MatrixType nxphi_matrix(
        const Float f,
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh
        ) const
    {
        Complex k = base::structure_.background_material().k(f);
        Complex eps_eff = base::structure_.background_material().eps_eff(f);

        RotGradScalarSingleLayerOp op_Lp;
        EdgeFaceOperatorAssembler assm (obs_mesh, src_mesh);
        MatrixType Lp;
        assm.assemble(Lp, op_Lp, k);
        Lp.scale(-one / eps_eff / (J * two_pi * f));

        return Lp;
    };

};

/**
* @}
*/

}

#endif
