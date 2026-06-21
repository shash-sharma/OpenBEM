// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Lumped element functionality for the AEFIE.
*/

#ifndef BEM_RWG_AEFIE_LUMPED_ELEM_H
#define BEM_RWG_AEFIE_LUMPED_ELEM_H

#include <vector>

#include "types.hpp"
#include "constants.hpp"

#include "geometry/mesh/base.hpp"
#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/incidence.hpp"
#include "rwg/assemblers/operator_assembler.hpp"
#include "rwg/lumped_elements/base.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgexc
* @{
*/

/**
* @brief Class providing lumped elements functionality for the AEFIE.
* @tparam MatrixType - Matrix type, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
class AefieLumpedElement: public LumpedElement<MatrixType>
{

    using base = LumpedElement<MatrixType>;
    using base::base;

public:

    /**
    * @brief Returns the matrix the couples the element port current to the AEFIE.
    * @param[in] f - Frequency in Hz.
    * @return Coupling matrix.
    */
    MatrixType coupling_matrix(const Float f) const
    {
        MatrixType mat;
        mat.set_matmul(base::terminal_mapping_matrix(), base::current_mapping_matrix());
        return mat;
    };


    /**
    * @brief Returns the matrix associated with the potential differences between terminals.
    * @param[in] f - Frequency in Hz.
    * @return Voltage matrix.
    */
    MatrixType voltage_matrix(const Float f) const
    {
        TriangleMesh<3> port_mesh = base::port_mesh_view().mesh();
        MatrixType voltage_map = base::voltage_mapping_matrix();
        MatrixType Lp = phi_matrix(f, port_mesh, base::structure_.mesh());

        MatrixType mat;
        mat.set_matmul(voltage_map, Lp, -one);

        return mat;
    };


    /**
    * @brief Returns the matrix associated with terminal currents.
    * @param[in] f - Frequency in Hz.
    * @return Current matrix.
    */
    MatrixType current_matrix(const Float f) const
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
        OperatorAssembler assm (obs_mesh, src_mesh);
        MatrixType Lp;
        assm.assemble(Lp, op_Lp, k);
        Lp.scale(one / eps_eff);

        return Lp;
    };

};

/**
* @}
*/

}

#endif
