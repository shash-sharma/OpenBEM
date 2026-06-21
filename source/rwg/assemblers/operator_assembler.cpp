// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling RWG-based BEM operator matrices.
*/

#include "rwg/assemblers/operator_assembler.hpp"

#include "types.hpp"
#include "matrix/base.hpp"
#include "rwg/function_space.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem::rwg
{

void OperatorAssembler::assemble(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    const Complex k
    )
{

    prep_matrix(mat, op);

#pragma omp parallel
    {
        std::unique_ptr<OperatorBase> opc = op.clone();

#pragma omp for
        for (Index ii = 0; ii < elem_pairs_.cols(); ++ii)
        {
            Triangle<3> obs_tri = obs_mesh_.elem_primitive(elem_pairs_(0, ii));
            Triangle<3> src_tri = src_mesh_.elem_primitive(elem_pairs_(1, ii));

            EigMat<Complex> values = opc->compute(
                k, obs_tri, src_tri
                );

#pragma omp critical
            fill_matrix(mat, op, elem_pairs_.col(ii), values);
        }
    }

    mat.assemble();
    return;

};


void OperatorAssembler::prep_matrix(
    MatrixBase<Complex>& mat,
    const OperatorBase& op
    )
{

    if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::EDGE)
    {
        mat.resize(obs_mesh_.num_edges(), src_mesh_.num_edges());
        mat.preallocate(elem_pairs_.cols() * EDGE_ELEM_RATIO * EDGE_ELEM_RATIO);
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::EDGE)
    {
        mat.resize(obs_mesh_.num_elems(), src_mesh_.num_edges());
        mat.preallocate(elem_pairs_.cols() * EDGE_ELEM_RATIO);
    }

    else if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::FACE)
    {
        mat.resize(obs_mesh_.num_edges(), src_mesh_.num_elems());
        mat.preallocate(elem_pairs_.cols() * EDGE_ELEM_RATIO);
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::FACE)
    {
        mat.resize(obs_mesh_.num_elems(), src_mesh_.num_elems());
        mat.preallocate(elem_pairs_.cols());
    }

    return;

}


void OperatorAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMat<Complex>> values
    )
{

    if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::EDGE)
    {
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = src_mesh_.elem_edges()(src_edge, elem_pair[1]);
            for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
            {
                Index row = obs_mesh_.elem_edges()(obs_edge, elem_pair[0]);
                mat.add_value(row, col, values(obs_edge, src_edge));
            }
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::EDGE)
    {
        Index row = elem_pair[0];
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = src_mesh_.elem_edges()(src_edge, elem_pair[1]);
            mat.add_value(row, col, values(0, src_edge));
        }
    }

    else if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::FACE)
    {
        Index col = elem_pair[1];
        for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
        {
            Index row = obs_mesh_.elem_edges()(obs_edge, elem_pair[0]);
            mat.add_value(row, col, values(obs_edge, 0));
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::FACE)
    {
        mat.set_value(elem_pair[0], elem_pair[1], values(0, 0));
    }

    return;

};

}

