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

#ifndef BEM_RWG_OP_ASSEMBLER_I
#define BEM_RWG_OP_ASSEMBLER_I

#include "rwg/assemblers/operator_matrix.hpp"

#include "types.hpp"
#include "matrix/base.hpp"
#include "rwg/function_space.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem::rwg
{

template <typename TestSpace, typename ExpansionSpace>
void OperatorAssembler<TestSpace, ExpansionSpace>::assemble(
    MatrixBase<Complex>& mat,
    const OperatorBase<TestSpace, ExpansionSpace>& op,
    const Complex k
    )
{

    prep_matrix(mat);

#pragma omp parallel
    {
        std::unique_ptr<OperatorBase<TestSpace, ExpansionSpace>> opc = op.clone();

#pragma omp for
        for (Index ii = 0; ii < elem_pairs_.cols(); ++ii)
        {
            Triangle<3> obs_tri = obs_mesh_.elem_primitive(elem_pairs_(0, ii));
            Triangle<3> src_tri = src_mesh_.elem_primitive(elem_pairs_(1, ii));

            EigMatMN<Complex, TestSpace::dof, ExpansionSpace::dof> values = opc->compute(
                k, obs_tri, src_tri
                );

#pragma omp critical
            fill_matrix(mat, elem_pairs_.col(ii), values);
        }
    }

    mat.assemble();
    return;

};


template <typename TestSpace, typename ExpansionSpace>
void OperatorAssembler<TestSpace, ExpansionSpace>::prep_matrix(MatrixBase<Complex>& mat)
{

    if constexpr (TestSpace::dof == 3 && ExpansionSpace::dof == 3)
    {
        mat.resize(obs_mesh_.num_edges(), src_mesh_.num_edges());
        mat.preallocate(elem_pairs_.cols() * EDGE_ELEM_RATIO * EDGE_ELEM_RATIO);
    }

    else if constexpr (TestSpace::dof == 1 && ExpansionSpace::dof == 3)
    {
        mat.resize(obs_mesh_.num_elems(), src_mesh_.num_edges());
        mat.preallocate(elem_pairs_.cols() * EDGE_ELEM_RATIO);
    }

    else if constexpr (TestSpace::dof == 3 && ExpansionSpace::dof == 1)
    {
        mat.resize(obs_mesh_.num_edges(), src_mesh_.num_elems());
        mat.preallocate(elem_pairs_.cols() * EDGE_ELEM_RATIO);
    }

    else if constexpr (TestSpace::dof == 1 && ExpansionSpace::dof == 1)
    {
        mat.resize(obs_mesh_.num_elems(), src_mesh_.num_elems());
        mat.preallocate(elem_pairs_.cols());
    }

    return;

}


template <typename TestSpace, typename ExpansionSpace>
void OperatorAssembler<TestSpace, ExpansionSpace>::fill_matrix(
    MatrixBase<Complex>& mat,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMatMN<Complex, TestSpace::dof, ExpansionSpace::dof>> values
    )
{

    if constexpr (TestSpace::dof == 3 && ExpansionSpace::dof == 3)
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

    else if constexpr (TestSpace::dof == 1 && ExpansionSpace::dof == 3)
    {
        Index row = elem_pair[0];
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = src_mesh_.elem_edges()(src_edge, elem_pair[1]);
            mat.add_value(row, col, values(src_edge));
        }
    }

    else if constexpr (TestSpace::dof == 3 && ExpansionSpace::dof == 1)
    {
        Index col = elem_pair[1];
        for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
        {
            Index row = obs_mesh_.elem_edges()(obs_edge, elem_pair[0]);
            mat.add_value(row, col, values(obs_edge));
        }
    }

    else if constexpr (TestSpace::dof == 1 && ExpansionSpace::dof == 1)
    {
        mat.set_value(elem_pair[0], elem_pair[1], values[0]);
    }

    return;

};




// template <>
// void VectorOperatorsAssembler::prep_matrix(MatrixBase<Complex>& mat)
// {
//     mat.resize(obs_mesh_.num_edges(), src_mesh_.num_edges() * 4);
//     mat.preallocate(elem_pairs_.cols() * 4 * EDGE_ELEM_RATIO * EDGE_ELEM_RATIO);
//     return;
// };


// template <>
// void VectorOperatorsAssembler::fill_matrix(
//     MatrixBase<Complex>& mat,
//     ConstEigRef<EigColVecN<Index, 2>> elem_pair,
//     ConstEigRef<EigMatMN<Complex, 3, 12>> values
//     )
// {
//     for (uint8_t op = 0; op < 4; ++op)
//     {
//         Index offset = src_mesh_.num_edges() * op;

//         for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
//         {
//             Index col = src_mesh_.elem_edges()(src_edge, elem_pair[1]) + offset;
//             for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
//             {
//                 Index row = obs_mesh_.elem_edges()(obs_edge, elem_pair[0]);
//                 mat.add_value(row, col, values(obs_edge, src_edge + 3 * op));
//             }
//         }
//     }
//     return;
// };

}

#endif
