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

#include "rwg/assemblers/operator_matrix.hpp"

#include "types.hpp"
#include "matrix/base.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem::rwg
{

void EdgeOperatorAssembler::prep_matrix(MatrixBase<Complex>& mat)
{
    mat.resize(base::obs_mesh_.num_edges(), base::src_mesh_.num_edges());
    mat.preallocate(base::elem_pairs_.cols() * EDGE_ELEM_RATIO * EDGE_ELEM_RATIO);
    return;
};


void EdgeOperatorAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMatMN<Complex, 3, 3>> values
    )
{
    for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
    {
        Index col = base::src_mesh_.elem_edges()(src_edge, elem_pair[1]);
        for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
        {
            Index row = base::obs_mesh_.elem_edges()(obs_edge, elem_pair[0]);
            mat.add_value(row, col, values(obs_edge, src_edge));
        }
    }
    return;
};


void FaceOperatorAssembler::prep_matrix(MatrixBase<Complex>& mat)
{
    mat.resize(base::obs_mesh_.num_elems(), base::src_mesh_.num_elems());
    mat.preallocate(base::elem_pairs_.cols());
    return;
};


void FaceOperatorAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMatMN<Complex, 1, 1>> values
    )
{
    mat.set_value(elem_pair[0], elem_pair[1], values[0]);
    return;
};


void FaceEdgeOperatorAssembler::prep_matrix(MatrixBase<Complex>& mat)
{
    mat.resize(base::obs_mesh_.num_elems(), base::src_mesh_.num_edges());
    mat.preallocate(base::elem_pairs_.cols() * EDGE_ELEM_RATIO);
    return;
};


void FaceEdgeOperatorAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMatMN<Complex, 1, 3>> values
    )
{
    Index row = elem_pair[0];
    for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
    {
        Index col = base::src_mesh_.elem_edges()(src_edge, elem_pair[1]);
        mat.add_value(row, col, values(src_edge));
    }
    return;
};


void EdgeFaceOperatorAssembler::prep_matrix(MatrixBase<Complex>& mat)
{
    mat.resize(base::obs_mesh_.num_edges(), base::src_mesh_.num_elems());
    mat.preallocate(base::elem_pairs_.cols() * EDGE_ELEM_RATIO);
    return;
};


void EdgeFaceOperatorAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMatMN<Complex, 3, 1>> values
    )
{
    Index col = elem_pair[1];
    for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
    {
        Index row = base::obs_mesh_.elem_edges()(obs_edge, elem_pair[0]);
        mat.add_value(row, col, values(obs_edge));
    }
    return;
};


void VectorOperatorsAssembler::prep_matrix(MatrixBase<Complex>& mat)
{
    mat.resize(base::obs_mesh_.num_edges(), base::src_mesh_.num_edges() * 4);
    mat.preallocate(base::elem_pairs_.cols() * 4 * EDGE_ELEM_RATIO * EDGE_ELEM_RATIO);
    return;
};


void VectorOperatorsAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMatMN<Complex, 3, 12>> values
    )
{
    for (uint8_t op = 0; op < 4; ++op)
    {
        Index offset = base::src_mesh_.num_edges() * op;

        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = base::src_mesh_.elem_edges()(src_edge, elem_pair[1]) + offset;
            for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
            {
                Index row = base::obs_mesh_.elem_edges()(obs_edge, elem_pair[0]);
                mat.add_value(row, col, values(obs_edge, src_edge + 3 * op));
            }
        }
    }
    return;
};

}

