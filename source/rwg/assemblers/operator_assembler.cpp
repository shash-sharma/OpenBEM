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

#pragma omp parallel for
    for (Index ii = 0; ii < face_pairs_.cols(); ++ii)
    {
        Triangle<3> obs_tri = obs_mesh_.face_primitive(face_pairs_(0, ii));
        Triangle<3> src_tri = src_mesh_.face_primitive(face_pairs_(1, ii));

        EigMat<Complex> values = op.compute(
            k, obs_tri, src_tri
            );

#pragma omp critical
        fill_matrix(mat, op, face_pairs_.col(ii), values);
    }

    mat.assemble();
    return;

};


void OperatorAssembler::assemble(
    std::vector<std::shared_ptr<MatrixBase<Complex>>>& mats,
    const std::vector<std::shared_ptr<OperatorBase>>& ops,
    const Complex k,
    ObsIntegratorBase& obs_integrator
    )
{

    if (mats.size() != ops.size())
        throw std::invalid_argument(
            std::string("OperatorAssembler::assemble(): `mats` must have been ") +
            std::string("initialized and have the same size as `ops`.")
            );

    for (auto& mat : mats)
        if (!mat)
            throw std::runtime_error(
                std::string("OperatorAssembler::assemble(): Each matrix pointer in ") +
                std::string("`mats` must have been initialized before assembly.")
                );

    for (Index ii = 0; ii < ops.size(); ++ii)
        prep_matrix(*mats[ii], *ops[ii]);

#pragma omp parallel for
    for (Index ii = 0; ii < face_pairs_.cols(); ++ii)
    {
        Triangle<3> obs_tri = obs_mesh_.face_primitive(face_pairs_(0, ii));
        Triangle<3> src_tri = src_mesh_.face_primitive(face_pairs_(1, ii));

        Triangle<3> obs_tri_local;
        Triangle<2> src_tri_local;
        OperatorBase::transform_coordinates(obs_tri_local, src_tri_local, obs_tri, src_tri);

        const ObsResult obs_result = obs_integrator.integrate(
            k, obs_tri_local, src_tri_local, true, true, true, true
            );

        for (Index jj = 0; jj < ops.size(); ++jj)
        {
            EigMat<Complex> values = ops[jj]->assemble(
                k, obs_tri_local, src_tri_local.to_3d(), obs_result
                );

#pragma omp critical
            fill_matrix(*mats[jj], *ops[jj], face_pairs_.col(ii), values);
        }
    }

    for (auto& mat: mats)
        mat->assemble();

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
        mat.preallocate(face_pairs_.cols() * EDGE_ELEM_RATIO * EDGE_ELEM_RATIO);
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::EDGE)
    {
        mat.resize(obs_mesh_.num_faces(), src_mesh_.num_edges());
        mat.preallocate(face_pairs_.cols() * EDGE_ELEM_RATIO);
    }

    else if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::FACE)
    {
        mat.resize(obs_mesh_.num_edges(), src_mesh_.num_faces());
        mat.preallocate(face_pairs_.cols() * EDGE_ELEM_RATIO);
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::FACE)
    {
        mat.resize(obs_mesh_.num_faces(), src_mesh_.num_faces());
        mat.preallocate(face_pairs_.cols());
    }

    return;

}


void OperatorAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    ConstEigRef<EigColVecN<Index, 2>> face_pair,
    ConstEigRef<EigMat<Complex>> values
    )
{

    if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::EDGE)
    {
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = src_mesh_.face_edges()(src_edge, face_pair[1]);
            for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
            {
                Index row = obs_mesh_.face_edges()(obs_edge, face_pair[0]);
                mat.add_value(row, col, values(obs_edge, src_edge));
            }
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::EDGE)
    {
        Index row = face_pair[0];
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = src_mesh_.face_edges()(src_edge, face_pair[1]);
            mat.add_value(row, col, values(0, src_edge));
        }
    }

    else if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::FACE)
    {
        Index col = face_pair[1];
        for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
        {
            Index row = obs_mesh_.face_edges()(obs_edge, face_pair[0]);
            mat.add_value(row, col, values(obs_edge, 0));
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::FACE)
    {
        mat.set_value(face_pair[0], face_pair[1], values(0, 0));
    }

    return;

};

}

