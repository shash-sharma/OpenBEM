// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Index sets and generators for assemblers.
*/

#include "rwg/assemblers/indexing.hpp"

#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"


namespace bem
{

EigMatNX<Index, 2> IndexGenerator::face_pairs(
    const TriangleMesh<3>& obs_mesh,
    const TriangleMesh<3>& src_mesh
    )
{
    Index num_pairs = obs_mesh.num_faces() * src_mesh.num_faces();
    EigMatNX<Index, 2> pairs = EigMatNX<Index, 2>::Zero(2, num_pairs);

    for (Index ii = 0; ii < num_pairs; ++ii)
    {
        pairs(0, ii) = ii / src_mesh.num_faces();
        pairs(1, ii) = ii % src_mesh.num_faces();
    }

    return pairs;
};


EigMatNX<Index, 2> IndexGenerator::face_pairs(
    ConstEigRef<EigRowVec<Index>> obs_faces,
    ConstEigRef<EigRowVec<Index>> src_faces
    )
{
    Index num_pairs = obs_faces.size() * src_faces.size();
    EigMatNX<Index, 2> pairs = EigMatNX<Index, 2>::Zero(2, num_pairs);

    for (Index ii = 0; ii < obs_faces.size(); ++ii)
    {
        for (Index jj = 0; jj < src_faces.size(); ++jj)
        {
            pairs(0, jj + ii * src_faces.size()) = obs_faces[ii];
            pairs(1, jj + ii * src_faces.size()) = src_faces[jj];
        }
    }

    return pairs;
};


EigMatNX<Index, 2> IndexGenerator::face_pairs(
    const TriangleMesh<3>& mesh,
    const IndexSet& index_set,
    const DofSpace row_dof,
    const DofSpace col_dof
    )
{

    if (row_dof == DofSpace::EDGE && col_dof == DofSpace::EDGE)
    {
        EigRowVec<Index> obs_faces = faces_from_edges(mesh, index_set.rows());
        EigRowVec<Index> src_faces = faces_from_edges(mesh, index_set.cols());
        return face_pairs(obs_faces, src_faces);
    }

    else if (row_dof == DofSpace::FACE && col_dof == DofSpace::EDGE)
    {
        EigRowVec<Index> src_faces = faces_from_edges(mesh, index_set.cols());
        return face_pairs(index_set.rows(), src_faces);
    }

    else if (row_dof == DofSpace::EDGE && col_dof == DofSpace::FACE)
    {
        EigRowVec<Index> obs_faces = faces_from_edges(mesh, index_set.rows());
        return face_pairs(obs_faces, index_set.cols());
    }

    else if (row_dof == DofSpace::FACE && col_dof == DofSpace::FACE)
    {
        return face_pairs(index_set.rows(), index_set.cols());
    }

    else
    {
        throw std::invalid_argument(
            "IndexGenerator::face_pairs(): `index_set` has invalid row or column dofs."
            );
    }

}


EigMatNX<Index, 2> IndexGenerator::unique_pairs(
    ConstEigRef<EigMatNX<Index, 2>> pairs
    )
{
    std::set<std::pair<Index, Index>> set_pairs;
    for (Index ii = 0; ii < pairs.cols(); ++ii)
        set_pairs.insert(std::make_pair(pairs(0, ii), pairs(1, ii)));

    EigMatNX<Index, 2> unique_pairs (2, set_pairs.size());
    Index ii = 0;

    for (auto it = set_pairs.begin(); it != set_pairs.end(); ++it)
    {
        unique_pairs(0, ii) = it->first;
        unique_pairs(1, ii) = it->second;
        ii++;
    }

    return unique_pairs;
};


EigRowVec<Index> IndexGenerator::faces_from_edges(
    const TriangleMesh<3>& mesh,
    ConstEigRef<EigRowVec<Index>> edges
    )
{
    std::vector<Index> unique_faces;
    unique_faces.reserve(edges.size() * 2);

    for (Index ii = 0; ii < edges.size(); ++ii)
        for (uint8_t iip = 0; iip < 2; ++iip)
            unique_faces.push_back(mesh.edge_faces()(iip, edges[ii]));

    std::sort(unique_faces.begin(), unique_faces.end());
    unique_faces.erase(std::unique(unique_faces.begin(), unique_faces.end()), unique_faces.end());

    return Eigen::Map<const EigRowVec<Index>> (unique_faces.data(), unique_faces.size());
}

}
