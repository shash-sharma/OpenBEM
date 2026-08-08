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

EigMatNX<Index, 2> IndexGenerator::elem_pairs(
    const TriangleMesh<3>& obs_mesh,
    const TriangleMesh<3>& src_mesh
    )
{
    Index num_pairs = obs_mesh.num_elems() * src_mesh.num_elems();
    EigMatNX<Index, 2> pairs = EigMatNX<Index, 2>::Zero(2, num_pairs);

    for (Index ii = 0; ii < num_pairs; ++ii)
    {
        pairs(0, ii) = ii / src_mesh.num_elems();
        pairs(1, ii) = ii % src_mesh.num_elems();
    }

    return pairs;
};


EigMatNX<Index, 2> IndexGenerator::elem_pairs(
    ConstEigRef<EigRowVec<Index>> obs_elems,
    ConstEigRef<EigRowVec<Index>> src_elems
    )
{
    Index num_pairs = obs_elems.size() * src_elems.size();
    EigMatNX<Index, 2> pairs = EigMatNX<Index, 2>::Zero(2, num_pairs);

    for (Index ii = 0; ii < obs_elems.size(); ++ii)
    {
        for (Index jj = 0; jj < src_elems.size(); ++jj)
        {
            pairs(0, jj + ii * src_elems.size()) = obs_elems[ii];
            pairs(1, jj + ii * src_elems.size()) = src_elems[jj];
        }
    }

    return pairs;
};


EigMatNX<Index, 2> IndexGenerator::elem_pairs(
    const TriangleMesh<3>& mesh,
    const IndexSet& index_set,
    const OperatorDof row_dof,
    const OperatorDof col_dof
    )
{

    if (row_dof == OperatorDof::EDGE && col_dof == OperatorDof::EDGE)
    {
        EigRowVec<Index> obs_elems = elems_from_edges(mesh, index_set.rows());
        EigRowVec<Index> src_elems = elems_from_edges(mesh, index_set.cols());
        return elem_pairs(obs_elems, src_elems);
    }

    else if (row_dof == OperatorDof::FACE && col_dof == OperatorDof::EDGE)
    {
        EigRowVec<Index> src_elems = elems_from_edges(mesh, index_set.cols());
        return elem_pairs(index_set.rows(), src_elems);
    }

    else if (row_dof == OperatorDof::EDGE && col_dof == OperatorDof::FACE)
    {
        EigRowVec<Index> obs_elems = elems_from_edges(mesh, index_set.rows());
        return elem_pairs(obs_elems, index_set.cols());
    }

    else if (row_dof == OperatorDof::FACE && col_dof == OperatorDof::FACE)
    {
        return elem_pairs(index_set.rows(), index_set.cols());
    }

    else
    {
        throw std::invalid_argument(
            "IndexGenerator::elem_pairs(): `index_set` has invalid row or column dofs."
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


EigRowVec<Index> IndexGenerator::elems_from_edges(
    const TriangleMesh<3>& mesh,
    ConstEigRef<EigRowVec<Index>> edges
    )
{
    std::vector<Index> unique_elems;
    unique_elems.reserve(edges.size() * 2);

    for (Index ii = 0; ii < edges.size(); ++ii)
        for (uint8_t iip = 0; iip < 2; ++iip)
            unique_elems.push_back(mesh.edge_elems()(iip, edges[ii]));

    std::sort(unique_elems.begin(), unique_elems.end());
    unique_elems.erase(std::unique(unique_elems.begin(), unique_elems.end()), unique_elems.end());

    return Eigen::Map<const EigRowVec<Index>> (unique_elems.data(), unique_elems.size());
}

}
