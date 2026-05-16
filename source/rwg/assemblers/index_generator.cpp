// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Utility functions for assemblers.
*/

#include "rwg/assemblers/index_generator.hpp"

#include <vector>
#include <set>

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


EigMatNX<Index, 2> IndexGenerator::elem_pairs(const TriangleMesh<3>& mesh)
{
    return elem_pairs(mesh, mesh);
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


EigMatNX<Index, 2> IndexGenerator::elem_pairs(ConstEigRef<EigRowVec<Index>> elems)
{
    return elem_pairs(elems, elems);
};


EigMatNX<Index, 2> IndexGenerator::elem_pairs_self(ConstEigRef<EigRowVec<Index>> elems)
{
    Index num_pairs = elems.size();
    EigMatNX<Index, 2> pairs = EigMatNX<Index, 2>::Zero(2, num_pairs);

    for (Index ii = 0; ii < elems.size(); ++ii)
        for (uint8_t jj = 0; jj < 2; ++jj)
            pairs(jj, ii) = elems[ii];

    return pairs;
};


EigMatNX<Index, 2> IndexGenerator::elem_pairs_from_edges(
    const TriangleMesh<3>& mesh,
    ConstEigRef<EigRowVec<Index>> obs_edges,
    ConstEigRef<EigRowVec<Index>> src_edges
    )
{

    std::set<std::pair<Index, Index>> unique_pairs;

    for (Index ii = 0; ii < obs_edges.size(); ++ii)
    {
        for (uint8_t iip = 0; iip < 2; ++iip)
        {
            for (Index jj = 0; jj < src_edges.size(); ++jj)
            {
                for (uint8_t jjp = 0; jjp < 2; ++jjp)
                {
                    unique_pairs.insert(
                        std::make_pair(
                            mesh.edge_elems()(iip, obs_edges[ii]),
                            mesh.edge_elems()(jjp, src_edges[ii])
                            )
                        );
                }
            }
        }
    }

    EigMatNX<Index, 2> pairs = EigMatNX<Index, 2>::Zero(2, unique_pairs.size());
    for (auto it = unique_pairs.begin(); it != unique_pairs.end(); ++it)
    {
        Index idx = std::distance(unique_pairs.begin(), it);
        pairs(0, idx) = it->first;
        pairs(1, idx) = it->second;
    }

    return pairs;

};


EigMatNX<Index, 2> IndexGenerator::elem_pairs_from_edges(
    const TriangleMesh<3>& mesh,
    ConstEigRef<EigRowVec<Index>> edges
    )
{
    return elem_pairs_from_edges(mesh, edges, edges);
};

}

