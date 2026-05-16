// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Index generators for assemblers.
*/

#ifndef ASSEMBLERS_INDGEN_H
#define ASSEMBLERS_INDGEN_H

#include <vector>

#include "types.hpp"


namespace bem
{

// Forward declarations
template <uint8_t dim> class TriangleMesh;

/**
* \defgroup assmutil Utility
* \ingroup assm
* @brief Utility and helper functions for assemblers.
* @{
*/

class IndexGenerator
{
public:

    /**
    * @brief Generates all possible pairs of triangle indices for given observation and
    * source triangle meshes.
    * @param[in] obs_mesh - Observation triangle mesh.
    * @param[in] src_mesh - Source triangle mesh.
    * @return All possible triangle index pairs, with observation indices in the first row,
    * and source indices in the second row.
    */
    static EigMatNX<Index, 2> elem_pairs(
        const TriangleMesh<3>& obs_mesh,
        const TriangleMesh<3>& src_mesh
        );


    /**
     * @brief Generates all possible pairs of triangle indices for a given triangle mesh.
     * @param[in] mesh - Triangle mesh.
     * @return All possible triangle index pairs, with observation indices in the first row,
     * and source indices in the second row.
     */
    static EigMatNX<Index, 2> elem_pairs(const TriangleMesh<3>& mesh);


    /**
    * @brief Makes all possible pairs of triangle indices for given observation and source triangle indices.
    * @param[in] obs_elems - Observation triangle indices.
    * @param[in] src_elems - Source triangle indices.
    * @return All possible triangle index pairs, with observation indices in the first row,
    * and source indices in the second row.
    */
    static EigMatNX<Index, 2> elem_pairs(
        ConstEigRef<EigRowVec<Index>> obs_elems,
        ConstEigRef<EigRowVec<Index>> src_elems
        );


    /**
    * @brief Makes all possible pairs of given triangle indices.
    * @param[in] elems - Triangle indices.
    * @return Triangle index pairs, with observation indices in the first row,
    * and source indices in the second row.
    */
    static EigMatNX<Index, 2> elem_pairs(ConstEigRef<EigRowVec<Index>> elems);


    /**
    * @brief Makes self-pairs of given triangle indices.
    * @param[in] elems - Triangle indices.
    * @return Triangle index pairs, with observation indices in the first row,
    * and source indices in the second row.
    */
    static EigMatNX<Index, 2> elem_pairs_self(ConstEigRef<EigRowVec<Index>> elems);


    /**
    * @brief Makes all possible unique element pairs from given edge pairs.
    * @param[in] mesh - Triangle mesh.
    * @param[in] obs_edges - Observation edge indices.
    * @param[in] src_edges - Source edge indices.
    * @return Triangle index pairs, with observation indices in the first row,
    * and source indices in the second row.
    */
    static EigMatNX<Index, 2> elem_pairs_from_edges(
        const TriangleMesh<3>& mesh,
        ConstEigRef<EigRowVec<Index>> obs_edges,
        ConstEigRef<EigRowVec<Index>> src_edges
        );


    /**
    * @brief Makes all possible unique element pairs from given edge indices.
    * @param[in] mesh - Triangle mesh.
    * @param[in] edges - Edge indices.
    * @return Triangle index pairs, with observation indices in the first row,
    * and source indices in the second row.
    */
    static EigMatNX<Index, 2> elem_pairs_from_edges(
        const TriangleMesh<3>& mesh,
        ConstEigRef<EigRowVec<Index>> edges
        );

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/index_generator.cpp"
#endif

#endif

