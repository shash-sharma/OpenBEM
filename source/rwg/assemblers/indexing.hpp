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

#ifndef ASSEMBLERS_IND_H
#define ASSEMBLERS_IND_H

#include <vector>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "rwg/function_space.hpp"


namespace bem
{

/**
* \ingroup assm
* @{
*/

/**
* @brief Data structure of index sets defining a block of a matrix.
*/
struct IndexSet
{
public:

    /**
    * @brief Constructs an `IndexSet` object with row and column index lists.
    * @param[in] rows - List of row indices.
    * @param[in] cols - List of column indices.
    */
    IndexSet(
        ConstEigRef<EigRowVec<Index>> rows,
        ConstEigRef<EigRowVec<Index>> cols
        ): rows_(rows), cols_(cols) {};


    /**
    * @brief Constructs an `IndexSet` object for a contiguous block.
    * @param[in] row_start - Starting row index.
    * @param[in] col_start - Starting column index.
    * @param[in] num_rows - Number of rows in the block.
    * @param[in] num_cols - Number of columns in the block.
    */
    IndexSet(
        const Index row_start,
        const Index col_start,
        const Index num_rows,
        const Index num_cols
        ):
        IndexSet(
            EigRowVec<Index>::LinSpaced(num_rows, row_start, row_start + num_rows - 1),
            EigRowVec<Index>::LinSpaced(num_cols, col_start, col_start + num_cols - 1)
            ) {};


    /**
    * @brief Returns a read-only reference to the row indices.
    * @return Read-only reference to row indices.
    */
    const EigRowVec<Index>& rows() const { return rows_; };


    /**
    * @brief Returns a read-only reference to the column indices.
    * @return Read-only reference to column indices.
    */
    const EigRowVec<Index>& cols() const { return cols_; };


    /**
    * @brief Returns the number of rows.
    * @return Number of rows.
    */
    Index num_rows() const { return rows_.size(); };


    /**
    * @brief Returns the number of columns.
    * @return Number of columns.
    */
    Index num_cols() const { return cols_.size(); };


protected:

    const EigRowVec<Index> rows_;
    const EigRowVec<Index> cols_;

};


/**
* @brief Class to generate index sets and maps across element types.
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
    * @brief Returns all unique element pairs associated with a given index set.
    * @param[in] mesh - Triangle mesh.
    * @param[in] index_set - Index set defining degrees of freedom.
    * @param[in] row_dof - Degrees of freedom associated with rows.
    * @param[in] col_dof - Degrees of freedom associated with columns.
    * @return Triangle index pairs.
    */
    static EigMatNX<Index, 2> elem_pairs(
        const TriangleMesh<3>& mesh,
        const IndexSet& index_set,
        const OperatorDof row_dof,
        const OperatorDof col_dof
        );


    /**
    * @brief Returns unique pairs from given pairs.
    * @param[in] pairs - Index pairs.
    * @return Unique index pairs.
    */
    static EigMatNX<Index, 2> unique_pairs(
        ConstEigRef<EigMatNX<Index, 2>> pairs
        );


    /**
    * @brief Returns all unique element indices associated with given edge indices.
    * @param[in] mesh - Triangle mesh.
    * @param[in] edges - Edge indices.
    * @return Triangle indices.
    */
    static EigRowVec<Index> elems_from_edges(
        const TriangleMesh<3>& mesh,
        ConstEigRef<EigRowVec<Index>> edges
        );

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/indexing.cpp"
#endif

#endif

