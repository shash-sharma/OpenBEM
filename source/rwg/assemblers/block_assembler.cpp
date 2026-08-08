/**
* @file
* Classes for assembling RWG-based BEM operator matrix sub-blocks.
*/

#include "rwg/assemblers/block_assembler.hpp"

#include <memory>
#include <unordered_map>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/assemblers/indexing.hpp"


namespace bem::rwg
{

void BlockAssembler::assemble(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    const Complex k
    )
{

    ConstEigRef<EigRowVec<Index>> obs_elems = (op.obs_dof() == OperatorDof::EDGE) ?
        row_elems_from_edges_ : index_set_.rows();

    ConstEigRef<EigRowVec<Index>> src_elems = (op.src_dof() == OperatorDof::EDGE) ?
        col_elems_from_edges_ : index_set_.cols();

    assemble_from_elems(
        mat, op, k,
        obs_elems, src_elems,
        row_map_, col_map_,
        index_set_.num_rows(), index_set_.num_cols()
        );

    return;

};


void BlockAssembler::assemble(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    const Complex k,
    ConstEigRef<EigRowVec<Index>> local_rows,
    ConstEigRef<EigRowVec<Index>> local_cols
    )
{

    bool all_rows = (local_rows.size() == index_set_.num_rows());
    bool all_cols = (local_cols.size() == index_set_.num_cols());

    EigRowVec<Index> obs_elems;
    std::unordered_map<Index, Index> local_row_map;

    if (all_rows)
        obs_elems = (op.obs_dof() == OperatorDof::EDGE) ?
            row_elems_from_edges_ : index_set_.rows();
    else
    {
        std::vector<Index> rows_global_vec;
        rows_global_vec.reserve(local_rows.size());
        for (Index ii = 0; ii < local_rows.size(); ++ii)
        {
            if (local_rows[ii] >= index_set_.num_rows())
                continue;
            Index global_idx = index_set_.rows()[local_rows[ii]];
            rows_global_vec.push_back(global_idx);
            local_row_map.insert({ global_idx, ii });
        }
        EigRowVec<Index> rows_global = Eigen::Map<const EigRowVec<Index>> (
            rows_global_vec.data(), rows_global_vec.size()
            );
        obs_elems = (op.obs_dof() == OperatorDof::EDGE) ?
            IndexGenerator::elems_from_edges(mesh_, rows_global) : rows_global;
    }

    EigRowVec<Index> src_elems;
    std::unordered_map<Index, Index> local_col_map;

    if (all_cols)
        src_elems = (op.src_dof() == OperatorDof::EDGE) ?
            col_elems_from_edges_ : index_set_.cols();
    else
    {
        std::vector<Index> cols_global_vec;
        cols_global_vec.reserve(local_cols.size());
        for (Index ii = 0; ii < local_cols.size(); ++ii)
        {
            if (local_cols[ii] >= index_set_.num_cols())
                continue;
            Index global_idx = index_set_.cols()[local_cols[ii]];
            cols_global_vec.push_back(global_idx);
            local_col_map.insert({ global_idx, ii });
        }
        EigRowVec<Index> cols_global = Eigen::Map<const EigRowVec<Index>> (
            cols_global_vec.data(), cols_global_vec.size()
            );
        src_elems = (op.src_dof() == OperatorDof::EDGE) ?
            IndexGenerator::elems_from_edges(mesh_, cols_global) : cols_global;
    }

    assemble_from_elems(
        mat, op, k,
        obs_elems, src_elems,
        all_rows ? row_map_ : local_row_map,
        all_cols ? col_map_ : local_col_map,
        local_rows.size(), local_cols.size()
        );

    return;

};


void BlockAssembler::assemble_from_elems(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    const Complex k,
    ConstEigRef<EigRowVec<Index>> obs_elems,
    ConstEigRef<EigRowVec<Index>> src_elems,
    const std::unordered_map<Index, Index>& active_row_map,
    const std::unordered_map<Index, Index>& active_col_map,
    const Index out_rows,
    const Index out_cols
    )
{

    EigMatNX<Index, 2> elem_pairs = IndexGenerator::elem_pairs(obs_elems, src_elems);

    mat.resize(out_rows, out_cols);

#pragma omp parallel for
    for (Index ii = 0; ii < elem_pairs.cols(); ++ii)
    {
        Triangle<3> obs_tri = mesh_.elem_primitive(elem_pairs(0, ii));
        Triangle<3> src_tri = mesh_.elem_primitive(elem_pairs(1, ii));

        EigMat<Complex> values = op.compute(
            k, obs_tri, src_tri
            );

#pragma omp critical
        fill_matrix(mat, op, elem_pairs.col(ii), values, active_row_map, active_col_map);
    }

    mat.assemble();

    return;

};


void BlockAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    ConstEigRef<EigColVecN<Index, 2>> elem_pair,
    ConstEigRef<EigMat<Complex>> values,
    const std::unordered_map<Index, Index>& active_row_map,
    const std::unordered_map<Index, Index>& active_col_map
    )
{

    if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::EDGE)
    {
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = mesh_.elem_edges()(src_edge, elem_pair[1]);
            auto icol = active_col_map.find(col);
            if (icol != active_col_map.end())
                for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
                {
                    Index row = mesh_.elem_edges()(obs_edge, elem_pair[0]);
                    auto irow = active_row_map.find(row);
                    if (irow != active_row_map.end())
// #pragma omp critical
                        mat.add_value(irow->second, icol->second, values(obs_edge, src_edge));
                }
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::EDGE)
    {
        Index row = active_row_map.at(elem_pair[0]);
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = mesh_.elem_edges()(src_edge, elem_pair[1]);
            auto icol = active_col_map.find(col);
            if (icol != active_col_map.end())
// #pragma omp critical
                mat.add_value(row, icol->second, values(0, src_edge));
        }
    }

    else if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::FACE)
    {
        Index col = active_col_map.at(elem_pair[1]);
        for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
        {
            Index row = mesh_.elem_edges()(obs_edge, elem_pair[0]);
            auto irow = active_row_map.find(row);
            if (irow != active_row_map.end())
// #pragma omp critical
                mat.add_value(irow->second, col, values(obs_edge, 0));
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::FACE)
    {
        Index row = active_row_map.at(elem_pair[0]);
        Index col = active_col_map.at(elem_pair[1]);
// #pragma omp critical
        mat.set_value(row, col, values(0, 0));
    }

    return;

};

}

