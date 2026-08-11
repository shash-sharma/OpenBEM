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

    ConstEigRef<EigRowVec<Index>> obs_faces = (op.obs_dof() == OperatorDof::EDGE) ?
        row_faces_from_edges_ : index_set_.rows();

    ConstEigRef<EigRowVec<Index>> src_faces = (op.src_dof() == OperatorDof::EDGE) ?
        col_faces_from_edges_ : index_set_.cols();

    assemble_from_faces(
        mat, op, k,
        obs_faces, src_faces,
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

    EigRowVec<Index> obs_faces;
    std::unordered_map<Index, Index> local_row_map;

    if (all_rows)
        obs_faces = (op.obs_dof() == OperatorDof::EDGE) ?
            row_faces_from_edges_ : index_set_.rows();
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
        obs_faces = (op.obs_dof() == OperatorDof::EDGE) ?
            IndexGenerator::faces_from_edges(mesh_, rows_global) : rows_global;
    }

    EigRowVec<Index> src_faces;
    std::unordered_map<Index, Index> local_col_map;

    if (all_cols)
        src_faces = (op.src_dof() == OperatorDof::EDGE) ?
            col_faces_from_edges_ : index_set_.cols();
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
        src_faces = (op.src_dof() == OperatorDof::EDGE) ?
            IndexGenerator::faces_from_edges(mesh_, cols_global) : cols_global;
    }

    assemble_from_faces(
        mat, op, k,
        obs_faces, src_faces,
        all_rows ? row_map_ : local_row_map,
        all_cols ? col_map_ : local_col_map,
        local_rows.size(), local_cols.size()
        );

    return;

};


void BlockAssembler::assemble_from_faces(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    const Complex k,
    ConstEigRef<EigRowVec<Index>> obs_faces,
    ConstEigRef<EigRowVec<Index>> src_faces,
    const std::unordered_map<Index, Index>& active_row_map,
    const std::unordered_map<Index, Index>& active_col_map,
    const Index out_rows,
    const Index out_cols
    )
{

    EigMatNX<Index, 2> face_pairs = IndexGenerator::face_pairs(obs_faces, src_faces);

    mat.resize(out_rows, out_cols);

    if (use_integration_cache_)
    {
        std::lock_guard<std::mutex> lock (integration_cache_mutex_);
        if (integration_cache_op_ != &op || integration_cache_k_ != k)
        {
            integration_cache_.clear();
            integration_cache_op_ = &op;
            integration_cache_k_ = k;
        }
    }

    std::mutex mat_mutex;

#pragma omp parallel for
    for (Index ii = 0; ii < face_pairs.cols(); ++ii)
    {
        EigMat<Complex> values;
        bool cached = false;

        std::pair<Index, Index> pair_key (face_pairs(0, ii), face_pairs(1, ii));

        if (use_integration_cache_)
        {
            std::lock_guard<std::mutex> lock (integration_cache_mutex_);
            auto it = integration_cache_.find(pair_key);
            if (it != integration_cache_.end())
            {
                values = it->second;
                cached = true;
            }
        }

        if (!cached)
        {
            Triangle<3> obs_tri = mesh_.face_primitive(face_pairs(0, ii));
            Triangle<3> src_tri = mesh_.face_primitive(face_pairs(1, ii));

            values = op.compute(
                k, obs_tri, src_tri
                );

            if (use_integration_cache_)
            {
                std::lock_guard<std::mutex> lock (integration_cache_mutex_);
                integration_cache_.insert({ pair_key, values });
            }
        }

        {
            std::lock_guard<std::mutex> lock (mat_mutex);
            fill_matrix(mat, op, face_pairs.col(ii), values, active_row_map, active_col_map);
        }
    }

    mat.assemble();

    return;

};


void BlockAssembler::fill_matrix(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    ConstEigRef<EigColVecN<Index, 2>> face_pair,
    ConstEigRef<EigMat<Complex>> values,
    const std::unordered_map<Index, Index>& active_row_map,
    const std::unordered_map<Index, Index>& active_col_map
    )
{

    if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::EDGE)
    {
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = mesh_.face_edges()(src_edge, face_pair[1]);
            auto icol = active_col_map.find(col);
            if (icol != active_col_map.end())
                for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
                {
                    Index row = mesh_.face_edges()(obs_edge, face_pair[0]);
                    auto irow = active_row_map.find(row);
                    if (irow != active_row_map.end())
// #pragma omp critical
                        mat.add_value(irow->second, icol->second, values(obs_edge, src_edge));
                }
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::EDGE)
    {
        Index row = active_row_map.at(face_pair[0]);
        for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
        {
            Index col = mesh_.face_edges()(src_edge, face_pair[1]);
            auto icol = active_col_map.find(col);
            if (icol != active_col_map.end())
// #pragma omp critical
                mat.add_value(row, icol->second, values(0, src_edge));
        }
    }

    else if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::FACE)
    {
        Index col = active_col_map.at(face_pair[1]);
        for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
        {
            Index row = mesh_.face_edges()(obs_edge, face_pair[0]);
            auto irow = active_row_map.find(row);
            if (irow != active_row_map.end())
// #pragma omp critical
                mat.add_value(irow->second, col, values(obs_edge, 0));
        }
    }

    else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::FACE)
    {
        Index row = active_row_map.at(face_pair[0]);
        Index col = active_col_map.at(face_pair[1]);
// #pragma omp critical
        mat.set_value(row, col, values(0, 0));
    }

    return;

};

}

