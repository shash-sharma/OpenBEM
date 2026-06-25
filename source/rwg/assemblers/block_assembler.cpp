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
#include "rwg/assemblers/operator_assembler.hpp"


namespace bem::rwg
{

void BlockAssembler::assemble(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    const Complex k
    )
{

    EigMatNX<Index, 2> elem_pairs = IndexGenerator::elem_pairs(
        mesh_, index_set_, op.obs_dof(), op.src_dof()
        );

    std::unique_ptr<MatrixBase<Complex>> temp = mat.clone();

    OperatorAssembler assm (mesh_, elem_pairs);
    assm.assemble(*temp, op, k);

    mat.resize(temp->num_rows(), temp->num_cols());

#pragma omp parallel for collapse(2)
    for (Index ci = 0; ci < index_set_.num_cols(); ++ci)
    {
        for (Index ri = 0; ri < index_set_.num_rows(); ++ri)
        {
            Index row = index_set_.rows()[ri];
            Index col = index_set_.cols()[ci];

            mat.set_value(row, col, temp->value(row, col));
        }
    }

    mat.assemble();

    return;

};


// void BlockAssembler::get_block(
//     EigMat<Complex>& mat,
//     const OperatorBase& op,
//     const Complex k
//     )
// {

//     EigMatNX<Index, 2> elem_pairs = IndexGenerator::elem_pairs(
//         mesh_, index_set_, op.obs_dof(), op.src_dof()
//         );

//     EigenMatrix<Complex, EigenMatrixType::EIGEN_SPARSE> temp;

//     OperatorAssembler assm (mesh_, elem_pairs);
//     assm.assemble(temp, op, k);

//     mat.resize(index_set_.num_rows(), index_set_.num_cols());

// #pragma omp parallel for collapse(2)
//     for (Index ci = 0; ci < index_set_.num_cols(); ++ci)
//     {
//         for (Index ri = 0; ri < index_set_.num_rows(); ++ri)
//         {
//             Index row = index_set_.rows()[ri];
//             Index col = index_set_.cols()[ci];

//             mat(ri, ci) = temp.value(row, col);
//         }
//     }

//     return;

// };


void BlockAssembler::get_block(
    EigMat<Complex>& mat,
    const OperatorBase& op,
    const Complex k
    )
{

    EigMatNX<Index, 2> elem_pairs = IndexGenerator::elem_pairs(
        mesh_, index_set_, op.obs_dof(), op.src_dof()
        );

    std::unordered_map<Index, Index> row_map;
    for (Index ii = 0; ii < index_set_.num_rows(); ++ii)
        row_map.insert({ index_set_.rows()[ii], ii });

    std::unordered_map<Index, Index> col_map;
    for (Index ii = 0; ii < index_set_.num_cols(); ++ii)
        col_map.insert({ index_set_.cols()[ii], ii });

    mat.resize(index_set_.num_rows(), index_set_.num_cols());
    mat.setZero();

#pragma omp parallel
    {
        std::unique_ptr<OperatorBase> opc = op.clone();

#pragma omp for
        for (Index ii = 0; ii < elem_pairs.cols(); ++ii)
        {
            Triangle<3> obs_tri = mesh_.elem_primitive(elem_pairs(0, ii));
            Triangle<3> src_tri = mesh_.elem_primitive(elem_pairs(1, ii));

            EigMat<Complex> values = opc->compute(
                k, obs_tri, src_tri
                );

            if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::EDGE)
            {
                for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
                {
                    Index col = mesh_.elem_edges()(src_edge, elem_pairs(1, ii));
                    auto icol = col_map.find(col);
                    if (icol != col_map.end())
                        for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
                        {
                            Index row = mesh_.elem_edges()(obs_edge, elem_pairs(0, ii));
                            auto irow = row_map.find(row);
                            if (irow != row_map.end())
#pragma omp critical
                                mat(irow->second, icol->second) += values(obs_edge, src_edge);
                        }
                }
            }

            else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::EDGE)
            {
                Index row = row_map.at(elem_pairs(0, ii));
                for (uint8_t src_edge = 0; src_edge < 3; ++src_edge)
                {
                    Index col = mesh_.elem_edges()(src_edge, elem_pairs(1, ii));
                    auto icol = col_map.find(col);
                    if (icol != col_map.end())
#pragma omp critical
                        mat(row, icol->second) += values(0, src_edge);
                }
            }

            else if (op.obs_dof() == OperatorDof::EDGE && op.src_dof() == OperatorDof::FACE)
            {
                Index col = col_map.at(elem_pairs(1, ii));
                for (uint8_t obs_edge = 0; obs_edge < 3; ++obs_edge)
                {
                    Index row = mesh_.elem_edges()(obs_edge, elem_pairs(0, ii));
                    auto irow = row_map.find(row);
                    if (irow != row_map.end())
#pragma omp critical
                        mat(irow->second, col) += values(obs_edge, 0);
                }
            }

            else if (op.obs_dof() == OperatorDof::FACE && op.src_dof() == OperatorDof::FACE)
            {
                Index row = row_map.at(elem_pairs(0, ii));
                Index col = col_map.at(elem_pairs(1, ii));
#pragma omp critical
                mat(row, col) = values(0, 0);
            }
        }
    }

    return;

};

}

