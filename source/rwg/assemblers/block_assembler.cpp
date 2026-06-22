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


void BlockAssembler::get_block(
    EigMat<Complex>& mat,
    const OperatorBase& op,
    const Complex k
    )
{

    EigMatNX<Index, 2> elem_pairs = IndexGenerator::elem_pairs(
        mesh_, index_set_, op.obs_dof(), op.src_dof()
        );

    EigenMatrix<Complex, EigenMatrixType::EIGEN_SPARSE> temp;

    OperatorAssembler assm (mesh_, elem_pairs);
    assm.assemble(temp, op, k);

    mat.resize(index_set_.num_rows(), index_set_.num_cols());

#pragma omp parallel for collapse(2)
    for (Index ci = 0; ci < index_set_.num_cols(); ++ci)
    {
        for (Index ri = 0; ri < index_set_.num_rows(); ++ri)
        {
            Index row = index_set_.rows()[ri];
            Index col = index_set_.cols()[ci];

            mat(ri, ci) = temp.value(row, col);
        }
    }

    return;

};

}

