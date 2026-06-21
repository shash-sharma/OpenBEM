/**
* @file
* Classes for assembling RWG-based BEM operator matrix sub-blocks.
*/

#include "rwg/assemblers/block_assembler.hpp"

#include <memory>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "matrix/base.hpp"
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

    EigMatNX<Index, 2> elem_pairs;

    if (op.obs_dof() == 3 && op.src_dof() == 3)
        elem_pairs = IndexGenerator::elem_pairs_from_edges(
            mesh_, index_set_.rows(), index_set_.cols()
            );

    else if (op.obs_dof() == 1 && op.src_dof() == 3)
        elem_pairs = IndexGenerator::elem_pairs_from_elems_edges(
            mesh_, index_set_.rows(), index_set_.cols()
            );

    else if (op.obs_dof() == 3 && op.src_dof() == 1)
        elem_pairs = IndexGenerator::elem_pairs_from_edges_elems(
            mesh_, index_set_.rows(), index_set_.cols()
            );

    else if (op.obs_dof() == 1 && op.src_dof() == 1)
        elem_pairs = IndexGenerator::elem_pairs(
            index_set_.rows(), index_set_.cols()
            );

    std::unique_ptr<MatrixBase<Complex>> temp = mat.clone();

    OperatorAssembler assm (mesh_, elem_pairs);
    assm.assemble(*temp, op, k);

    mat.resize(temp->num_rows(), temp->num_cols());

    for (Index ri = 0; ri < index_set_.num_rows(); ++ri)
    {
        for (Index ci = 0; ci < index_set_.num_cols(); ++ci)
        {
            Index row = index_set_.rows()[ri];
            Index col = index_set_.cols()[ci];

            mat.set_value(row, col, temp->value(row, col));
        }
    }

    mat.assemble();

    return;

};


// void BlockAssembler::assemble_test(
//     MatrixBase<Complex>& mat,
//     const OperatorBase& op,
//     const Complex k
//     )
// {

//     EigMatNX<Index, 2> elem_pairs;

//     if (op.obs_dof() == 3 && op.src_dof() == 3)
//     {
//         mat.resize(obs_mesh_.num_edges(), src_mesh_.num_edges());
//         elem_pairs = IndexGenerator::elem_pairs_from_edges(
//             mesh_, index_set_.rows(), index_set_.cols()
//             );
//     }

//     else if (op.obs_dof() == 1 && op.src_dof() == 3)
//     {
//         mat.resize(obs_mesh_.num_elems(), src_mesh_.num_edges());
//         elem_pairs = IndexGenerator::elem_pairs_from_elems_edges(
//             mesh_, index_set_.rows(), index_set_.cols()
//             );
//     }

//     else if (op.obs_dof() == 3 && op.src_dof() == 1)
//     {
//         mat.resize(obs_mesh_.num_edges(), src_mesh_.num_elems());
//         elem_pairs = IndexGenerator::elem_pairs_from_edges_elems(
//             mesh_, index_set_.rows(), index_set_.cols()
//             );
//     }

//     else if (op.obs_dof() == 1 && op.src_dof() == 1)
//     {
//         mat.resize(obs_mesh_.num_elems(), src_mesh_.num_elems());
//         elem_pairs = IndexGenerator::elem_pairs(
//             index_set_.rows(), index_set_.cols()
//             );
//     }

//     EigMatNX<Complex, 3> values (op.obs_dof(), op.src_dof() * elem_pairs.size());

// #pragma omp parallel
//     {
//         std::unique_ptr<OperatorBase> opc = op.clone();

// #pragma omp for
//         for (Index ii = 0; ii < elem_pairs.cols(); ++ii)
//         {
//             Triangle<3> obs_tri = obs_mesh_.elem_primitive(elem_pairs_(0, ii));
//             Triangle<3> src_tri = src_mesh_.elem_primitive(elem_pairs_(1, ii));

//             values.block(0, ii * op.src_dof(), op.obs_dof(), op.src_dof()) = opc->compute(
//                 k, obs_tri, src_tri
//                 );
//         }
//     }

//     mat.preallocate(index_set_.num_rows() * index_set_.num_cols());


// #pragma omp critical
//             fill_matrix(mat, op, elem_pairs_.col(ii), values);






//     std::unique_ptr<MatrixBase<Complex>> temp = mat.clone();

//     OperatorAssembler assm (mesh_, elem_pairs);
//     assm.assemble(*temp, op, k);

//     mat.resize(temp->num_rows(), temp->num_cols());

//     for (Index ri = 0; ri < index_set_.num_rows(); ++ri)
//     {
//         for (Index ci = 0; ci < index_set_.num_cols(); ++ci)
//         {
//             Index row = index_set_.rows()[ri];
//             Index col = index_set_.cols()[ci];

//             mat.set_value(row, col, temp->value(row, col));
//         }
//     }

//     mat.assemble();

//     return;

// };

}

