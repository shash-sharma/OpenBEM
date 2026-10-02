// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for assembling an ACA-compressed RWG-based BEM operator matrix block.
*/

#include "compression/assemblers/aca_assembler.hpp"

#include <stdexcept>

#include "types.hpp"
#include "matrix/eigen_matrix.hpp"
#include "rwg/assemblers/block_assembler.hpp"
#include "compression/aca/base.hpp"
#include "compression/matrix/low_rank_matrix.hpp"


namespace bem::rwg
{

void AcaAssembler::assemble(
    MatrixBase<Complex>& mat,
    const OperatorBase& op,
    const Complex k
    )
{

    LowRankMatrix<Complex>* mat_lr = dynamic_cast<LowRankMatrix<Complex>*> (&mat);

    if (mat_lr == nullptr)
        throw std::invalid_argument("AcaAssembler::assemble(): `mat` must be a `LowRankMatrix`.");

    BlockAssembler assm (mesh_, index_set_, true);

    auto eval = [&] (
        ConstEigRef<EigRowVec<Index>> local_rows,
        ConstEigRef<EigRowVec<Index>> local_cols
        ) -> EigMat<Complex>
        {
            EigenMatrix<Complex, EigenMatrixType::EIGEN_DENSE> block;
            assm.assemble(block, op, k, local_rows, local_cols);
            return block.raw_matrix();
        };

    aca_.compute(
        *mat_lr, eval, index_set_.rows().size(), index_set_.cols().size(), recompress_, tol_
        );

    return;

};

}
