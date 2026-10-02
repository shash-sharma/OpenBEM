// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* ACA+ matrix compression.
*/

#ifndef BEM_ACA_PLUS_H
#define BEM_ACA_PLUS_H

#include "types.hpp"
#include "compression/aca/base.hpp"
#include "compression/matrix/low_rank_matrix.hpp"


namespace bem
{

/**
* \ingroup compression
* @{
*/

/**
* @brief Class for compressing a matrix block with ACA+.
*/
class AcaPlus: public AcaBase
{
public:

    /**
    * @brief Computes the ACA-compressed matrix with ACA+.
    * @param[out] mat - Computed compressed matrix.
    * @param[in] eval - Block evaluator function, taking local row and column indices.
    * @param[in] num_rows - Number of rows of the matrix block to be compressed.
    * @param[in] num_cols - Number of columns of the matrix block to be compressed.
    * @param[in] recompress - Whether to recompress blocks using QR decomposition (optional).
    * @param[in] tol - Relative tolerance for ACA (optional).
    */
    void compute(
        LowRankMatrix<Complex>& mat,
        const EvalType& eval,
        const Index num_rows,
        const Index num_cols,
        const bool recompress = true,
        const Float tol = 1e-4
        ) const override;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "compression/aca/plus.cpp"
#endif

#endif
