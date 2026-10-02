// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for ACA-based matrix compression.
*/

#ifndef BEM_ACA_BASE_H
#define BEM_ACA_BASE_H

#include <functional>

#include "types.hpp"
#include "compression/matrix/low_rank_matrix.hpp"


namespace bem
{

/**
* \ingroup compression
* @{
*/

/**
* @brief Base class for ACA-based matrix compression.
*/
class AcaBase
{
public:

    using EvalType = std::function<EigMat<Complex> (
        ConstEigRef<EigRowVec<Index>> eval_rows,
        ConstEigRef<EigRowVec<Index>> eval_cols
        )>;


    /**
    * @brief Computes the ACA-compressed matrix.
    * @param[out] mat - Computed compressed matrix.
    * @param[in] eval - Block evaluator function, taking local row and column indices.
    * @param[in] num_rows - Number of rows of the matrix block to be compressed.
    * @param[in] num_cols - Number of columns of the matrix block to be compressed.
    * @param[in] recompress - Whether to recompress blocks using QR-SVD (optional).
    * @param[in] tol - Relative tolerance for ACA (optional).
    */
    virtual void compute(
        LowRankMatrix<Complex>& mat,
        const EvalType& eval,
        const Index num_rows,
        const Index num_cols,
        const bool recompress = true,
        const Float tol = 1e-4
        ) const = 0;


    /**
    * @brief Virtual destructor.
    */
    virtual ~AcaBase() = default;

};

/**
* @}
*/

}

#endif
