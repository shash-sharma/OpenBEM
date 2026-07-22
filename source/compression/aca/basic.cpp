// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Basic ACA-based matrix compression.
*/

#include "compression/aca/basic.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "types.hpp"
#include "constants.hpp"
#include "compression/aca/base.hpp"
#include "compression/matrix/low_rank_matrix.hpp"


namespace bem
{

void AcaBasic::compute(
    LowRankMatrix<Complex>& mat,
    const EvalType& eval,
    const Index num_rows,
    const Index num_cols,
    const bool recompress,
    const Float tol
    ) const
{

    Index max_iter = std::max(num_rows, num_cols);
    Index max_rank = std::min(num_rows, num_cols);

    EigRowVec<Index> full_row_idx = EigRowVec<Index>::LinSpaced(num_rows, 0, num_rows - 1);
    EigRowVec<Index> full_col_idx = EigRowVec<Index>::LinSpaced(num_cols, 0, num_cols - 1);

    mat.u().resize(num_rows, 0);
    mat.v().resize(num_cols, 0);

    std::vector<Index> unused_rows (num_rows);
    std::iota(unused_rows.begin(), unused_rows.end(), 0);

    std::vector<Index> unused_cols (num_cols);
    std::iota(unused_cols.begin(), unused_cols.end(), 0);

    Float norm_estimate = 0;
    Float pivot_scale = 0;
    Index rank = 0;
    Index converged_steps = 0;
    Index pivot_row = 0;
    unused_rows.erase(unused_rows.begin());

    for (Index ii = 0; ii < max_iter; ++ii)
    {

        mat.u().conservativeResize(num_rows, rank + 1);
        mat.v().conservativeResize(num_cols, rank + 1);

        EigRowVec<Index> pivot_row_idx (1);
        pivot_row_idx[0] = pivot_row;
        mat.v().col(rank) = eval(pivot_row_idx, full_col_idx).transpose();
        mat.v().col(rank) -= mat.v().leftCols(rank) * mat.u().leftCols(rank).row(pivot_row).transpose();

        Index pivot_col = unused_cols[0];
        Float max_val = 0;
        for (Index col: unused_cols)
        {
            if (std::abs(mat.v()(col, rank)) > max_val)
            {
                max_val = std::abs(mat.v()(col, rank));
                pivot_col = col;
            }
        }

        Complex pivot_val = mat.v()(pivot_col, rank);
        pivot_scale = std::max(pivot_scale, max_val);

        if (max_val == 0 || max_val < float_eps * pivot_scale)
        {
            if (unused_rows.empty())
                break;

            pivot_row = unused_rows[0];
            unused_rows.erase(unused_rows.begin());
            continue;
        }

        unused_cols.erase(
            std::remove(unused_cols.begin(), unused_cols.end(), pivot_col),
            unused_cols.end()
            );

        mat.v().col(rank) /= pivot_val;

        EigRowVec<Index> pivot_col_idx (1);
        pivot_col_idx[0] = pivot_col;
        mat.u().col(rank) = eval(full_row_idx, pivot_col_idx);
        mat.u().col(rank) -= mat.u().leftCols(rank) * mat.v().leftCols(rank).row(pivot_col).transpose();

        Float norm_update = mat.u().col(rank).squaredNorm() * mat.v().col(rank).squaredNorm();

        norm_update += 2 * std::real(
            (
                (mat.u().leftCols(rank).adjoint() * mat.u().col(rank)).transpose() *
                (mat.v().leftCols(rank).adjoint() * mat.v().col(rank))
                )[0]
            );

        norm_estimate += norm_update;

        if (mat.u().col(rank).norm() * mat.v().col(rank).norm() < tol * std::sqrt(norm_estimate))
            converged_steps++;
        else
            converged_steps = 0;

        if (converged_steps >= 1)
        {
            rank++;
            break;
        }

        if (unused_rows.empty())
        {
            rank++;
            break;
        }

        pivot_row = unused_rows[0];
        Float row_max_val = 0;
        for (Index row: unused_rows)
        {
            if (std::abs(mat.u()(row, rank)) > row_max_val)
            {
                row_max_val = std::abs(mat.u()(row, rank));
                pivot_row = row;
            }
        }

        unused_rows.erase(
            std::remove(unused_rows.begin(), unused_rows.end(), pivot_row),
            unused_rows.end()
            );

        rank++;
        if (rank >= max_rank)
            break;

    }

    mat.u().conservativeResize(num_rows, rank);
    mat.v().conservativeResize(num_cols, rank);
    mat.s().setOnes(rank);

    if (recompress && rank > 0)
        mat.recompress(tol);

    return;

};

}
