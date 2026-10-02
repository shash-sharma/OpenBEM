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

#include "compression/aca/plus.hpp"

#include <vector>
#include <algorithm>
#include <cmath>

#include "types.hpp"
#include "constants.hpp"
#include "compression/aca/base.hpp"
#include "compression/matrix/low_rank_matrix.hpp"


namespace bem
{

void AcaPlus::compute(
    LowRankMatrix<Complex>& mat,
    const EvalType& eval,
    const Index num_rows,
    const Index num_cols,
    const bool recompress,
    const Float tol
    ) const
{

    const Index max_rank = std::min(num_rows, num_cols);

    const EigRowVec<Index> full_row_idx = EigRowVec<Index>::LinSpaced(num_rows, 0, num_rows - 1);
    const EigRowVec<Index> full_col_idx = EigRowVec<Index>::LinSpaced(num_cols, 0, num_cols - 1);

    mat.u().resize(num_rows, 0);
    mat.v().resize(num_cols, 0);

    std::vector<bool> used_rows (num_rows, false);
    std::vector<bool> used_cols (num_cols, false);
    std::vector<bool> probed_rows (num_rows, false);
    std::vector<bool> probed_cols (num_cols, false);

    Index rank = 0;
    Float norm_estimate = 0;
    Float pivot_scale = 0;

    const auto residual_row = [&] (const Index row) -> EigColVec<Complex>
    {
        EigRowVec<Index> row_idx (1);
        row_idx[0] = row;

        EigColVec<Complex> res = eval(row_idx, full_col_idx).transpose();
        res -= mat.v().leftCols(rank) * mat.u().leftCols(rank).row(row).transpose();

        return res;
    };

    const auto residual_col = [&] (const Index col) -> EigColVec<Complex>
    {
        EigRowVec<Index> col_idx (1);
        col_idx[0] = col;

        EigColVec<Complex> res = eval(full_row_idx, col_idx);
        res -= mat.u().leftCols(rank) * mat.v().leftCols(rank).row(col).transpose();

        return res;
    };

    const auto largest_unused = [] (
        const EigColVec<Complex>& vec,
        const std::vector<bool>& used
        ) -> Index
    {
        Index best = vec.size();
        Float best_val = -1;

        for (Index ii = 0; ii < (Index) vec.size(); ++ii)
        {
            if (used[ii])
                continue;

            if (std::abs(vec[ii]) > best_val)
            {
                best_val = std::abs(vec[ii]);
                best = ii;
            }
        }

        return best;
    };

    Index ref_row = num_rows;
    Index ref_col = num_cols;
    EigColVec<Complex> ref_row_res, ref_col_res;

    const auto refresh_ref_row = [&] ()
    {
        ref_row = num_rows;
        Float min_coverage = 0;
        Float min_val = 0;

        for (Index row = 0; row < num_rows; ++row)
        {
            if (used_rows[row] || probed_rows[row])
                continue;

            const Float coverage = mat.u().row(row).squaredNorm();

            Float val = 0;
            if (ref_col < num_cols)
                val = std::abs(ref_col_res[row]);

            if (ref_row == num_rows || coverage < min_coverage ||
                (coverage == min_coverage && val < min_val))
            {
                ref_row = row;
                min_coverage = coverage;
                min_val = val;
            }
        }

        if (ref_row == num_rows)
            return;

        probed_rows[ref_row] = true;
        ref_row_res = residual_row(ref_row);
    };

    const auto refresh_ref_col = [&] ()
    {
        ref_col = num_cols;
        Float min_coverage = 0;
        Float min_val = 0;

        for (Index col = 0; col < num_cols; ++col)
        {
            if (used_cols[col] || probed_cols[col])
                continue;

            const Float coverage = mat.v().row(col).squaredNorm();

            Float val = 0;
            if (ref_row < num_rows)
                val = std::abs(ref_row_res[col]);

            if (ref_col == num_cols || coverage < min_coverage ||
                (coverage == min_coverage && val < min_val))
            {
                ref_col = col;
                min_coverage = coverage;
                min_val = val;
            }
        }

        if (ref_col == num_cols)
            return;

        probed_cols[ref_col] = true;
        ref_col_res = residual_col(ref_col);
    };

    const auto probes_converged = [&] (const Float norm) -> bool
    {
        bool converged = true;

        if (ref_row < num_rows && ref_row_res.norm() * std::sqrt((Float) num_rows) >= tol * norm)
            converged = false;

        if (ref_col < num_cols && ref_col_res.norm() * std::sqrt((Float) num_cols) >= tol * norm)
            converged = false;

        return converged;
    };

    refresh_ref_row();
    refresh_ref_col();

    const Index max_steps = num_rows + num_cols + max_rank;

    for (Index step = 0; step < max_steps && rank < max_rank; ++step)
    {

        const Float zero_level = float_eps * pivot_scale;

        Index cand_col = num_cols;
        Float row_probe_val = 0;
        if (ref_row < num_rows)
        {
            cand_col = largest_unused(ref_row_res, used_cols);
            if (cand_col < num_cols)
                row_probe_val = std::abs(ref_row_res[cand_col]);
        }

        Index cand_row = num_rows;
        Float col_probe_val = 0;
        if (ref_col < num_cols)
        {
            cand_row = largest_unused(ref_col_res, used_rows);
            if (cand_row < num_rows)
                col_probe_val = std::abs(ref_col_res[cand_row]);
        }

        if (ref_row < num_rows && row_probe_val <= zero_level)
        {
            refresh_ref_row();
            continue;
        }

        if (ref_col < num_cols && col_probe_val <= zero_level)
        {
            refresh_ref_col();
            continue;
        }

        if (ref_row == num_rows && ref_col == num_cols)
            break;

        const bool row_first = col_probe_val > row_probe_val;

        Index pivot_row = num_rows;
        Index pivot_col = num_cols;
        EigColVec<Complex> u_new, v_new;

        if (row_first)
        {
            pivot_row = cand_row;
            v_new = residual_row(pivot_row);
            pivot_col = largest_unused(v_new, used_cols);

            if (pivot_col == num_cols)
            {
                used_rows[pivot_row] = true;
                continue;
            }

            u_new = residual_col(pivot_col);
        }
        else
        {
            pivot_col = cand_col;
            u_new = residual_col(pivot_col);
            pivot_row = largest_unused(u_new, used_rows);

            if (pivot_row == num_rows)
            {
                used_cols[pivot_col] = true;
                continue;
            }

            v_new = residual_row(pivot_row);
        }

        const Complex pivot_val = v_new[pivot_col];
        const Float pivot_abs = std::abs(pivot_val);

        pivot_scale = std::max(pivot_scale, pivot_abs);

        if (pivot_abs == 0 || pivot_abs < float_eps * pivot_scale)
        {
            if (row_first)
                used_rows[pivot_row] = true;
            else
                used_cols[pivot_col] = true;

            continue;
        }

        v_new /= pivot_val;

        Float norm_update = u_new.squaredNorm() * v_new.squaredNorm();

        norm_update += 2 * std::real(
            (
                (mat.u().leftCols(rank).adjoint() * u_new).transpose() *
                (mat.v().leftCols(rank).adjoint() * v_new)
                )[0]
            );

        norm_estimate += norm_update;

        mat.u().conservativeResize(num_rows, rank + 1);
        mat.v().conservativeResize(num_cols, rank + 1);
        mat.u().col(rank) = u_new;
        mat.v().col(rank) = v_new;

        used_rows[pivot_row] = true;
        used_cols[pivot_col] = true;

        if (ref_row < num_rows)
            ref_row_res -= u_new[ref_row] * v_new;

        if (ref_col < num_cols)
            ref_col_res -= u_new * v_new[ref_col];

        rank++;

        if (ref_row == pivot_row)
            refresh_ref_row();

        if (ref_col == pivot_col)
            refresh_ref_col();

        const Float norm = std::sqrt(norm_estimate);

        if (u_new.norm() * v_new.norm() >= tol * norm || !probes_converged(norm))
            continue;

        refresh_ref_row();
        refresh_ref_col();

        if (probes_converged(norm))
            break;

    }

    mat.s().setOnes(rank);

    if (recompress && rank > 0)
        mat.recompress(tol);

    return;

};

}
