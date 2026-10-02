// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class wrapping Eigen-based low-rank compressed matrices.
*/

#include "compression/matrix/low_rank_matrix.hpp"

#include <type_traits>
#include <stdexcept>
#include <memory>
#include <vector>
#include <algorithm>

#include <external/Eigen/Dense>
#include <external/Eigen/SVD>

#include "types.hpp"
#include "constants.hpp"
#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"


namespace bem
{

template <typename T>
void LowRankMatrix<T>::set_usv(
    ConstEigRef<EigMat<T>> u,
    ConstEigRef<EigColVec<T>> s,
    ConstEigRef<EigMat<T>> v
    )
{
    if (u.cols() != v.cols())
        throw std::invalid_argument("LowRankMatrix::set_usv(): Incompatible `u` and `v` sizes.");
    if (u.cols() != s.size())
        throw std::invalid_argument("LowRankMatrix::set_usv(): Incompatible `s` size.");

    u_ = u;
    s_ = s;
    v_ = v;
    return;
};


template <typename T>
void LowRankMatrix<T>::append(
    ConstEigRef<EigMat<T>> u,
    ConstEigRef<EigMat<T>> v
    )
{

    if (u.cols() != v.cols())
        throw std::invalid_argument("LowRankMatrix::append(): Incompatible `u` and `v` sizes.");

    const Index rank = u_.cols();
    const Index extra = u.cols();

    if (rank > 0 && (u.rows() != u_.rows() || v.rows() != v_.rows()))
        throw std::invalid_argument("LowRankMatrix::append(): Incompatible factor sizes.");

    if (extra == 0)
        return;

    u_.conservativeResize(u.rows(), rank + extra);
    v_.conservativeResize(v.rows(), rank + extra);
    s_.conservativeResize(rank + extra);

    u_.rightCols(extra) = u;
    v_.rightCols(extra) = v;
    s_.tail(extra).setOnes();

    return;

};


template <typename T>
void LowRankMatrix<T>::recompress(const Float tol, const Float reference)
{

    const Index rank = u_.cols();

    if (rank == 0)
        return;

    for (Index ii = 0; ii < rank; ++ii)
        u_.col(ii) *= s_[ii];

    Eigen::HouseholderQR<EigMat<T>> qr_u (u_);
    Eigen::HouseholderQR<EigMat<T>> qr_v (v_);

    const Index rows_u = std::min(rank, (Index) u_.rows());
    const Index rows_v = std::min(rank, (Index) v_.rows());

    EigMat<T> r_u = EigMat<T>::Zero(rows_u, rank);
    r_u.template triangularView<Eigen::Upper>() = qr_u.matrixQR().topRows(rows_u);

    EigMat<T> r_v = EigMat<T>::Zero(rows_v, rank);
    r_v.template triangularView<Eigen::Upper>() = qr_v.matrixQR().topRows(rows_v);

    EigMat<T> core;

    if (rows_u == rank)
        core.noalias() = r_u.template triangularView<Eigen::Upper>() * r_v.transpose();
    else
        core.noalias() = r_u * r_v.transpose();

    Eigen::BDCSVD<EigMat<T>, Eigen::ComputeThinU | Eigen::ComputeThinV> svd (core);

    Float norm = svd.singularValues().norm();
    if (reference > 0)
        norm = reference;

    const Float threshold = tol * norm;

    Index new_rank = svd.singularValues().size();

    for (Index ii = 0; ii < svd.singularValues().size(); ++ii)
    {
        if (svd.singularValues()[ii] < threshold || svd.singularValues()[ii] == 0)
        {
            new_rank = ii;
            break;
        }
    }

    const Index u_rows = u_.rows();
    const Index v_rows = v_.rows();

    u_.setZero(u_rows, new_rank);
    u_.topRows(rows_u) = svd.matrixU().leftCols(new_rank);
    u_.applyOnTheLeft(qr_u.householderQ());

    v_.setZero(v_rows, new_rank);
    v_.topRows(rows_v) = svd.matrixV().leftCols(new_rank).conjugate();
    v_.applyOnTheLeft(qr_v.householderQ());

    s_ = svd.singularValues().head(new_rank).template cast<T> ();

    return;

};


template <typename T>
void LowRankMatrix<T>::matmul_low_rank(
    LowRankMatrix<T>& x,
    const LowRankMatrix<T>& y,
    const T& a
    ) const
{

    if (u_.cols() == 0 || y.u_.cols() == 0)
    {
        x.set_usv(
            EigMat<T>::Zero(u_.rows(), 0),
            EigColVec<T>::Zero(0),
            EigMat<T>::Zero(y.v_.rows(), 0)
            );
        return;
    }

    const EigMat<T> core = a * (
        s_.asDiagonal() * (v_.transpose() * y.u_) * y.s_.asDiagonal()
        );

    if (core.rows() <= core.cols())
        x.set_usv(u_, EigColVec<T>::Ones(core.rows()), y.v_ * core.transpose());
    else
        x.set_usv(u_ * core, EigColVec<T>::Ones(core.cols()), y.v_);

    return;

};


template <typename T>
void LowRankMatrix<T>::resize(Index rows, Index cols)
{
    u_.resize(rows, 0);
    s_.resize(0);
    v_.resize(cols, 0);
    return;
};


template <typename T>
void LowRankMatrix<T>::clear()
{
    resize(0, 0);
    return;
};


template <typename T>
void LowRankMatrix<T>::assemble(DenseMatrixType& mat) const
{
    mat.raw_matrix() = u_ * s_.asDiagonal() * v_.transpose();
    return;
};


template <typename T>
T LowRankMatrix<T>::value(Index row, Index col) const
{
    return (u_.row(row) * s_.asDiagonal() * v_.row(col).transpose())[0];
};


template <typename T>
void LowRankMatrix<T>::set_zero()
{
    u_.setZero();
    s_.setZero();
    v_.setZero();
    return;
};


template <typename T>
void LowRankMatrix<T>::set_block(
    const MatrixBase<T>& x,
    Index row_start,
    Index col_start,
    const T& a
    )
{

    if (row_start != 0 || col_start != 0 ||
        x.num_rows() != num_rows() || x.num_cols() != num_cols())
        throw std::invalid_argument(
            "LowRankMatrix::set_block(): only a block spanning this whole matrix can be set."
            );

    EigMat<T> dense;

    EigenMatrix<T>::dispatch(x, [&] (const auto& xr)
    { dense = EigenMatrix<T>::template as<EigMat<T>> (xr); });

    set_usv(
        a * dense,
        EigColVec<T>::Ones(num_cols()),
        EigMat<T>::Identity(num_cols(), num_cols())
        );

    recompress(float_eps);

    return;

};


template <typename T>
void LowRankMatrix<T>::set_transpose(const MatrixBase<T>& x)
{
    const LowRankMatrix<T>* xlr = dynamic_cast<const LowRankMatrix<T>*> (&x);

    if (xlr == nullptr)
        throw std::invalid_argument("LowRankMatrix::set_transpose(): `x` must be a `LowRankMatrix`.");

    u_ = xlr->v();
    s_ = xlr->s();
    v_ = xlr->u();
    return;
};


template <typename T>
void LowRankMatrix<T>::get_diagonal(MatrixBase<T>& x) const
{
    if (num_rows() * num_cols() == 0)
        return;

    EigenMatrix<T>::dispatch(x, [&] (auto& xr)
    {
        xr = (
            (u_ * s_.asDiagonal()).array() * v_.array()
            ).rowwise().sum().matrix().asDiagonal();
    });

    return;
};


template <typename T>
void LowRankMatrix<T>::matmul(
    MatrixBase<T>& x,
    const MatrixBase<T>& y,
    const T& a,
    const bool accumulate
    ) const
{
    if (y.num_rows() * y.num_cols() == 0)
        return;

    EigenMatrix<T>::dispatch(x, y, [&] (auto& xr, const auto& yr)
    {
        if (accumulate)
            xr += EigenMatrix<T>::template as<std::decay_t<decltype(xr)>> (
                (u_ * (s_.asDiagonal() * (v_.transpose() * yr * a))).eval()
                );
        else
            xr = EigenMatrix<T>::template as<std::decay_t<decltype(xr)>> (
                (u_ * (s_.asDiagonal() * (v_.transpose() * yr * a))).eval()
                );
    });

    return;
};


template <typename T>
void LowRankMatrix<T>::matmul_block(
    MatrixBase<T>& x,
    const MatrixBase<T>& y,
    const Index x_row_start,
    const Index y_row_start,
    const T& a,
    const bool accumulate
    ) const
{
    if (y.num_rows() * y.num_cols() == 0)
        return;

    if (y.num_rows() < num_cols())
        throw std::invalid_argument("LowRankMatrix::matmul_block(): `y` must have at least as many rows as `this` matrix has columns.");

    if (x.num_rows() == 0 && x.num_cols() == 0)
    {
        x.resize(x_row_start + num_rows(), y.num_cols());
    }
    else if (x.num_rows() < x_row_start + num_rows() || x.num_cols() < y.num_cols())
    {
        throw std::invalid_argument("LowRankMatrix::matmul_block(): `x` must be either unallocated or already large enough to hold the destination block.");
    }

    EigenMatrix<T>::dispatch(x, y, [&] (auto& xr, const auto& yr)
    {
        if constexpr (std::is_same_v<std::decay_t<decltype(xr)>, Eigen::SparseMatrix<T>>)
        {
            SparseMatrixType temp;
            temp.raw_matrix() = EigenMatrix<T>::template as<std::decay_t<decltype(xr)>> ((
                u_ * (s_.asDiagonal() * (v_.transpose() * yr.block(y_row_start, 0, num_cols(), y.num_cols()) * a))
                ).eval());

            if (accumulate)
                x.add_block(temp, x_row_start, 0);
            else
                x.set_block(temp, x_row_start, 0);
        }
        else
        {
            if (accumulate)
                xr.block(x_row_start, 0, num_rows(), y.num_cols()) +=
                    EigenMatrix<T>::template as<std::decay_t<decltype(xr)>> ((
                        u_ * (s_.asDiagonal() * (v_.transpose() * yr.block(y_row_start, 0, num_cols(), y.num_cols()) * a))
                        ).eval());
            else
                xr.block(x_row_start, 0, num_rows(), y.num_cols()) =
                    EigenMatrix<T>::template as<std::decay_t<decltype(xr)>> ((
                        u_ * (s_.asDiagonal() * (v_.transpose() * yr.block(y_row_start, 0, num_cols(), y.num_cols()) * a))
                        ).eval());
        }
    });

    return;
};


template <typename T>
void LowRankMatrix<T>::matmul_triple(
    MatrixBase<T>& x,
    const MatrixBase<T>& left,
    const MatrixBase<T>& right
    ) const
{

    LowRankMatrix<T>* xlr = dynamic_cast<LowRankMatrix<T>*> (&x);

    if (xlr == nullptr)
        throw std::invalid_argument("LowRankMatrix::matmul_triple(): `x` must be a `LowRankMatrix`.");

    EigMat<T> u_new = u_, v_new = v_;

    if (left.size() > 0)
        EigenMatrix<T>::dispatch(left, [&] (const auto& l) { u_new = l * u_; });

    if (right.size() > 0)
        EigenMatrix<T>::dispatch(right, [&] (const auto& r) { v_new = r.transpose() * v_; });

    xlr->set_usv(u_new, s_, v_new);

    return;
};


template <typename T>
void LowRankMatrix<T>::get_block(
    MatrixBase<T>& x,
    Index row_start,
    Index col_start,
    Index b_rows,
    Index b_cols
    ) const
{
    EigenMatrix<T>::dispatch(x, [&] (auto& xr)
    {
        xr = EigenMatrix<T>::template as<std::decay_t<decltype(xr)>> ((
            u_.block(row_start, 0, b_rows, u_.cols()) *
            s_.asDiagonal() *
            v_.block(col_start, 0, b_cols, v_.cols()).transpose()
            ).eval());
    });
    return;
};


template <typename T>
Float LowRankMatrix<T>::cond() const
{
    DenseMatrixType mat;
    assemble(mat);
    return mat.cond();
};


template <typename T>
void LowRankMatrix<T>::print(const std::string name) const
{
    DenseMatrixType mat;
    assemble(mat);
    mat.print(name);
    return;
};


template class LowRankMatrix<Complex>;

}
