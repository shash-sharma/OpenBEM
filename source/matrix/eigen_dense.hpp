// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Eigen-based dense matrix wrapper.
*/

#ifndef EIGEN_DENSE_MATRIX_HPP
#define EIGEN_DENSE_MATRIX_HPP

#include <external/Eigen/Dense>
#include <external/Eigen/IterativeLinearSolvers>
#include <external/EigenUnsupported/Eigen/IterativeSolvers>

#include "types.hpp"
#include "matrix/base.hpp"
#include "matrix/eigen_base.hpp"


namespace bem
{

/**
* \addtogroup eigmat
* @{
*/

/**
* @brief Eigen-based dense matrix wrapper.
* @tparam T - Data type to be stored in the matrix (e.g., float, double, std::complex).
* @tparam StorageOrder - Storage order of the matrix, either `Eigen::ColMajor` or `Eigen::RowMajor`.
*/
template <typename T, int StorageOrder = Eigen::ColMajor>
class EigenDenseMatrix: public EigenMatrixBase<T, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>>
{

    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;
    using base = EigenMatrixBase<T, MatrixType>;
    using base::base;
    using base::matrix_;

public:


    /**
    * @brief Returns the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @return Value at the specified position in the matrix.
    */
    T value(Index row, Index col) const override
    {
        return (*matrix_)(row, col);
    };


    /**
    * @brief Sets the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to set.
    */
    void set_value(Index row, Index col, const T& a) override
    {
        (*matrix_)(row, col) = a;
        return;
    };


    /**
    * @brief Adds to the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to add.
    */
    void add_value(Index row, Index col, const T& a) override
    {
        (*matrix_)(row, col) += a;
        return;
    };


    /**
    * @brief Scales all matrix entries by a given value.
    * @param[in] a - Value by which to scale.
    */
    void scale(const T& a) override
    {
        (*matrix_) *= a;
        return;
    };


    /**
    * @brief Sets all matrix entries to a given constant value.
    * @param[in] a - Constant value to set.
    */
    void set_constant(const T& a)
    {
        matrix_->setConstant(a);
        return;
    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with a
    * direct solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    */
    void mat_solve(MatrixBase<T>& x, const MatrixBase<T>& b) override
    {
        const EigenDenseMatrix<T>& bd = dynamic_cast<const EigenDenseMatrix<T>&> (b);
        EigenDenseMatrix<T>& xd = dynamic_cast<EigenDenseMatrix<T>&> (x);
        // xd.raw_matrix() = matrix_->householderQr().solve(bd.raw_matrix());
        // xd.raw_matrix() = matrix_->colPivHouseholderQr().solve(bd.raw_matrix());
        // xd.raw_matrix() = matrix_->fullPivHouseholderQr().solve(bd.raw_matrix());
        xd.raw_matrix() = matrix_->partialPivLu().solve(bd.raw_matrix());
        // xd.raw_matrix() = matrix_->fullPivLu().solve(bd.raw_matrix());
        return;
    };


    /**
    * @brief Sets a block of values in this matrix to the values of a given matrix, starting at a given position.
    * @param[in] x - Matrix to insert.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] a - Scalar to multiply the values of `x` before inserting (optional).
    */
    void set_block(
        const MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        const T& a = T(1)
        ) override
    {
        const EigenDenseMatrix<T>& xd = dynamic_cast<const EigenDenseMatrix<T>&> (x);
        matrix_->block(row_start, col_start, xd.num_rows(), xd.num_cols()) = xd.raw_matrix() * a;
        return;
    };


    /**
    * @brief Adds a block of values to this matrix from the values of a given matrix,
    * starting at a given position.
    * @param[in] x - Matrix whose values should be added to a block of this matrix.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] a - Scalar to multiply the values of `x` before adding (optional).
    */
    void add_block(
        const MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        const T& a = T(1)
        ) override
    {
        const EigenDenseMatrix<T>& xd = dynamic_cast<const EigenDenseMatrix<T>&> (x);
        matrix_->block(row_start, col_start, xd.num_rows(), xd.num_cols()) += a * xd.raw_matrix();
        return;
    };


    /**
    * @brief Retrieves a block of values from this matrix.
    * @param[out] x - Matrix to store the retrieved block of values.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] b_rows - Number of rows in the block to retrieve.
    * @param[in] b_cols - Number of columns in the block to retrieve.
    */
    void get_block(
        MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        Index b_rows,
        Index b_cols
        ) const override
    {
        EigenDenseMatrix<T>& xd = dynamic_cast<EigenDenseMatrix<T>&> (x);
        xd.resize(b_rows, b_cols);
        xd.raw_matrix() = matrix_->block(row_start, col_start, b_rows, b_cols);
        return;
    };


    /**
    * @brief Returns a read-only pointer to the underlying raw data.
    * @return Read-only pointer to the raw data.
    */
    virtual const T* data() const override
    { return matrix_->data(); };


    /**
    * @brief Returns a writable pointer to the underlying raw data - use with care.
    * @return Writable pointer to the raw data.
    */
    virtual T* data() override
    { return matrix_->data(); };


    /**
    * @brief Returns the Frobenius norm of the matrix, defined as the square root of the sum of the
    * squares of all matrix values.
    * @return Frobenius norm of the matrix.
    */
    T frobenius_norm() const
    {
        return matrix_->norm();
    };


    /**
    * @brief Returns the one-norm of the matrix, computed by taking the sum of absolute values of all
    * rows for each column, and then taking the largest of these sums across all columns.
    * @return One-norm of the matrix.
    */
    T one_norm() const
    {
        return matrix_->colwise().sum().array().abs().maxCoeff();
    };


    /**
    * @brief Returns the infinity-norm of the matrix, computed by taking the sum of absolute values of all
    * columns for each row, and then taking the largest of these sums across all rows.
    * @return Infinity-norm of the matrix.
    */
    T infinity_norm() const
    {
        return matrix_->rowwise().sum().array().abs().maxCoeff();
    };


    /**
    * @brief Computes the rank of the matrix.
    * @return Rank.
    */
    Index rank() const
    {
        return matrix_->fullPivLu().rank();
    };

};

/**
* @}
*/

}

#endif
