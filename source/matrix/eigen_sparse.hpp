// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Eigen-based sparse matrix wrapper.
*/

#ifndef EIGEN_SPARSE_MATRIX_HPP
#define EIGEN_SPARSE_MATRIX_HPP

#include <vector>
#include <numeric>
#include <stdexcept>

#include <external/Eigen/Sparse>
#include <external/Eigen/SparseLU>
#include <external/Eigen/IterativeLinearSolvers>
#include <external/Eigen/SVD>
#include <external/EigenUnsupported/Eigen/IterativeSolvers>

#include "types.hpp"
#include "matrix/base.hpp"
#include "matrix/eigen_base.hpp"
#include "matrix/eigen_dense.hpp"


namespace bem
{

/**
* \addtogroup eigmat
* @{
*/

/**
* @brief Eigen-based sparse matrix wrapper.
* @tparam T - Data type to be stored in the matrix (e.g., float, double, std::complex).
* @tparam StorageOrder - Storage order of the matrix, either `Eigen::ColMajor` or `Eigen::RowMajor`.
*/
template <typename T, int StorageOrder = Eigen::ColMajor>
class EigenSparseMatrix: public EigenMatrixBase<T, Eigen::SparseMatrix<T, StorageOrder>>
{

    using MatrixType = Eigen::SparseMatrix<T, StorageOrder>;
    using base = EigenMatrixBase<T, MatrixType>;
    using base::base;
    using base::matrix_;

public:

    using base::raw_matrix;

    /**
    * @brief Preallocates memory for a given number of non-zero values per row.
    * @param[in] nnz - Vector containing the number of non-zero values for each row.
    * @note This preallocates a cache for storing matrix values and their indices, not the actual matrix.
    * One must call `assemble()` to actually assemble the matrix from the cached values, before using the matrix.
    */
    void preallocate(const std::vector<Index>& nnz) override
    {
        triplets_.reserve(std::accumulate(nnz.begin(), nnz.end(), 0));
        assembled_ = false;
        return;
    };


    /**
    * @brief Preallocates memory for a given total number of non-zero values.
    * @param[in] num_entries - Total number of non-zero values.
    * @note This preallocates a cache for storing matrix values and their indices, not the actual matrix.
    * One must call `assemble()` to actually assemble the matrix from the cached values, before using the matrix.
    */
    void preallocate(Index num_entries) override
    {
        triplets_.reserve(num_entries);
        assembled_ = false;
        return;
    };


    /**
    * @brief Preallocates memory for a given number of non-zero values per row.
    * @param[in] nnz - Vector containing the number of non-zero values for each row.
    * @note This preallocates memory directly for the underlying raw matrix values, but `assemble()`
    * must still be called before using the matrix.
    */
    void preallocate_directly(const std::vector<Index>& nnz)
    {
        matrix_->reserve(nnz);
        assembled_ = false;
        return;
    };


    /**
    * @brief Assembles cached data into the matrix.
    */
    void assemble() override
    {
        if (assembled_)
            return;

        triplets_.shrink_to_fit();
        if (insert_mode_on_)
            matrix_->setFromTriplets(
                triplets_.begin(),
                triplets_.end(),
                [] (const T&, const T &b) { return b; }
                );
        else
            matrix_->setFromTriplets(triplets_.begin(), triplets_.end());
        triplets_.clear();
        assembled_ = true;
        return;
    };


    /**
    * @brief Returns the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @return Value at the specified position in the matrix.
    */
    T value(Index row, Index col) const override
    {
        return matrix_->coeff(row, col);
    };


    /**
    * @brief Sets the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to set.
    * @note This sets the value in a cache, not directly in the underlying raw matrix. One must call
    * `assemble()` to actually assemble the matrix from all cached values. It is recommended to call
    * `set_value()` for all values one wants to set, and then call `assemble()` once to create the matrix.
    */
    void set_value(Index row, Index col, const T& a) override
    {
        triplets_.push_back(Eigen::Triplet<T> (row, col, a));
        insert_mode_on_ = true;
        assembled_ = false;
        return;
    };


    /**
    * @brief Sets the matrix value at the specified row and column directly, without using a cache.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to set.
    */
    void set_value_directly(Index row, Index col, const T& a)
    {
        matrix_->coeffRef(row, col) = a;
        matrix_->makeCompressed();
        return;
    };


    /**
    * @brief Adds to the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to add.
    * @note This adds the value to a cache, not directly in the underlying raw matrix. One must call
    * `assemble()` to actually assemble the matrix from all cached values. It is recommended to call
    * `set_value()` or `add_value()` for all values one wants to set or add, and then call `assemble()`
    * once to create the matrix.
    */
    void add_value(Index row, Index col, const T& a) override
    {
        triplets_.push_back(Eigen::Triplet<T> (row, col, a));
        insert_mode_on_ = false;
        assembled_ = false;
        return;
    };


    /**
    * @brief Adds to the matrix value at the specified row and column directly, without using a cache.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to add.
    */
    void add_value_directly(Index row, Index col, const T& a)
    {
        matrix_->coeffRef(row, col) += a;
        return;
    };


    /**
    * @brief Scales all matrix entries by a given value.
    * @param[in] a - Value by which to scale.
    */
    void scale(const T& a) override
    {
        for (Index kk = 0; kk < matrix_->outerSize(); ++kk)
            for (typename MatrixType::InnerIterator it((*matrix_), kk); it; ++it)
                it.valueRef() = a * it.value();
        matrix_->makeCompressed();
        return;
    };


    /**
    * @brief Computes and stores the LU factors using sparse factorization. The original matrix is not modified.
    */
    void factorize()
    {
        solver_.compute((*matrix_));
        if (solver_.info() != Eigen::Success)
            throw std::runtime_error("Matrix decomposition failed.");
        factorized_ = true;
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
        if (!factorized_)
            factorize();

        const EigenDenseMatrix<T>& bd = dynamic_cast<const EigenDenseMatrix<T>&> (b);
        EigenDenseMatrix<T>& xd = dynamic_cast<EigenDenseMatrix<T>&> (x);

        xd.raw_matrix() = solver_.solve(bd.raw_matrix());

        if (solver_.info() != Eigen::Success)
            throw std::runtime_error("Matrix solve failed.");

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
        const EigenMatrixBase<T, MatrixType>& x,
        Index row_start,
        Index col_start,
        const T& a = T(1)
        ) override
    {
        std::vector<Eigen::Triplet<T>> x_triplets;
        x_triplets.reserve(x.raw_matrix().nonZeros());

        for (Index kk = 0; kk < x.raw_matrix().outerSize(); ++kk)
            for (typename MatrixType::InnerIterator it (x.raw_matrix(), kk); it; ++it)
                x_triplets.push_back(Eigen::Triplet<T> (row_start + it.row(), col_start + it.col(), it.value() * a));

        matrix_->insertFromTriplets(
            x_triplets.begin(),
            x_triplets.end(),
            [] (const T&, const T &b) { return b; }
            );

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
        const EigenMatrixBase<T, MatrixType>& x,
        Index row_start,
        Index col_start,
        const T& a = T(1)
        ) override
    {
        std::vector<Eigen::Triplet<T>> x_triplets;
        x_triplets.reserve(x.raw_matrix().nonZeros());

        for (Index kk = 0; kk < x.raw_matrix().outerSize(); ++kk)
            for (typename MatrixType::InnerIterator it (x.raw_matrix(), kk); it; ++it)
                x_triplets.push_back(Eigen::Triplet<T> (row_start + it.row(), col_start + it.col(), it.value() * a));

        matrix_->insertFromTriplets(x_triplets.begin(), x_triplets.end());

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
        EigenMatrixBase<T, MatrixType>& x,
        Index row_start,
        Index col_start,
        Index b_rows,
        Index b_cols
        ) const override
    {
        x.resize(b_rows, b_cols);
        x.raw_matrix() = matrix_->block(row_start, col_start, b_rows, b_cols);
        return;
    };


    /**
    * @brief Returns a read-only pointer to the underlying raw data.
    * @return Read-only pointer to the raw data.
    */
    virtual const T* data() const override
    { return matrix_->valuePtr(); };


    /**
    * @brief Returns a writable pointer to the underlying raw data - use with care.
    * @return Writable pointer to the raw data.
    */
    virtual T* data() override
    { return matrix_->valuePtr(); };


private:

    std::vector<Eigen::Triplet<T>> triplets_;
    bool insert_mode_on_ = true;
    bool assembled_ = false;

    Eigen::SparseLU<MatrixType> solver_;
    bool factorized_ = false;

};

/**
* @}
*/

}

#endif
