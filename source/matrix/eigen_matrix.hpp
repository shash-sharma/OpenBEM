// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class wrapping dense and sparse Eigen matrices.
*/

#ifndef EIGEN_MATRIX_HPP
#define EIGEN_MATRIX_HPP

#include <vector>
#include <memory>
#include <type_traits>
#include <variant>
#include <functional>
#include <stdexcept>

#include <external/Eigen/Dense>
#include <external/Eigen/Sparse>
#include <external/Eigen/SparseLU>
#include <external/Eigen/IterativeLinearSolvers>
#include <external/Eigen/SVD>
#include <external/EigenUnsupported/Eigen/IterativeSolvers>

#include "types.hpp"
#include "constants.hpp"
#include "matrix/base.hpp"


namespace bem
{

/**
* \defgroup eigmat Eigen Matrix Wrapper
* \ingroup matr
* @{
*/

/**
* @brief Enumeration of wrapped Eigen matrix types.
*/
enum class EigenMatrixType
{
    EIGEN_DENSE,
    EIGEN_SPARSE
};


/**
* @brief Enumeration of supported iterative solvers.
*/
enum class EigenIterativeSolverType
{
    EIGEN_GMRES,
    EIGEN_BICGSTAB
};


/**
* @brief Summary of an iterative solve.
*/
struct IterativeSolveInfo
{
    Eigen::ComputationInfo status = Eigen::Success;
    Index iterations = 0;
    Index max_iterations_allowed = 0;
    Float rtol_requested = 0;
    Float rtol_achieved = 0;
};


/**
* @brief Helper Eigen matrix type for custom matrix-vector products in iterative solvers.
*/
template <typename MatrixType> class MatmulMatrix;


/**
* @brief Class wrapping dense and sparse Eigen matrices.
* @tparam T - Data type to be stored in the matrix (e.g., float, double, std::complex).
* @tparam type - Dense (default) or sparse matrix.
* @tparam storage_order - Column (default) or row major ordering.
*/
template <
    typename T = Complex,
    EigenMatrixType type = EigenMatrixType::EIGEN_DENSE,
    int storage_order = Eigen::ColMajor
    >
class EigenMatrix: public MatrixBase<T>
{

    using DenseMatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, storage_order>;
    using SparseMatrixType = Eigen::SparseMatrix<T, storage_order>;

    using MatrixType = typename std::conditional<
        type == EigenMatrixType::EIGEN_DENSE,
        DenseMatrixType,
        SparseMatrixType
        >::type;

public:

    /**
    * @brief Constructs an `EigenMatrix` object with a specified number of rows and columns.
    * @param[in] rows - Number of rows (optional).
    * @param[in] cols - Number of columns (optional).
    */
    EigenMatrix(Index rows = 0, Index cols = 0)
    {
        resize(rows, cols);
        set_zero();
        return;
    };


    /**
    * @brief Returns the Eigen type name of the matrix.
    * @return - Type name.
    */
    static EigenMatrixType matrix_type() { return type; };


    /**
    * @brief Returns a writable reference to the underlying raw matrix - use with caution.
    */
    MatrixType& raw_matrix()
    {
        return matrix_;
    };


    /**
    * @brief Returns a read-only reference to the underlying raw matrix.
    */
    const MatrixType& raw_matrix() const
    {
        return matrix_;
    };


    /**
    * @brief Returns a read-only pointer to the underlying raw data.
    * @return Read-only pointer to the raw data.
    */
    const T* data() const override
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            return matrix_.data();

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
            return matrix_.valuePtr();
    };


    /**
    * @brief Returns a writable pointer to the underlying raw data.
    * @return Writable pointer to the raw data.
    */
    T* data() override
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            return matrix_.data();

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
            return matrix_.valuePtr();
    };


    /**
    * @brief Returns a unique pointer to a newly constructed object of the derived type.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<MatrixBase<T>> clone() const override
    {
        return std::make_unique<EigenMatrix<T, type, storage_order>> (
            EigenMatrix<T, type, storage_order> ()
            );
    };


    /**
    * @brief Returns the total number of rows in the matrix.
    * @return Number of rows.
    */
    Index num_rows() const override
    {
        return matrix_.rows();
    };


    /**
    * @brief Returns the total number of columns in the matrix.
    * @return Number of columns.
    */
    Index num_cols() const override
    {
        return matrix_.cols();
    };


    /**
    * @brief Returns the size (rows * cols) of the matrix.
    * @return Size (rows * cols).
    */
    Index size() const override
    {
        return matrix_.size();
    };


    /**
    * @brief Resizes the matrix and sets it to zeros.
    * @param[in] rows - New number of rows.
    * @param[in] cols - New number of columns.
    */
    void resize(Index rows, Index cols) override
    {
        matrix_.resize(rows, cols);
        set_zero();
        return;
    };


    /**
    * @brief Clears all data in the matrix and sets its size to 0.
    */
    void clear() override
    {
        resize(0, 0);
        triplets_.clear();
        assembled_ = false;
        factorized_ = false;
        return;
    };


    /**
    * @brief Preallocates memory for a given number of non-zero values per row.
    * @param[in] nnz - Vector containing the number of non-zero values for each row.
    * @note This preallocates a cache for storing matrix values and their indices, not the actual matrix.
    * One must call `assemble()` to actually assemble the matrix from the cached values, before using the matrix.
    */
    void preallocate(const std::vector<Index>& nnz) override
    {
        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            triplets_.reserve(std::accumulate(nnz.begin(), nnz.end(), 0));
            assembled_ = false;
        }
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
        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            triplets_.reserve(num_entries);
            assembled_ = false;
        }
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
        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            matrix_.reserve(nnz);
            assembled_ = false;
        }
        return;
    };


    /**
    * @brief Assembles cached data into the matrix.
    */
    void assemble() override
    {
        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            if (assembled_)
                return;

            triplets_.shrink_to_fit();
            if (insert_mode_on_)
                matrix_.setFromTriplets(
                    triplets_.begin(),
                    triplets_.end(),
                    [] (const T&, const T &b) { return b; }
                    );
            else
                matrix_.setFromTriplets(triplets_.begin(), triplets_.end());
            triplets_.clear();
            assembled_ = true;
        }

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
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            return matrix_(row, col);

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
            return matrix_.coeff(row, col);
    };


    /**
    * @brief Sets the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to set.
    */
    void set_value(Index row, Index col, const T& a) override
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            matrix_(row, col) = a;

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            triplets_.emplace_back(Eigen::Triplet<T> (row, col, a));
            insert_mode_on_ = true;
            assembled_ = false;
        }

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
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            matrix_(row, col) += a;

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            triplets_.emplace_back(Eigen::Triplet<T> (row, col, a));
            insert_mode_on_ = false;
            assembled_ = false;
        }

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
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            matrix_(row, col) = a;

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            matrix_.coeffRef(row, col) = a;
            matrix_.makeCompressed();
        }

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
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            matrix_(row, col) += a;

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            matrix_.coeffRef(row, col) += a;
        }

        return;
    };


    /**
    * @brief Scales all matrix entries by a given value.
    * @param[in] a - Value by which to scale.
    */
    void scale(const T& a) override
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            matrix_ *= a;

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            for (Index kk = 0; kk < matrix_.outerSize(); ++kk)
                for (typename MatrixType::InnerIterator it(matrix_, kk); it; ++it)
                    it.valueRef() = a * it.value();
            matrix_.makeCompressed();
        }

        return;
    };


    /**
    * @brief Sets all matrix entries to zero.
    */
    void set_zero() override
    {
        matrix_.setZero();
        return;
    };


    /**
    * @brief Sets the matrix to identity (ones along the diagonal).
    */
    void set_identity() override
    {
        matrix_.setIdentity();
        return;
    };


    /**
    * @brief Sets all matrix entries to a given constant value.
    * @param[in] a - Constant value to set.
    */
    void set_constant(const T& a) override
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            matrix_.setConstant(a);

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            for (Index kk = 0; kk < matrix_.outerSize(); ++kk)
                for (typename MatrixType::InnerIterator it(matrix_, kk); it; ++it)
                    it.valueRef() = a;
            matrix_.makeCompressed();
        }

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{X}^T \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ \mathbf{X} \f$ is a given matrix, and \f$ \mathbf{X}^T \f$ is its transpose.
    * @param[in] x - Matrix to transpose.
    */
    void set_transpose(const MatrixBase<T>& x) override
    {
        dispatch(x, [&](auto& xr)
        {
            matrix_ = as<MatrixType> (xr.transpose());
        });

        return;
    };


    /**
    * @brief Retrieves matrix values on the diagonal.
    * @param[out] x - Diagonal matrix.
    */
    void get_diagonal(MatrixBase<T>& x) const override
    {
        dispatch(x, [&](auto& xr)
        {
            xr.resize(num_rows(), num_cols());
            xr.setIdentity();
            xr.diagonal() = matrix_.diagonal();
        });

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{M} + a\mathbf{X} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ is a matrix.
    * @param[in] x - Matrix to scale and add, must have the same dimensions as this matrix.
    * @param[in] a - Scalar with which to scale `x`.
    */
    void add_ax(const MatrixBase<T>& x, const T& a = T(1)) override
    {
        if (x.num_rows() * x.num_cols() == 0)
            return;

        if (!(std::abs(a) > float_eps))
            return;

        dispatch(x, [&](const auto& xr)
        {
            matrix_ += as<MatrixType> (a * xr);
        });
        
        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = a\mathbf{X} + b\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ and \f$ b \f$ are scalars, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to scale and add, must have the same dimensions as `y`.
    * @param[in] y - Second matrix to scale and add, must have the same dimensions as `x`.
    * @param[in] a - Scalar with which to scale `x`.
    * @param[in] b - Scalar with which to scale `y`.
    */
    void set_axpby(
        const EigenMatrix<T, type, storage_order>& x,
        const EigenMatrix<T, type, storage_order>& y,
        const T& a = T(1),
        const T& b = T(1)
        )
    {
        dispatch(x, y, [&](const auto& xr, const auto& yr)
        {
            matrix_ = a * xr + b * yr;
        });

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{X} = a\mathbf{M}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[out] x - Multiplication result.
    * @param[in] y - Matrix with which to multiply, must have the same number of rows as this matrix.
    * @param[in] a - Scalar with which to scale the product.
    * @param[in] accumulate - Whether to accumulate the product into a pre-existing `x`.
    */
    void matmul(
        MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1),
        const bool accumulate = false
        ) const override
    {
        if (y.num_rows() * y.num_cols() == 0)
            return;
        
        if (!(std::abs(a) > float_eps) && accumulate)
            return;

        dispatch(x, y, [&](auto& xr, const auto& yr)
        {
            if (accumulate)
                xr += as<std::decay_t<decltype(xr)>> ((matrix_ * yr * a).eval());
            else
                xr = as<std::decay_t<decltype(xr)>> ((matrix_ * yr * a).eval());
        });

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{X}_b = a\mathbf{M}\mathbf{Y}_b \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices, and the subscript \f$ b \f$
    * indicates that the matrix is multiplied into a sub-block of \f$ \mathbf{Y} \f$, and the result is 
    * placed in a sub-block of \f$ \mathbf{X} \f$.
    * @param[in,out] x - Destination matrix, with at least as many rows as rows of this matrix. If
    * `x` is completely unallocated (0x0), it is resized automatically to fit the destination
    * block; otherwise it must already be at least as large as the destination block requires
    * (an exception is thrown if it is undersized), and the existing block is updated in place.
    * @param[in] y - Matrix whose sub-block to multiply, with at least as many rows as columns of this matrix.
    * @param[in] x_row_start - Starting row index for the destination block.
    * @param[in] y_row_start - Starting row index for the multiplier block.
    * @param[in] a - Scalar with which to scale the product.
    * @param[in] accumulate - Whether to accumulate the product into a pre-existing `x`.
    */
    virtual void matmul_block(
        MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const Index x_row_start,
        const Index y_row_start,
        const T& a = T(1),
        const bool accumulate = false
        ) const override
    {

        if (y.num_rows() * y.num_cols() == 0)
            return;

        if (y.num_rows() < num_cols())
            throw std::invalid_argument("EigenMatrix::matmul_block(): `y` must have at least as many rows as `this` matrix has columns.");

        if (x.num_rows() == 0 && x.num_cols() == 0)
        {
            x.resize(x_row_start + num_rows(), y.num_cols());
        }
        else if (x.num_rows() < x_row_start + num_rows() || x.num_cols() < y.num_cols())
        {
            throw std::invalid_argument("EigenMatrix::matmul_block(): `x` must be either unallocated or already large enough to hold the destination block.");
        }

        if (!(std::abs(a) > float_eps) && accumulate)
            return;

        dispatch(x, y, [&](auto& xr, const auto& yr)
        {
            if constexpr (std::is_same_v<std::decay_t<decltype(xr)>, SparseMatrixType>)
            {
                EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE, storage_order> temp;
                temp.raw_matrix() = as<std::decay_t<decltype(xr)>> ((matrix_ * yr.block(
                    y_row_start, 0, num_cols(), y.num_cols()
                    ) * a).eval());

                if (accumulate)
                    x.add_block(temp, x_row_start, 0);
                else
                    x.set_block(temp, x_row_start, 0);
            }
            else
            {
                if (accumulate)
                    xr.block(
                        x_row_start, 0, num_rows(), y.num_cols()
                        ) += as<std::decay_t<decltype(xr)>> ((matrix_ * yr.block(
                            y_row_start, 0, num_cols(), y.num_cols()
                            ) * a).eval());
                else
                    xr.block(
                        x_row_start, 0, num_rows(), y.num_cols()
                        ) = as<std::decay_t<decltype(xr)>> ((matrix_ * yr.block(
                            y_row_start, 0, num_cols(), y.num_cols()
                            ) * a).eval());
            }
        });

        return;

    };


    /**
    * @brief Sets a block of this matrix to the values of a given matrix, starting at a given position.
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

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            dispatch(x, [&](const auto& xr)
            {
                std::vector<Eigen::Triplet<T>> x_triplets;

                if constexpr (std::is_same_v<std::decay_t<decltype(xr)>, SparseMatrixType>)
                {
                    x_triplets.reserve(xr.nonZeros());

                    for (Index kk = 0; kk < xr.outerSize(); ++kk)
                        for (typename SparseMatrixType::InnerIterator it (xr, kk); it; ++it)
                            x_triplets.emplace_back(
                                Eigen::Triplet<T> (
                                    row_start + it.row(), col_start + it.col(), it.value() * a
                                    )
                                );
                }
                else
                {
                    x_triplets.reserve(xr.rows() * xr.cols());

                    for (Index ii = 0; ii < xr.rows(); ++ii)
                        for (Index jj = 0; jj < xr.cols(); ++jj)
                            x_triplets.emplace_back(
                                Eigen::Triplet<T> (
                                    row_start + ii, col_start + jj, xr(ii, jj) * a
                                    )
                                );
                }
    
                matrix_.insertFromTriplets(
                    x_triplets.begin(),
                    x_triplets.end(),
                    [] (const T&, const T& b) { return b; }
                    );
            });
        }
        else
        {
            dispatch(x, [&](const auto& xr)
            {
                matrix_.block(row_start, col_start, xr.rows(), xr.cols()) = as<MatrixType> (xr * a);
            });
        }

        return;

    };


    /**
    * @brief Adds a block to this matrix from the values of a given matrix, starting at a given position.
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

        if (!(std::abs(a) > float_eps))
            return;

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            dispatch(x, [&](const auto& xr)
            {
                std::vector<Eigen::Triplet<T>> x_triplets;

                if constexpr (std::is_same_v<std::decay_t<decltype(xr)>, SparseMatrixType>)
                {
                    x_triplets.reserve(xr.nonZeros());

                    for (Index kk = 0; kk < xr.outerSize(); ++kk)
                        for (typename SparseMatrixType::InnerIterator it (xr, kk); it; ++it)
                            x_triplets.emplace_back(
                                Eigen::Triplet<T> (
                                    row_start + it.row(), col_start + it.col(), it.value() * a
                                    )
                                );
                }
                else
                {
                    x_triplets.reserve(xr.rows() * xr.cols());

                    for (Index ii = 0; ii < xr.rows(); ++ii)
                        for (Index jj = 0; jj < xr.cols(); ++jj)
                            x_triplets.emplace_back(
                                Eigen::Triplet<T> (
                                    row_start + ii, col_start + jj, xr(ii, jj) * a
                                    )
                                );
                }
    
                matrix_.insertFromTriplets(
                    x_triplets.begin(),
                    x_triplets.end()
                    );
            });
        }
        else
        {
            dispatch(x, [&](const auto& xr)
            {
                matrix_.block(row_start, col_start, xr.rows(), xr.cols()) += as<MatrixType> (xr * a);
            });
        }

        return;

    };


    /**
    * @brief Sets a block of this matrix to a block of a given matrix, starting at a given position.
    * @param[in] x - Matrix to insert.
    * @param[in] src_row_start - Starting row index for the source block.
    * @param[in] src_col_start - Starting column index for the source block.
    * @param[in] dst_row_start - Starting row index for the destination block.
    * @param[in] dst_col_start - Starting column index for the destination block.
    * @param[in] b_rows - Number of rows to retrieve.
    * @param[in] b_cols - Number of columns to retrieve.
    * @param[in] a - Scalar to multiply the values of `x` before inserting (optional).
    */
    virtual void set_block(
        const MatrixBase<T>& x,
        Index src_row_start,
        Index src_col_start,
        Index dst_row_start,
        Index dst_col_start,
        Index b_rows,
        Index b_cols,
        const T& a = T(1)
        ) override
    { 

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            dispatch(x, [&](const auto& xr)
            {
                std::vector<Eigen::Triplet<T>> x_triplets;

                if constexpr (std::is_same_v<std::decay_t<decltype(xr)>, SparseMatrixType>)
                {
                    x_triplets.reserve(b_rows * b_cols);

                    for (Index kk = src_col_start; kk < src_col_start + b_cols; ++kk)
                        for (typename SparseMatrixType::InnerIterator it (xr, kk); it; ++it)
                        {
                            const Index rr = it.row();
                            if (rr >= src_row_start && rr < src_row_start + b_rows)
                            {
                                x_triplets.emplace_back(
                                    Eigen::Triplet<T> (
                                        dst_row_start + (rr - src_row_start),
                                        dst_col_start + (kk - src_col_start),
                                        it.value() * a
                                        )
                                    );
                            }
                        }
                }
                else
                {
                    x_triplets.reserve(b_rows * b_cols);

                    for (Index ii = 0; ii < b_rows; ++ii)
                        for (Index jj = 0; jj < b_cols; ++jj)
                            x_triplets.emplace_back(
                                Eigen::Triplet<T> (
                                    dst_row_start + ii,
                                    dst_col_start + jj,
                                    xr(src_row_start + ii, src_col_start + jj) * a
                                    )
                                );
                }
    
                matrix_.insertFromTriplets(
                    x_triplets.begin(),
                    x_triplets.end(),
                    [] (const T&, const T& b) { return b; }
                    );
            });
        }
        else
        {
            dispatch(x, [&](const auto& xr)
            {
                matrix_.block(
                    dst_row_start,
                    dst_col_start,
                    b_rows,
                    b_cols
                    ) = as<MatrixType> (xr.block(
                        src_row_start,
                        src_col_start,
                        b_rows,
                        b_cols
                        ) * a);
            });
        }

        return;

    };


    /**
    * @brief Adds a block to this matrix from a block of a given matrix, starting at a given position.
    * @param[in] x - Matrix whose block should be added to a block of this matrix.
    * @param[in] src_row_start - Starting row index for the source block.
    * @param[in] src_col_start - Starting column index for the source block.
    * @param[in] dst_row_start - Starting row index for the destination block.
    * @param[in] dst_col_start - Starting column index for the destination block.
    * @param[in] b_rows - Number of rows to retrieve.
    * @param[in] b_cols - Number of columns to retrieve.
    * @param[in] a - Scalar to multiply the values of `x` before inserting (optional).
    */
    virtual void add_block(
        const MatrixBase<T>& x,
        Index src_row_start,
        Index src_col_start,
        Index dst_row_start,
        Index dst_col_start,
        Index b_rows,
        Index b_cols,
        const T& a = T(1)
        ) override
    { 
        
        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            dispatch(x, [&](const auto& xr)
            {
                std::vector<Eigen::Triplet<T>> x_triplets;

                if constexpr (std::is_same_v<std::decay_t<decltype(xr)>, SparseMatrixType>)
                {
                    x_triplets.reserve(b_rows * b_cols);

                    for (Index kk = src_col_start; kk < src_col_start + b_cols; ++kk)
                        for (typename SparseMatrixType::InnerIterator it (xr, kk); it; ++it)
                        {
                            const Index rr = it.row();
                            if (rr >= src_row_start && rr < src_row_start + b_rows)
                            {
                                x_triplets.emplace_back(
                                    Eigen::Triplet<T> (
                                        dst_row_start + (rr - src_row_start),
                                        dst_col_start + (kk - src_col_start),
                                        it.value() * a
                                        )
                                    );
                            }
                        }
                }
                else
                {
                    x_triplets.reserve(b_rows * b_cols);

                    for (Index ii = 0; ii < b_rows; ++ii)
                        for (Index jj = 0; jj < b_cols; ++jj)
                            x_triplets.emplace_back(
                                Eigen::Triplet<T> (
                                    dst_row_start + ii,
                                    dst_col_start + jj,
                                    xr(src_row_start + ii, src_col_start + jj) * a
                                    )
                                );
                }
    
                matrix_.insertFromTriplets(
                    x_triplets.begin(),
                    x_triplets.end()
                    );
            });
        }
        else
        {
            dispatch(x, [&](const auto& xr)
            {
                matrix_.block(
                    dst_row_start,
                    dst_col_start,
                    b_rows,
                    b_cols
                    ) += as<MatrixType> (xr.block(
                        src_row_start,
                        src_col_start,
                        b_rows,
                        b_cols
                        ) * a);
            });
        }

        return;

    };


    /**
    * @brief Scales a block of this matrix.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] b_rows - Number of rows in the block to scale.
    * @param[in] b_cols - Number of columns in the block to scale.
    * @param[in] a - Scaling factor.
    */
    void scale_block(
        Index row_start,
        Index col_start,
        Index b_rows,
        Index b_cols,
        const T& a
        ) override
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            matrix_.block(row_start, col_start, b_rows, b_cols) *= a;
        else
        {
            EigenMatrix<T, type, storage_order> temp;
            get_block(temp, row_start, col_start, b_rows, b_cols);
            set_block(temp, row_start, col_start, a);
        }

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
        dispatch(x, [&](auto& xr)
        {
            xr = as<std::decay_t<decltype(xr)>> (matrix_.block(row_start, col_start, b_rows, b_cols));
        });

        return;
    };


    /**
    * @brief Computes and stores the LU factors. The original matrix is not modified.
    */
    void factorize() override
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
        {
            dense_solver_ = std::make_shared<DenseSolverType> ();
            dense_solver_->compute(matrix_);
            if (dense_solver_->info() != Eigen::Success)
                throw std::runtime_error("EigenMatrix::factorize(): Dense factorization failed.");
            factorized_ = true;
        }

        else if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            sparse_solver_ = std::make_shared<SparseSolverType> ();
            sparse_solver_->compute(matrix_);
            if (sparse_solver_->info() != Eigen::Success)
                throw std::runtime_error("EigenMatrix::factorize(): Sparse factorization failed.");
            factorized_ = true;
        }

        return;
    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with a
    * direct solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    */
    void mat_solve(MatrixBase<T>& x, const MatrixBase<T>& b) const override
    {
        if (!factorized_)
            throw std::runtime_error("EigenMatrix::solve(): Matrix must be factorized first.");

        dispatch(x, b, [&](auto& xr, const auto& br)
        {
            DenseMatrixType rhs = as<DenseMatrixType> (br);
            DenseMatrixType sol;

            if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            {
                sol = dense_solver_->solve(rhs);
                if (dense_solver_->info() != Eigen::Success)
                    throw std::runtime_error("EigenMatrix::mat_solve(): Dense solver failed.");
            }
            else
            {
                sol = sparse_solver_->solve(rhs);
                if (sparse_solver_->info() != Eigen::Success)
                    throw std::runtime_error("EigenMatrix::mat_solve(): Sparse solver failed.");
            }

            xr = as<std::decay_t<decltype(xr)>> (sol);
        });

        return;
    };


    /**
    * @brief Factorizes this matrix via a Schur-complement decomposition, using an analytical
    * inverse for the top-left block if possible, depending on `pattern`.
    * @param[in] pivot - Size of the top-left block.
    * @param[in] pattern - Known structure of the top-left block (optional).
    */
    void factorize_schur(Index pivot, MatrixStructure pattern = MatrixStructure::NONE) override
    {

        const Index n2 = matrix_.rows() - pivot;

        schur_pattern_ = pattern;

        B_schur_ = std::make_shared<EigenMatrix<T, type, storage_order>> ();
        C_schur_ = std::make_shared<EigenMatrix<T, type, storage_order>> ();
        S_schur_ = std::make_shared<EigenMatrix<T, type, storage_order>> ();

        B_schur_->raw_matrix() = matrix_.block(0, pivot, pivot, n2);
        C_schur_->raw_matrix() = matrix_.block(pivot, 0, n2, pivot);

        MatrixType D = matrix_.block(pivot, pivot, n2, n2);

        A_inv_schur_.clear();

        if (pattern == MatrixStructure::DIAGONAL_2X2)
        {
            const Index h = pivot / 2;

            EigMat<T> a1 = MatrixType(matrix_.block(0, 0, h, h)).diagonal();
            EigMat<T> a2 = MatrixType(matrix_.block(0, h, h, h)).diagonal();
            EigMat<T> a3 = MatrixType(matrix_.block(h, 0, h, h)).diagonal();
            EigMat<T> a4 = MatrixType(matrix_.block(h, h, h, h)).diagonal();
            EigMat<T> det = a1.array() * a4.array() - a2.array() * a3.array();

            A_inv_schur_.resize(4);
            A_inv_schur_[0].raw_matrix() = (a4.array() / det.array()).matrix().asDiagonal();
            A_inv_schur_[1].raw_matrix() = (-a2.array() / det.array()).matrix().asDiagonal();
            A_inv_schur_[2].raw_matrix() = (-a3.array() / det.array()).matrix().asDiagonal();
            A_inv_schur_[3].raw_matrix() = (a1.array() / det.array()).matrix().asDiagonal();
        }

        else if (pattern == MatrixStructure::DIAGONAL)
        {
            EigMat<T> a = MatrixType(matrix_.block(0, 0, pivot, pivot)).diagonal();

            A_inv_schur_.resize(1);
            A_inv_schur_[0].raw_matrix() = a.array().inverse().matrix().asDiagonal();
        }

        else
        {
            A_inv_schur_.resize(1);
            A_inv_schur_[0].raw_matrix() = matrix_.block(0, 0, pivot, pivot);
            A_inv_schur_[0].factorize();
        }

        EigenMatrix<T, type, storage_order> AinvB;
        apply_ainv_schur(AinvB, *B_schur_);

        S_schur_->raw_matrix() = D - C_schur_->raw_matrix() * AinvB.raw_matrix();
        S_schur_->factorize();

        schur_factorized_ = true;

        return;

    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ using the factorization computed by
    * `factorize_schur()`.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix.
    */
    void mat_solve_schur(MatrixBase<T>& x, const MatrixBase<T>& b) const override
    {

        if (!schur_factorized_)
            throw std::runtime_error("EigenMatrix::mat_solve_schur(): Matrix must be factorized first.");

        const Index pivot = B_schur_->num_rows();
        const Index n2 = b.num_rows() - pivot;

        EigenMatrix<T, type, storage_order> temp1, temp2;

        b.get_block(temp1, 0, 0, pivot, b.num_cols());

        EigenMatrix<T, type, storage_order> y1;
        apply_ainv_schur(y1, temp1);

        C_schur_->matmul(temp1, y1);

        b.get_block(temp2, pivot, 0, n2, b.num_cols());
        temp2.raw_matrix() -= temp1.raw_matrix();

        EigenMatrix<T, type, storage_order> x2;
        S_schur_->mat_solve(x2, temp2);

        B_schur_->matmul(temp1, x2);
        apply_ainv_schur(temp2, temp1);
        temp2.raw_matrix() = y1.raw_matrix() - temp2.raw_matrix();

        x.resize(pivot + n2, b.num_cols());
        x.set_block(temp2, 0, 0);
        x.set_block(x2, pivot, 0);

        return;

    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with an
    * iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] solver_type - Type of iterative solver to use (optional).
    * @param[in] tol - Tolerance for convergence (optional).
    * @param[in] restart - Restart iteration at which the Krylov subspace is discarded (optional).
    * @return Summary of the solve.
    */
    IterativeSolveInfo mat_solve_iterative(
        MatrixBase<T>& x,
        const MatrixBase<T>& b,
        const EigenIterativeSolverType solver_type = EigenIterativeSolverType::EIGEN_GMRES,
        const Float tol = 1e-4,
        const Index restart = 100
        ) const
    {
        return run_iterative_by_type(raw_matrix(), x, b, solver_type, tol, restart);
    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with an
    * iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix, with a wrapper matrix that defines a custom matrix product.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] mat - Custom matrix product object.
    * @param[in] solver_type - Type of iterative solver to use (optional).
    * @param[in] tol - Tolerance for convergence (optional).
    * @param[in] restart - Restart iteration at which the Krylov subspace is discarded (optional).
    * @return Summary of the solve.
    */
    template <typename MatmulMatrixType>
    static IterativeSolveInfo mat_solve_iterative(
        MatrixBase<T>& x,
        const MatrixBase<T>& b,
        const MatmulMatrixType& mat,
        const EigenIterativeSolverType solver_type = EigenIterativeSolverType::EIGEN_GMRES,
        const Float tol = 1e-4,
        const Index restart = 100
        )
    {
        return run_iterative_by_type(mat, x, b, solver_type, tol, restart);
    };


    /**
    * @brief Returns the Frobenius norm of the matrix, defined as the square root of the sum of the
    * squares of all matrix values.
    * @return Frobenius norm of the matrix.
    */
    T frobenius_norm() const
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            return matrix_.norm();
    };


    /**
    * @brief Returns the one-norm of the matrix, computed by taking the sum of absolute values of all
    * rows for each column, and then taking the largest of these sums across all columns.
    * @return One-norm of the matrix.
    */
    T one_norm() const
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            return matrix_.colwise().sum().array().abs().maxCoeff();
    };


    /**
    * @brief Returns the infinity-norm of the matrix, computed by taking the sum of absolute values of all
    * columns for each row, and then taking the largest of these sums across all rows.
    * @return Infinity-norm of the matrix.
    */
    T infinity_norm() const
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            return matrix_.rowwise().sum().array().abs().maxCoeff();
    };


    /**
    * @brief Computes the rank of the matrix.
    * @return Rank.
    */
    Index rank() const
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            return matrix_.fullPivLu().rank();
    };


    /**
    * @brief Computes the condition number (ratio of largest to smallest singular value) of the matrix.
    * @return Condition number.
    */
    Float cond() const
    {
        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
        {
            // Eigen::JacobiSVD<DenseMatrixType> svd ((DenseMatrixType) matrix_);
            Eigen::BDCSVD<DenseMatrixType> svd ((DenseMatrixType) matrix_);
            EigColVec<Float> s = svd.singularValues();
            return s(0) / s(s.size() - 1);
        }
    };


    /**
    * @brief Prints the matrix to the terminal in a formatted manner.
    */
    void print(const std::string name = "matrix") const override
    {
        std::cout << "\n--- " << name << " (" << num_rows() << " x " << num_cols() << ") ---" << std::endl;
        std::cout << matrix_ << std::endl;
        std::cout << "---" << std::endl;
        return;
    };


    /**
    * @brief Dispatches a callable object to the appropriate concrete matrix type based on the
    * dynamic type of a `MatrixBase` read-only reference.
    * @tparam F - Type of the callable object to apply to the cast matrix.
    * @param[in] x - Matrix reference to cast.
    * @param[in] f - Callable object to apply to the cast matrix.
    * @return Result of applying `f` to the cast matrix.
    */
    template <typename F>
    static decltype(auto) dispatch(const MatrixBase<T>& x, F&& f)
    {
        using Dense = EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>;
        using Sparse = EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE, storage_order>;
        if (auto* xd = dynamic_cast<const Dense*> (&x)) return f(xd->raw_matrix());
        if (auto* xs = dynamic_cast<const Sparse*> (&x)) return f(xs->raw_matrix());
        throw std::invalid_argument("EigenMatrix::dispatch(): incompatible matrix type.");
    };


    /**
    * @brief Dispatches a callable object to the appropriate concrete matrix type based on the
    * dynamic type of a `MatrixBase` writable reference.
    * @tparam F - Type of the callable object to apply to the cast matrix.
    * @param[in] x - Matrix reference to cast.
    * @param[in] f - Callable object to apply to the cast matrix.
    * @return Result of applying `f` to the cast matrix.
    */
    template <typename F>
    static decltype(auto) dispatch(MatrixBase<T>& x, F&& f)
    {
        using Dense = EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>;
        using Sparse = EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE, storage_order>;
        if (auto* xd = dynamic_cast<Dense*> (&x)) return f(xd->raw_matrix());
        if (auto* xs = dynamic_cast<Sparse*> (&x)) return f(xs->raw_matrix());
        throw std::invalid_argument("EigenMatrix::dispatch(): incompatible matrix type.");
    };


    /**
    * @brief Dispatches a callable object to the appropriate concrete matrix types based on the
    * dynamic types of two `MatrixBase` read-only references.
    * @tparam F - Type of the callable object to apply to the cast matrix.
    * @param[in] x - First matrix reference to cast.
    * @param[in] y - Second matrix reference to cast.
    * @param[in] f - Callable object to apply to the cast matrix.
    * @return Result of applying `f` to the cast matrices.
    */
    template <typename F>
    static decltype(auto) dispatch(const MatrixBase<T>& x, const MatrixBase<T>& y, F&& f)
    {
        return dispatch(
            x,
            [&] (const auto& xr)
            { return dispatch(y, [&](const auto& yr) { return f(xr, yr); }); }
            );
    };


    /**
    * @brief Dispatches a callable object to the appropriate concrete matrix types based on the
    * dynamic types of two `MatrixBase` references.
    * @tparam F - Type of the callable object to apply to the cast matrix.
    * @param[in] x - First matrix reference to cast.
    * @param[in] y - Second matrix reference to cast.
    * @param[in] f - Callable object to apply to the cast matrix.
    * @return Result of applying `f` to the cast matrices.
    */
    template <typename F>
    static decltype(auto) dispatch(MatrixBase<T>& x, const MatrixBase<T>& y, F&& f)
    {
        return dispatch(
            x,
            [&] (auto& xr)
            { return dispatch(y, [&](const auto& yr) { return f(xr, yr); }); }
            );
    };


    /**
    * @brief Converts a given Eigen expression to a target matrix type (dense or sparse).
    * @tparam Target - Target matrix type to convert to (DenseMatrixType or SparseMatrixType).
    * @tparam Expr - Type of the Eigen expression to convert.
    * @param[in] expr - Eigen expression to convert.
    * @return Converted matrix of the target type.
    */
    template <typename Target, typename Expr>
    static Target as(const Expr& expr)
    {
        if constexpr (std::is_same_v<Target, DenseMatrixType>)
        {
            return Target(expr); // Eigen assigns dense <- dense or dense <- sparse directly
        }
        else // Target == SparseMatrixType
        {
            using StorageKind = typename Eigen::internal::traits<Expr>::StorageKind;
            if constexpr (std::is_same_v<StorageKind, Eigen::Sparse>)
                return Target(expr); // already sparse
            else
                return Target(DenseMatrixType(expr).sparseView()); // dense -> sparse
        }
    }


protected:

    /**
    * @brief Runs an iterative solver based on the given solver and matrix definitions.
    * @tparam Solver - Type of iterative solver to run.
    * @tparam Op - Type of the matrix or shell operator.
    * @param[in] op - Matrix or shell operator.
    * @param[in] x - Solution.
    * @param[in] b - Right-hand side.
    * @param[in] tol - Relative residual error tolerance (optional).
    * @param[in] restart - GMRES restart iteration (optional).
    * @return Summary of the solve.
    */
    template <typename Solver, typename Op>
    static IterativeSolveInfo run_iterative(
        const Op& op,
        MatrixBase<T>& x,
        const MatrixBase<T>& b,
        const Float tol = 1e-4,
        const Index restart = 100
        )
    {

        const Index max_iterations = 1000;

        Solver solver;
        solver.setTolerance(tol);
        solver.setMaxIterations(max_iterations);
        if constexpr (std::is_same_v<Solver, Eigen::GMRES<Op, Eigen::IdentityPreconditioner>>)
            solver.set_restart(restart);

        solver.compute(op);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("EigenMatrix::mat_solve_iterative(): Solver initialization failed.");

        IterativeSolveInfo info;
        info.max_iterations_allowed = max_iterations;
        info.rtol_requested = tol;

        dispatch(x, b, [&](auto& xr, const auto& br)
        {

            if (br.squaredNorm() < float_eps)
            {
                xr.resize(solver.rows(), br.cols());
                xr.setZero();
                info.iterations = 0;
                info.status = Eigen::Success;
                info.rtol_achieved = 0;
                return;
            }

            DenseMatrixType rhs = as<DenseMatrixType> (br);
            DenseMatrixType sol = solver.solve(rhs);
            xr = as<std::decay_t<decltype(xr)>> (sol);

            info.iterations = solver.iterations();
            info.status = solver.info();
            info.rtol_achieved = solver.error();

        });

        return info;

    };


    /**
    * @brief Runs an iterative solver based on the given solver and matrix definitions.
    * @tparam Op - Type of the matrix or shell operator.
    * @param[in] op - Matrix or shell operator.
    * @param[in] x - Solution.
    * @param[in] b - Right-hand side.
    * @param[in] solver_type - Iterative solver type to use.
    * @param[in] tol - Relative residual error tolerance (optional).
    * @param[in] restart - GMRES restart iteration (optional).
    * @return Summary of the solve.
    */
    template <typename Op>
    static IterativeSolveInfo run_iterative_by_type(
        const Op& op,
        MatrixBase<T>& x,
        const MatrixBase<T>& b,
        const EigenIterativeSolverType solver_type,
        const Float tol = 1e-4,
        const Index restart = 100
        )
    {
        if (solver_type == EigenIterativeSolverType::EIGEN_GMRES)
            return run_iterative<Eigen::GMRES<Op, Eigen::IdentityPreconditioner>> (op, x, b, tol, restart);
        else
            return run_iterative<Eigen::BiCGSTAB<Op, Eigen::IdentityPreconditioner>> (op, x, b, tol, restart);
    };


    /**
    * @brief Applies the analytical (or factorized) inverse of `factorize_schur()`'s top-left block.
    * @param[out] out - Result.
    * @param[in] in - Matrix to which the inverse is applied.
    */
    void apply_ainv_schur(
        EigenMatrix<T, type, storage_order>& out,
        const EigenMatrix<T, type, storage_order>& in
        ) const
    {
        if (schur_pattern_ == MatrixStructure::DIAGONAL_2X2)
        {
            const Index h = A_inv_schur_[0].num_rows();

            out.resize(2 * h, in.num_cols());

            A_inv_schur_[0].matmul_block(out, in, 0, 0, T(1), false);
            A_inv_schur_[1].matmul_block(out, in, 0, h, T(1), true);
            A_inv_schur_[2].matmul_block(out, in, h, 0, T(1), false);
            A_inv_schur_[3].matmul_block(out, in, h, h, T(1), true);
        }
        else if (schur_pattern_ == MatrixStructure::DIAGONAL)
        {
            A_inv_schur_[0].matmul(out, in);
        }
        else
        {
            A_inv_schur_[0].mat_solve(out, in);
        }

        return;
    };


    MatrixType matrix_;

    std::vector<Eigen::Triplet<T>> triplets_;
    bool insert_mode_on_ = true;
    bool assembled_ = false;

    using SparseSolverType = Eigen::SparseLU<SparseMatrixType>;

    using DenseSolverType = Eigen::PartialPivLU<DenseMatrixType>;
    // using DenseSolverType = Eigen::FullPivLU<MatrixType>;
    // using DenseSolverType = Eigen::HouseholderQr<MatrixType>;
    // using DenseSolverType = Eigen::ColPivHouseholderQr<MatrixType>;
    // using DenseSolverType = Eigen::FullPivHouseholderQr<MatrixType>;

    std::shared_ptr<SparseSolverType> sparse_solver_ = nullptr;
    std::shared_ptr<DenseSolverType> dense_solver_ = nullptr;

    bool factorized_ = false;

    bool schur_factorized_ = false;
    MatrixStructure schur_pattern_ = MatrixStructure::NONE;
    std::vector<EigenMatrix<T, type, storage_order>> A_inv_schur_;
    std::shared_ptr<EigenMatrix<T, type, storage_order>> B_schur_ = nullptr;
    std::shared_ptr<EigenMatrix<T, type, storage_order>> C_schur_ = nullptr;
    std::shared_ptr<EigenMatrix<T, type, storage_order>> S_schur_ = nullptr;

};

/**
* @}
*/

}


namespace Eigen
{
namespace internal
{

/**
* @brief Defines sparse matrix traits for the custom matrix multiply shell matrix.
* @tparam MatrixType - Underlying matrix type associated with the shell matrix.
*/
template <typename MatrixType>
struct traits<bem::MatmulMatrix<MatrixType>>:
        public Eigen::internal::traits<Eigen::SparseMatrix<bem::Complex>> {};

}
}

namespace bem
{

/**
* @brief Defines a shell matrix for custom matrix products.
* @tparam MatrixType - Underlying matrix type associated with the shell matrix.
*/
template <typename MatrixType>
class MatmulMatrix: public Eigen::EigenBase<MatmulMatrix<MatrixType>>
{
public:

    // --- Requirements of EigenBase ---

    typedef Complex Scalar;
    typedef Float RealScalar;
    typedef Int StorageIndex;
    enum
    {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return num_rows_; };
    Index cols() const { return num_cols_; };

    /**
    * @brief Wraps the custom matrix product.
    * @tparam Rhs - Type of the matrix with which to multiply.
    * @param[in] x - Matrix with which to multiply.
    * @return Product.
    */
    template <typename Rhs>
    Eigen::Product<MatmulMatrix<MatrixType>, Rhs, Eigen::AliasFreeProduct> operator*(
        const Eigen::MatrixBase<Rhs>& x
        ) const
    {
        return Eigen::Product<
            MatmulMatrix<MatrixType>,
            Rhs,
            Eigen::AliasFreeProduct
            > (*this, x.derived());
    };

    // --- Custom ---

    using Func = std::function<void (MatrixType&, const MatrixType&)>;

    /**
    * @brief Attaches external custom data to the shell object.
    * @param[in] func - Custom function describing the matrix multiplication.
    * @param[in] num_rows - Number of matrix rows.
    * @param[in] num_cols - Number of matrix columns.
    */
    void attach(const Func& func, Index num_rows, Index num_cols)
    {
        func_ = &func;
        num_rows_ = num_rows;
        num_cols_ = num_cols;
        return;
    };

    /**
    * @brief Returns the attached custom matrix multiplication function.
    * @return Custom function describing the matrix multiplication.
    */
    const Func& func() const { return *func_; };


private:

    const Func* func_ = nullptr;
    Index num_rows_ = 0;
    Index num_cols_ = 0;

};

}

namespace Eigen
{
namespace internal
{

/**
* @brief Defines the underlying implementation of custom matrix products.
* @tparam Rhs - Type of the matrix with which to multiply.
* @tparam MatrixType - Type of the shell matrix.
*/
template <typename Rhs, typename MatrixType>
struct generic_product_impl<bem::MatmulMatrix<MatrixType>, Rhs, SparseShape, DenseShape, GemvProduct>:
    generic_product_impl_base<
        bem::MatmulMatrix<MatrixType>,
        Rhs,
        generic_product_impl<bem::MatmulMatrix<MatrixType>, Rhs>
        >
{

    typedef typename Product<bem::MatmulMatrix<MatrixType>, Rhs>::Scalar Scalar;

    /**
    * @brief Defines the custom matrix product implementation.
    * @param[out] dst - Destination to place the computed matrix product.
    * @param[in] lhs - Custom matrix to multiply.
    * @param[in] rhs - Matrix with which to multiply.
    * @param[in] alpha - Scaling (unused).
    */
    template <typename Dest>
    static void scaleAndAddTo(
        Dest& dst,
        const bem::MatmulMatrix<MatrixType>& lhs,
        const Rhs& rhs,
        const Scalar& alpha
        )
    {
        eigen_assert(alpha == Scalar(1) && "MatmulMatrix::Product: scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);

        MatrixType rhs_mat;
        rhs_mat.raw_matrix() = std::move(rhs);

        MatrixType dst_mat;
        lhs.func()(dst_mat, rhs_mat);

        dst = std::move(dst_mat.raw_matrix());
        return;
    };
};

}
}


#endif

