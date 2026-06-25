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
    * @brief Resizes the matrix to a new number of rows and columns.
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
    void set_constant(const T& a)
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
    void set_transpose(const EigenMatrix<T, type, storage_order>& x)
    {
        matrix_ = x.raw_matrix().transpose();
        return;
    };


    /**
    * @brief Retrieves matrix values on the diagonal.
    * @param[out] x - Diagonal matrix.
    */
    void get_diagonal(MatrixBase<T>& x) const override
    {
        auto visitor = [&] (auto* xc)
        {
            xc->raw_matrix().resize(num_rows(), num_cols());
            xc->raw_matrix().setIdentity();
            xc->raw_matrix().diagonal() = matrix_.diagonal();
        };

        std::visit(visitor, to_variant(x));

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

        auto visitor = [this, a] (const auto* xc)
        { matrix_ += a * xc->raw_matrix(); };

        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            std::visit(visitor, to_variant(x));
        else
            matrix_ += a * to_same(x)->raw_matrix();

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
        auto visitor = [this, a, b] (const auto* xc, const auto* yc)
        { matrix_ = a * xc->raw_matrix() + b * yc->raw_matrix(); };

        std::visit(visitor, to_variant(x), to_variant(y));

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = a\mathbf{X}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to multiply, must have the same number of columns as `y` has rows.
    * @param[in] y - Second matrix to multiply , must have the same number of rows as `x` has columns.
    * @param[in] a - Scalar with which to scale the product of `x` and `y`.
    */
    void set_matmul(
        const MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        ) override
    {
        if (x.num_rows() * x.num_cols() * y.num_rows() * y.num_cols() == 0)
            return;

        auto visitor = [this, a] (const auto* xc, const auto* yc)
        { matrix_ = xc->raw_matrix() * yc->raw_matrix() * a; };

        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            std::visit(visitor, to_variant(x), to_variant(y));
        else
            matrix_ = to_same(x)->raw_matrix() * to_same(y)->raw_matrix() * a;

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{M} + a\mathbf{X}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$
    * is this matrix, \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to multiply, must have the same number of columns as `y` has rows.
    * @param[in] y - Second matrix to multiply , must have the same number of rows as `x` has columns.
    * @param[in] a - Scalar with which to scale the product of `x` and `y`.
    */
    void add_matmul(
        const MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        ) override
    {
        if (x.num_rows() * x.num_cols() * y.num_rows() * y.num_cols() == 0)
            return;

        if (!(std::abs(a) > float_eps))
            return;

        auto visitor = [this, a] (const auto* xc, const auto* yc)
        { matrix_ += xc->raw_matrix() * yc->raw_matrix() * a; };

        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            std::visit(visitor, to_variant(x), to_variant(y));
        else
            matrix_ += to_same(x)->raw_matrix() * to_same(y)->raw_matrix() * a;

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{X} = a\mathbf{M}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[out] x - Multiplication result.
    * @param[in] y - Matrix with which to multiply, must have the same number of rows as this matrix.
    * @param[in] a - Scalar with which to scale the product.
    */
    void matmul(
        MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        ) const override
    {
        if (y.num_rows() * y.num_cols() == 0)
            return;

        auto visitor = [&] (const auto* yc)
        { to_same(x)->raw_matrix() = matrix_ * yc->raw_matrix() * a; };

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
            to_same(x)->raw_matrix() = matrix_ * to_same(y)->raw_matrix() * a;
        else
            std::visit(visitor, to_variant(y));

        return;
    };


    /**
    * @brief Computes \f$ \mathbf{X} += a\mathbf{M}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[out] x - Multiplication result.
    * @param[in] y - Matrix with which to multiply, must have the same number of rows as this matrix.
    * @param[in] a - Scalar with which to scale the product.
    */
    void matmuladd(
        MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        ) const override
    {
        if (y.num_rows() * y.num_cols() == 0)
            return;

        auto visitor = [&] (const auto* yc)
        { to_same(x)->raw_matrix() += matrix_ * yc->raw_matrix() * a; };

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
            to_same(x)->raw_matrix() += matrix_ * to_same(y)->raw_matrix() * a;
        else
            std::visit(visitor, to_variant(y));

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

        auto visitor = [&] (auto* xc)
        { matrix_.block(row_start, col_start, xc->num_rows(), xc->num_cols()) = xc->raw_matrix() * a; };

        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            std::visit(visitor, to_variant(x));
        else
        {
            const auto& xc = *to_same(x);

            std::vector<Eigen::Triplet<T>> x_triplets;
            x_triplets.reserve(xc.raw_matrix().nonZeros());

            for (Index kk = 0; kk < xc.raw_matrix().outerSize(); ++kk)
                for (typename MatrixType::InnerIterator it (xc.raw_matrix(), kk); it; ++it)
                    x_triplets.emplace_back(
                        Eigen::Triplet<T> (
                            row_start + it.row(), col_start + it.col(), it.value() * a
                            )
                        );

            matrix_.insertFromTriplets(
                x_triplets.begin(),
                x_triplets.end(),
                [] (const T&, const T &b) { return b; }
                );
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

        auto visitor = [&] (auto* xc)
        { matrix_.block(row_start, col_start, xc->num_rows(), xc->num_cols()) += xc->raw_matrix() * a; };

        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
            std::visit(visitor, to_variant(x));
        else
        {
            const auto& xc = *to_same(x);

            std::vector<Eigen::Triplet<T>> x_triplets;
            x_triplets.reserve(xc.raw_matrix().nonZeros());

            for (Index kk = 0; kk < xc.raw_matrix().outerSize(); ++kk)
                for (typename MatrixType::InnerIterator it (xc.raw_matrix(), kk); it; ++it)
                    x_triplets.emplace_back(
                        Eigen::Triplet<T> (
                            row_start + it.row(), col_start + it.col(), it.value() * a
                            )
                        );

            matrix_.insertFromTriplets(x_triplets.begin(), x_triplets.end());
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
        auto visitor = [&] (auto* xc)
        { xc->raw_matrix() = matrix_.block(row_start, col_start, b_rows, b_cols); };

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
            std::visit(visitor, to_variant(x));
        else
        {
            auto& xc = *to_same(x);
            xc.raw_matrix() = matrix_.block(row_start, col_start, b_rows, b_cols);
        }

        return;
    };


    /**
    * @brief Copies a block of values from this matrix to the corresponding block of another.
    * @param[out] x - Matrix to store the retrieved values in its corresponding block.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] b_rows - Number of rows in the block to retrieve.
    * @param[in] b_cols - Number of columns in the block to retrieve.
    */
    void copy_block(
        MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        Index b_rows,
        Index b_cols
        ) const override
    {
        std::unique_ptr<MatrixBase<T>> temp = x.clone();
        get_block(*temp, row_start, col_start, b_rows, b_cols);
        x.set_block(*temp, row_start, col_start);
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

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
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

        const auto& bd =
            dynamic_cast<const EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>&> (b);

        auto& xd =
            dynamic_cast<EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>&> (x);

        if constexpr (type == EigenMatrixType::EIGEN_DENSE)
        {
            xd.raw_matrix() = dense_solver_->solve(bd.raw_matrix());
            if (dense_solver_->info() != Eigen::Success)
                throw std::runtime_error("EigenMatrix::solve(): Dense solver failed.");
        }

        if constexpr (type == EigenMatrixType::EIGEN_SPARSE)
        {
            xd.raw_matrix() = sparse_solver_->solve(bd.raw_matrix());
            if (sparse_solver_->info() != Eigen::Success)
                throw std::runtime_error("EigenMatrix::solve(): Sparse solver failed.");
        }

        return;
    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with the
    * GMRES iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] tol - Tolerance for convergence (optional).
    * @param[in] restart - Restart iteration at which the Krylov subspace is discarded (optional).
    */
    void mat_solve_gmres(
        EigenMatrix<T, type, storage_order>& x,
        const EigenMatrix<T, type, storage_order>& b,
        const Float tol = 1e-4,
        const Index restart = 100
        )
    {
        Eigen::GMRES<MatrixType, Eigen::IdentityPreconditioner> solver;
        // Eigen::GMRES<MatrixType, Eigen::DiagonalPreconditioner<T>> solver;

        solver.setTolerance(tol);
        solver.setMaxIterations(1000);
        solver.set_restart(restart);

        solver.compute(raw_matrix());

        if (solver.info() != Eigen::Success)
            throw std::runtime_error("EigenMatrix::mat_solve_gmres(): Solver initialization failed.");

        x.raw_matrix() = solver.solve(b.raw_matrix());

        std::cout << "GMRES iterations: "
                  << solver.iterations()
                  << " | tolerance: "
                  << solver.tolerance()
                  << " | residual: "
                  << solver.error()
                  << std::endl;

        if (solver.info() != Eigen::Success)
            std::cout << "EigenMatrix::mat_solve_gmres(): [Warning] GMRES solve failed." << std::endl;

        return;
    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with the
    * BiCGSTAB iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] tol - Tolerance for convergence (optional).
    */
    void mat_solve_bicgstab(
        EigenMatrix<T, type, storage_order>& x,
        const EigenMatrix<T, type, storage_order>& b,
        const Float tol = 1e-4
        )
    {
        Eigen::BiCGSTAB<MatrixType, Eigen::IdentityPreconditioner> solver;
        // Eigen::BiCGSTAB<MatrixType, Eigen::DiagonalPreconditioner<T>> solver;

        solver.setTolerance(tol);
        solver.setMaxIterations(1000);

        solver.compute(raw_matrix());

        if (solver.info() != Eigen::Success)
            throw std::runtime_error("EigenMatrix::mat_solve_bicgstab(): Solver initialization failed.");

        x.raw_matrix() = solver.solve(b.raw_matrix());

        std::cout << "BiCGStab iterations: "
                  << solver.iterations()
                  << " | tolerance: "
                  << solver.tolerance()
                  << " | residual: "
                  << solver.error()
                  << std::endl;

        if (solver.info() != Eigen::Success)
            std::cout << "EigenMatrix::mat_solve_bicgstab(): [Warning] BiCGStab solve failed." << std::endl;

        return;
    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with an
    * iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] type - Type of iterative solver to use (optional).
    * @param[in] tol - Tolerance for convergence (optional).
    * @param[in] restart - Restart iteration at which the Krylov subspace is discarded (optional).
    */
    void mat_solve_iterative(
        EigenMatrix<T, type, storage_order>& x,
        const EigenMatrix<T, type, storage_order>& b,
        const EigenIterativeSolverType solver_type = EigenIterativeSolverType::EIGEN_GMRES,
        const Float tol = 1e-4,
        const Index restart = 100
        ) const
    {
        if (solver_type == EigenIterativeSolverType::EIGEN_GMRES)
        {
            mat_solve_iterative<Eigen::GMRES<MatmulMatrix<MatrixType>, Eigen::IdentityPreconditioner>> (
                x, b, tol, restart
                );
        }
        else if (solver_type == EigenIterativeSolverType::EIGEN_GMRES)
        {
            mat_solve_iterative<Eigen::GMRES<MatmulMatrix<MatrixType>, Eigen::IdentityPreconditioner>> (
                x, b, tol, restart
                );
        }

        return;
    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with an
    * iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] tol - Tolerance for convergence (optional).
    * @param[in] restart - Restart iteration at which the Krylov subspace is discarded (optional).
    */
    template<typename Solver = Eigen::GMRES<MatrixType, Eigen::IdentityPreconditioner>>
    void mat_solve_iterative(
        EigenMatrix<T, type, storage_order>& x,
        const EigenMatrix<T, type, storage_order>& b,
        const Float tol = 1e-4,
        const Index restart = 100
        ) const
    {

        Solver solver;

        solver.setTolerance(tol);
        solver.setMaxIterations(1000);
        if constexpr (
            std::is_same_v<Solver, Eigen::GMRES<MatrixType, Eigen::IdentityPreconditioner>>
            )
            solver.set_restart(restart);

        solver.compute(raw_matrix());

        if (solver.info() != Eigen::Success)
            throw std::runtime_error("EigenMatrix::mat_solve_iterative(): Solver initialization failed.");

        x.raw_matrix() = solver.solve(b.raw_matrix());

        std::cout << "Iterations: "
                  << solver.iterations()
                  << " | tolerance: "
                  << solver.tolerance()
                  << " | residual: "
                  << solver.error()
                  << std::endl;

        if (solver.info() != Eigen::Success)
            std::cout << "EigenMatrix::mat_solve_iterative(): [Warning] Iterative solve failed." << std::endl;

        return;

    };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with an
    * iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix, with a wrapper matrix that defines a custom matrix product.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] mat - Custom matrix product object.
    * @param[in] tol - Tolerance for convergence (optional).
    * @param[in] restart - Restart iteration at which the Krylov subspace is discarded (optional).
    */
    template<
        typename MatmulMatrixType,
        typename Solver = Eigen::GMRES<MatmulMatrixType, Eigen::IdentityPreconditioner>
        >
    static void mat_solve_iterative(
        EigenMatrix<T, type, storage_order>& x,
        const EigenMatrix<T, type, storage_order>& b,
        const MatmulMatrixType& mat,
        const Float tol = 1e-4,
        const Index restart = 100
        )
    {

        Solver solver;

        solver.setTolerance(tol);
        solver.setMaxIterations(1000);
        if constexpr (std::is_same_v<
                      Solver,
                      Eigen::GMRES<MatmulMatrixType, Eigen::IdentityPreconditioner>
                      >)
        {
            solver.set_restart(restart);
        }

        solver.compute(mat);

        if (solver.info() != Eigen::Success)
            throw std::runtime_error("EigenMatrix::mat_solve_iterative(): Solver initialization failed.");

        x.raw_matrix() = solver.solve(b.raw_matrix());

        std::cout << "Iterations: "
                  << solver.iterations()
                  << " | tolerance: "
                  << solver.tolerance()
                  << " | residual: "
                  << solver.error()
                  << std::endl;

        if (solver.info() != Eigen::Success)
            std::cout << "EigenMatrix::mat_solve_iterative(): [Warning] Iterative solve failed." << std::endl;

        return;

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


protected:

    // Type alias for the matrix variant to support both dense and sparse matrices.
    using MatrixVariant = std::variant<
        EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>*,
        EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE, storage_order>*
    >;


    // Type alias for the matrix variant to support both dense and sparse matrices in read-only mode.
    using ConstMatrixVariant = std::variant<
        const EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>*,
        const EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE, storage_order>*
    >;


    /**
    * @brief Casts a MatrixBase reference to a matching-type read-only matrix pointer.
    * @param[in] x - Matrix reference to cast.
    * @return Read-only cast matrix pointer.
    */
    const EigenMatrix<T, type, storage_order>* to_same(const MatrixBase<T>& x) const
    {
        if (auto xc = dynamic_cast<const EigenMatrix<T, type, storage_order>*> (&x))
            return xc;
        else
            throw std::invalid_argument("EigenMatrix::to_same(): Incompatible matrix types.");
    };


    /**
    * @brief Casts a MatrixBase reference to a matching-type writable matrix pointer.
    * @param[in] x - Matrix reference to cast.
    * @return Writable cast matrix pointer.
    */
    EigenMatrix<T, type, storage_order>* to_same(MatrixBase<T>& x) const
    {
        if (auto xc = dynamic_cast<EigenMatrix<T, type, storage_order>*> (&x))
            return xc;
        else
            throw std::invalid_argument("EigenMatrix::to_same(): Incompatible matrix types.");
    };


    /**
    * @brief Casts a MatrixBase reference to a type-erased writable matrix variant.
    * @param[in] x - Matrix reference to convert.
    * @return Variant containing a pointer to the concrete matrix type.
    */
    MatrixVariant to_variant(MatrixBase<T>& x) const
    {
        if (auto xd = dynamic_cast<EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>*> (&x))
            return xd;
        else if (auto xs = dynamic_cast<EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE, storage_order>*> (&x))
            return xs;
        else
            throw std::invalid_argument("EigenMatrix::to_variant(): Input matrix type not supported.");
    };


    /**
    * @brief Casts a MatrixBase reference to a type-erased read-only matrix variant.
    * @param[in] x - Matrix reference to convert.
    * @return Variant containing a pointer to the concrete matrix type.
    */
    ConstMatrixVariant to_variant(const MatrixBase<T>& x) const
    {
        if (auto xd = dynamic_cast<const EigenMatrix<T, EigenMatrixType::EIGEN_DENSE, storage_order>*> (&x))
            return xd;
        else if (auto xs = dynamic_cast<const EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE, storage_order>*> (&x))
            return xs;
        else
            throw std::invalid_argument("EigenMatrix::to_variant(): Input matrix type not supported.");
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

};

/**
* @}
*/

}


namespace Eigen
{
namespace internal
{

template <typename MatrixType>
struct traits<bem::MatmulMatrix<MatrixType>>:
        public Eigen::internal::traits<Eigen::SparseMatrix<bem::Complex>> {};

}
}

namespace bem
{

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

    void attach(const Func& func, Index num_rows, Index num_cols)
    {
        func_ = &func;
        num_rows_ = num_rows;
        num_cols_ = num_cols;
        return;
    };

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

template <typename Rhs, typename MatrixType>
struct generic_product_impl<bem::MatmulMatrix<MatrixType>, Rhs, SparseShape, DenseShape, GemvProduct>:
    generic_product_impl_base<
        bem::MatmulMatrix<MatrixType>,
        Rhs,
        generic_product_impl<bem::MatmulMatrix<MatrixType>, Rhs>
        >
{

    typedef typename Product<bem::MatmulMatrix<MatrixType>, Rhs>::Scalar Scalar;

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

