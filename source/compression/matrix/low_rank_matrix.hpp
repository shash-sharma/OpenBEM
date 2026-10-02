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

#ifndef BEM_LR_MATRIX_HPP
#define BEM_LR_MATRIX_HPP

#include <type_traits>
#include <stdexcept>
#include <memory>

#include <external/Eigen/Dense>
#include <external/Eigen/SVD>

#include "types.hpp"
#include "constants.hpp"
#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"


namespace bem
{

/**
* \ingroup compression
* @{
*/

/**
* @brief Class wrapping an Eigen-based low-rank matrix represented as
* \f$ \mathbf{U}\mathbf{S}\mathbf{V}^T \f$.
* @tparam T - Data type to be stored in the matrix (e.g., float, double, std::complex).
*/
template <typename T = Complex>
class LowRankMatrix: public MatrixBase<T>
{

    using DenseMatrixType = EigenMatrix<T, EigenMatrixType::EIGEN_DENSE>;
    using SparseMatrixType = EigenMatrix<T, EigenMatrixType::EIGEN_SPARSE>;

public:

    using MatrixBase<T>::assemble;

    
    /**
    * @brief Constructs an empty `LowRankMatrix` object.
    */
    LowRankMatrix() {};


    /**
    * @brief Constructs a `LowRankMatrix` object with specified compressed blocks.
    * @param[in] u - Matrix \f $\mathbf{U} \f$.
    * @param[in] s - Matrix \f $\mathbf{S} \f$.
    * @param[in] v - Matrix \f $\mathbf{V} \f$.
    */
    LowRankMatrix(
        ConstEigRef<EigMat<T>> u,
        ConstEigRef<EigColVec<T>> s,
        ConstEigRef<EigMat<T>> v
        )
    {
        set_usv(u, s, v);
        return;
    };


    /**
    * @brief Sets the compressed matrices for this `LowRankMatrix` object.
    * @param[in] u - Matrix \f $\mathbf{U} \f$.
    * @param[in] s - Diagonal of matrix \f $\mathbf{S} \f$.
    * @param[in] v - Matrix \f $\mathbf{V} \f$.
    */
    void set_usv(
        ConstEigRef<EigMat<T>> u,
        ConstEigRef<EigColVec<T>> s,
        ConstEigRef<EigMat<T>> v
        );


    /**
    * @brief Appends factors to this matrix, adding `u * v.transpose()` to its value.
    * @param[in] u - Factor to append to \f$\mathbf{U}\f$.
    * @param[in] v - Factor to append to \f$\mathbf{V}\f$.
    */
    void append(
        ConstEigRef<EigMat<T>> u,
        ConstEigRef<EigMat<T>> v
        );


    /**
    * @brief Recompresses the block to a given singular value tolerance.
    * @param[in] tol - Singular value tolerance.
    * @param[in] reference - Custom truncation norm (optional).
    */
    void recompress(const Float tol, const Float reference = 0);


    /**
    * @brief Computes \f$ \mathbf{X} = a\mathbf{M}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this
    * matrix, \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are low-rank
    * matrices.
    * @param[out] x - Multiplication result.
    * @param[in] y - Low-rank matrix with which to multiply, must have as many rows as this
    * matrix's columns.
    * @param[in] a - Scalar with which to scale the product.
    * @details Computes the full product with no recompression.
    */
    void matmul_low_rank(
        LowRankMatrix<T>& x,
        const LowRankMatrix<T>& y,
        const T& a = T(1)
        ) const;


    /**
    * @brief Returns a read-only reference to the \f$\mathbf{U}\f$ matrix.
    */
    const EigMat<T>& u() const
    { return u_; };


    /**
    * @brief Returns a read-only reference to the \f$\mathbf{S}\f$ matrix.
    */
    const EigColVec<T>& s() const
    { return s_; };


    /**
    * @brief Returns a read-only reference to the \f$\mathbf{V}\f$ matrix.
    */
    const EigMat<T>& v() const
    { return v_; };


    /**
    * @brief Returns a writable reference to the \f$\mathbf{U}\f$ matrix.
    */
    EigMat<T>& u()
    { return u_; };


    /**
    * @brief Returns a writable reference to the \f$\mathbf{S}\f$ matrix.
    */
    EigColVec<T>& s()
    { return s_; };


    /**
    * @brief Returns a writable reference to the \f$\mathbf{V}\f$ matrix.
    */
    EigMat<T>& v()
    { return v_; };


    /**
    * @brief Returns a unique pointer to an empty object of the derived type.
    * @return Unique pointer to the new object.
    */
    std::unique_ptr<MatrixBase<T>> clone() const override
    { return std::make_unique<LowRankMatrix<T>> (); };


    /**
    * @brief Returns the total number of rows in the matrix.
    * @return Number of rows.
    */
    Index num_rows() const override
    { return u_.rows(); };


    /**
    * @brief Returns the total number of columns in the matrix.
    * @return Number of columns.
    */
    Index num_cols() const override
    { return v_.rows(); };


    /**
    * @brief Resizes and resets the matrix to a new number of rows and columns.
    * @param[in] rows - New number of rows.
    * @param[in] cols - New number of columns.
    */
    void resize(Index rows, Index cols) override;


    /**
    * @brief Clears all data in the matrix and sets its size to 0.
    */
    void clear() override;


    /**
    * @brief Uncompresses and assembles the full dense matrix.
    */
    void assemble(DenseMatrixType& mat) const;


    /**
    * @brief Returns the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @return Value at the specified position in the matrix.
    */
    T value(Index row, Index col) const override;


    /**
    * @brief Sets the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to set.
    */
    void set_value(Index row, Index col, const T& a) override
    { throw std::runtime_error("LowRankMatrix::set_value(): Not available for this matrix type."); };


    /**
    * @brief Adds to the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to add.
    */
    void add_value(Index row, Index col, const T& a) override
    { throw std::runtime_error("LowRankMatrix::add_value(): Not available for this matrix type."); };


    /**
    * @brief Scales all matrix entries by a given value.
    * @param[in] a - Value by which to scale.
    */
    void scale(const T& a) override
    { s_ *= a; return; };


    /**
    * @brief Sets all matrix entries to zero.
    */
    void set_zero() override;


    /**
    * @brief Sets the matrix to identity (ones along the diagonal).
    */
    void set_identity() override
    { throw std::runtime_error("LowRankMatrix::set_identity(): Not available for this matrix type."); };


    /**
    * @brief Sets a block of this matrix to the values of a given matrix, starting at a given position.
    * @param[in] x - Matrix to insert, must span this whole matrix.
    * @param[in] row_start - Starting row index for the block, must be zero.
    * @param[in] col_start - Starting column index for the block, must be zero.
    * @param[in] a - Scalar to multiply the values of `x` before inserting (optional).
    */
    void set_block(
        const MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        const T& a = T(1)
        ) override;


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{X}^T \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ \mathbf{X} \f$ is a given matrix, and \f$ \mathbf{X}^T \f$ is its transpose.
    * @param[in] x - Matrix to transpose.
    */
    void set_transpose(const MatrixBase<T>& x) override;


    /**
    * @brief Retrieves matrix values on the diagonal.
    * @param[out] x - Diagonal matrix.
    */
    void get_diagonal(MatrixBase<T>& x) const override;


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{M} + a\mathbf{X} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ is a matrix.
    * @param[in] x - Matrix to scale and add, must have the same dimensions as this matrix.
    * @param[in] a - Scalar with which to scale `x`.
    */
    void add_ax(const MatrixBase<T>& x, const T& a = T(1)) override
    { throw std::runtime_error("LowRankMatrix::add_ax(): Not available for this matrix type."); };


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
        ) const override;


    /**
    * @brief Computes \f$ \mathbf{X}_b = a\mathbf{M}\mathbf{Y}_b \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices, and the subscript \f$ b \f$
    * indicates that the matrix is multiplied into a sub-block of \f$ \mathbf{Y} \f$, and the result is
    * placed in a sub-block of \f$ \mathbf{X} \f$.
    * @param[in,out] x - Destination matrix, with at least as many rows as rows of this matrix.
    * @param[in] y - Matrix whose sub-block to multiply, with at least as many rows as columns of this matrix.
    * @param[in] x_row_start - Starting row index for the destination block.
    * @param[in] y_row_start - Starting row index for the multiplier block.
    * @param[in] a - Scalar with which to scale the product.
    * @param[in] accumulate - Whether to accumulate the product into a pre-existing `x`.
    * @details
    * If `x` is completely unallocated (0x0), it is resized automatically to fit the destination
    * block; otherwise it must already be at least as large as the destination block requires
    * (an exception is thrown if it is undersized), and the existing block is updated in place.
    */
    void matmul_block(
        MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const Index x_row_start,
        const Index y_row_start,
        const T& a = T(1),
        const bool accumulate = false
        ) const override;


    /**
    * @brief Computes \f$ \mathbf{X} = \mathbf{L}\mathbf{M}\mathbf{R} \f$ where \f$ \mathbf{M} \f$ is
    * this matrix, and \f$ \mathbf{L} \f$ and \f$ \mathbf{R} \f$ are given matrices.
    * @param[out] x - Multiplication result, must be a `LowRankMatrix`.
    * @param[in] left - Matrix with which to multiply on the left, or empty for the identity.
    * @param[in] right - Matrix with which to multiply on the right, or empty for the identity.
    */
    void matmul_triple(
        MatrixBase<T>& x,
        const MatrixBase<T>& left,
        const MatrixBase<T>& right
        ) const override;


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
        ) const override;


    /**
    * @brief Returns the approximated rank of the matrix.
    * @return Rank.
    */
    Index rank() const
    { return u_.cols(); };


    /**
    * @brief Computes the condition number (ratio of largest to smallest singular value) of the matrix.
    * @return Condition number.
    */
    Float cond() const;


    /**
    * @brief Prints the matrix to the terminal in a formatted manner.
    */
    void print(const std::string name = "matrix") const override;


protected:

    EigMat<T> u_, v_;
    EigColVec<T> s_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "compression/matrix/low_rank_matrix.cpp"
#endif

#endif

