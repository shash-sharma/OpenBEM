// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for for matrix algebra containers.
*/

#ifndef MATRIX_BASE_HPP
#define MATRIX_BASE_HPP

#include <stdexcept>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>

#include "types.hpp"


namespace bem
{

/**
* \addtogroup matr
* @{
*/

/**
* @brief Base class for for matrix algebra containers.
* @tparam T - Data type to be stored in the matrix (e.g., float, double, std::complex).
*/
template <typename T>
class MatrixBase
{
public:

    /**
    * @brief Returns the total number of rows in the matrix.
    * @return Number of rows.
    */
    virtual Index num_rows() const = 0;


    /**
    * @brief Returns the total number of columns in the matrix.
    * @return Number of columns.
    */
    virtual Index num_cols() const = 0;


    /**
    * @brief Resizes the matrix to a new number of rows and columns.
    * @param[in] rows - New number of rows.
    * @param[in] cols - New number of columns.
    */
    virtual void resize(Index rows, Index cols) = 0;


    /**
    * @brief Clears all data in the matrix and sets its size to 0.
    */
    virtual void clear() = 0;


    /**
    * @brief Returns the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @return Value at the specified position in the matrix.
    */
    virtual T value(Index row, Index col) const = 0;


    /**
    * @brief Sets the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to set.
    */
    virtual void set_value(Index row, Index col, const T& a) = 0;


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{M} + a\mathbf{X} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ is a matrix.
    * @param[in] x - Matrix to scale and add, must have the same dimensions as this matrix.
    * @param[in] a - Scalar which which to scale `x`.
    */
    virtual void add_ax(const MatrixBase<T>& x, const T& a = T(1)) = 0;


    /**
    * @brief Scales all matrix entries by a given value.
    * @param[in] a - Value by which to scale.
    */
    virtual void scale(const T& a) = 0;


    /**
    * @brief Sets all matrix entries to zero.
    */
    virtual void set_zero() = 0;


    /**
    * @brief Returns the size (rows * cols) of the matrix.
    * @return Size (rows * cols).
    */
    virtual Index size() const { return num_rows() * num_cols(); };


    /**
    * @brief Adds to the matrix value at the specified row and column.
    * @param[in] row - Row index.
    * @param[in] col - Column index.
    * @param[in] a - Value to add.
    */
    virtual void add_value(Index row, Index col, const T& a)
    {
        set_value(row, col, value(row, col) + a);
        return;
    };


    /**
    * @brief Preallocates memory for a given number of non-zero values per row.
    * @param[in] nnz - Vector containing the number of non-zero values for each row.
    */
    virtual void preallocate(const std::vector<Index>& nnz) {};


    /**
    * @brief Preallocates memory for a given total number of non-zero values.
    * @param[in] num_entries - Total number of non-zero values.
    */
    virtual void preallocate(Index num_entries) {};


    /**
    * @brief Assembles cached data into the matrix.
    */
    virtual void assemble() {};


    /**
    * @brief Returns a read-only pointer to the underlying raw data.
    * @return Read-only pointer to the raw data.
    */
    virtual const T* data() const
    { throw std::runtime_error("MatrixBase::data(): Not implemented."); };


    /**
    * @brief Returns a writable pointer to the underlying raw data - use with care.
    * @return Writable pointer to the raw data.
    */
    virtual T* data()
    { throw std::runtime_error("MatrixBase::data(): Not implemented."); };


    /**
    * @brief Returns a unique pointer to a newly constructed object of the derived type.
    * @return Unique pointer to the new object.
    */
    virtual std::unique_ptr<MatrixBase<T>> clone() const
    { throw std::runtime_error("MatrixBase::clone(): Not implemented."); };


    /**
    * @brief Retrieves matrix values on the diagonal.
    * @param[out] x - Diagonal matrix.
    */
    virtual void get_diagonal(MatrixBase<T>& x) const
    { throw std::runtime_error("MatrixBase::get_diagonal(): Not implemented."); };


    /**
    * @brief Computes \f$ \mathbf{M} = a\mathbf{X}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to multiply, must have the same number of columns as `y` has rows.
    * @param[in] y - Second matrix to multiply , must have the same number of rows as `x` has columns.
    * @param[in] a - Scalar which which to scale the product of `x` and `y`.
    */
    virtual void set_matmul(
        const MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        )
    { throw std::runtime_error("MatrixBase::set_matmul(): Not implemented."); };


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{M} + a\mathbf{X}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$
    * is this matrix, \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to multiply, must have the same number of columns as `y` has rows.
    * @param[in] y - Second matrix to multiply , must have the same number of rows as `x` has columns.
    * @param[in] a - Scalar which which to scale the product of `x` and `y`.
    */
    virtual void add_matmul(
        const MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        )
    { throw std::runtime_error("MatrixBase::add_matmul(): Not implemented."); };


    /**
    * @brief Computes \f$ \mathbf{X} = a\mathbf{M}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[out] x - Multiplication result.
    * @param[in] y - Matrix with which to multiply, must have the same number of rows as this matrix.
    * @param[in] a - Scalar which which to scale the product.
    */
    virtual void matmul(
        MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        ) const
    { throw std::runtime_error("MatrixBase::matmul(): Not implemented."); };


    /**
    * @brief Computes \f$ \mathbf{X} += a\mathbf{M}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[out] x - Multiplication result.
    * @param[in] y - Matrix with which to multiply, must have the same number of rows as this matrix.
    * @param[in] a - Scalar which which to scale the product.
    */
    virtual void matmuladd(
        MatrixBase<T>& x,
        const MatrixBase<T>& y,
        const T& a = T(1)
        ) const
    { throw std::runtime_error("MatrixBase::matmul(): Not implemented."); };


    /**
    * @brief Sets a block of this matrix to the values of a given matrix, starting at a given position.
    * @param[in] x - Matrix to insert.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] a - Scalar to multiply the values of `x` before inserting (optional).
    */
    virtual void set_block(
        const MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        const T& a = T(1)
        )
    { throw std::runtime_error("MatrixBase::set_block(): Not implemented."); };


    /**
    * @brief Adds a block to this matrix from the values of a given matrix, starting at a given position.
    * @param[in] x - Matrix whose values should be added to a block of this matrix.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] a - Scalar to multiply the values of `x` before adding (optional).
    */
    virtual void add_block(
        const MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        const T& a = T(1)
        )
    { throw std::runtime_error("MatrixBase::add_block(): Not implemented."); };


    /**
    * @brief Scales a block of this matrix.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] b_rows - Number of rows in the block to scale.
    * @param[in] b_cols - Number of columns in the block to scale.
    * @param[in] a - Scaling factor.
    */
    virtual void scale_block(
        Index row_start,
        Index col_start,
        Index b_rows,
        Index b_cols,
        const T& a
        )
    { throw std::runtime_error("MatrixBase::scale_block(): Not implemented."); };


    /**
    * @brief Retrieves a block of values from this matrix.
    * @param[out] x - Matrix to store the retrieved block of values.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] b_rows - Number of rows in the block to retrieve.
    * @param[in] b_cols - Number of columns in the block to retrieve.
    */
    virtual void get_block(
        MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        Index b_rows,
        Index b_cols
        ) const
    { throw std::runtime_error("MatrixBase::get_block(): Not implemented."); };


    /**
    * @brief Copies a block of values from this matrix to the corresponding block of another.
    * @param[out] x - Matrix to store the retrieved values in its corresponding block.
    * @param[in] row_start - Starting row index for the block.
    * @param[in] col_start - Starting column index for the block.
    * @param[in] b_rows - Number of rows in the block to retrieve.
    * @param[in] b_cols - Number of columns in the block to retrieve.
    */
    virtual void copy_block(
        MatrixBase<T>& x,
        Index row_start,
        Index col_start,
        Index b_rows,
        Index b_cols
        ) const
    { throw std::runtime_error("MatrixBase::copy_block(): Not implemented."); };


    /**
    * @brief Computes and stores a factorization of the matrix.
    */
    virtual void factorize()
    { throw std::runtime_error("MatrixBase::factorize(): Not implemented."); };


    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with a
    * direct solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    */
    virtual void mat_solve(MatrixBase<T>& x, const MatrixBase<T>& b) const
    { throw std::runtime_error("MatrixBase::mat_solve(): Not implemented."); };


    /**
    * @brief Sets the matrix to identity (ones along the diagonal, zeros elsewhere).
    */
    virtual void set_identity()
    {
        set_zero();
        for (Index ii = 0; ii < num_rows(); ++ii)
            set_value(ii, ii, T(1));
        return;
    };


    /**
    * @brief Writes the matrix to a binary file.
    */
    virtual void write_binary(const std::string name = "matrix") const
    {
        std::ofstream out (name, std::ios::out | std::ios::binary | std::ios::trunc);
        Index rows = num_rows(), cols = num_cols();
        out.write((char*) (&rows), sizeof(Index));
        out.write((char*) (&cols), sizeof(Index));
        out.write((char*) data(), rows * cols * sizeof(T));
        out.close();
        return;
    };


    /**
    * @brief Reads the matrix from a binary file.
    */
    virtual void read_binary(const std::string name = "matrix")
    {
        std::ifstream in (name, std::ios::in | std::ios::binary);
        Index rows = 0, cols = 0;
        in.read((char*) (&rows), sizeof(Index));
        in.read((char*) (&cols), sizeof(Index));
        resize(rows, cols);
        in.read((char *) data() , rows * cols * sizeof(T));
        in.close();
        return;
    };


    /**
    * @brief Prints the matrix to the terminal in a formatted manner.
    */
    virtual void print(const std::string name = "matrix") const
    {
        std::cout << "\n--- " << name << " (" << num_rows() << " x " << num_cols() << ") ---" << std::endl;
        for (Index ii = 0; ii < num_rows(); ++ii)
        {
            for (Index jj = 0; jj < num_cols(); ++jj)
            {
                std::cout << std::scientific
                          << std::setprecision(4)
                          << std::setw(14)
                          << value(ii, jj)
                          << "  "
                          << std::flush;
            }
            std::cout << std::endl;
        }
        std::cout << "---" << std::endl;
        return;
    };


    /**
    * @brief Virtual destructor.
    */
    virtual ~MatrixBase() = default;

};

/**
* @}
*/

}

#endif
