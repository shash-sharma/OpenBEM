// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for Eigen-based matrices.
*/

#ifndef EIGEN_MATRIX_BASE_HPP
#define EIGEN_MATRIX_BASE_HPP

#include <memory>

#include <external/Eigen/Dense>
#include <external/Eigen/IterativeLinearSolvers>
#include <external/EigenUnsupported/Eigen/IterativeSolvers>

#include "types.hpp"
#include "matrix/base.hpp"


namespace bem
{

/**
* \defgroup eigmat Eigen Matrix Wrappers
* \ingroup matr
* @{
*/

/**
* @brief Base class for Eigen-based matrices.
* @tparam T - Data type to be stored in the matrix (e.g., float, double, std::complex).
* @tparam MatrixType - Specific Eigen matrix type.
*/
template <typename T, typename MatrixType>
class EigenMatrixBase: public MatrixBase<T>
{
public:

    /**
    * @brief Constructs an `EigenMatrixBase` object with a specified number of rows and columns.
    * @param[in] rows - Number of rows.
    * @param[in] cols - Number of columns.
    */
    EigenMatrixBase(Index rows, Index cols): matrix_(std::make_shared<MatrixType> ())
    {
        resize(rows, cols);
        set_zero();
        return;
    };


    /**
    * @brief Constructs an empty zero-size `EigenMatrixBase` object.
    */
    EigenMatrixBase(): EigenMatrixBase(0, 0) {};


    /**
    * @brief Sets a block of values in this matrix to the values of a given matrix, starting at a given position.
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
        ) = 0;


    /**
    * @brief Adds a block of values to this matrix from the values of a given matrix,
    * starting at a given position.
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
        ) = 0;


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
        ) const = 0;


    /**
    * @brief Returns a writable reference to the underlying raw matrix - use with caution.
    */
    MatrixType& raw_matrix()
    {
        return *matrix_;
    };


    /**
    * @brief Returns a read-only reference to the underlying raw matrix.
    */
    const MatrixType& raw_matrix() const
    {
        return *matrix_;
    };


    /**
    * @brief Sets the underlying raw matrix by copying over data from a given matrix.
    * @param[in] matrix - The raw matrix to set.
    */
    void set_raw_matrix(const MatrixType& matrix)
    {
        matrix_ = std::make_shared<MatrixType> (matrix);
        return;
    };


    /**
    * @brief Binds the underlying raw matrix to the data pointed at by a given matrix pointer.
    * @param[in] matrix - Pointer to the raw matrix to be bound.
    * @details
    * The purpose of this method is to allow creating an external Eigen matrix of type `MatrixType`,
    * and then transferring ownership of that matrix to this object, but without copying over the
    * underlying data, unlike `set_raw_matrix()`.
    */
    void bind_raw_matrix(std::shared_ptr<MatrixType> matrix)
    {
        matrix_ = std::move(matrix);
        return;
    };


    /**
    * @brief Returns the total number of rows in the matrix.
    * @return Number of rows.
    */
    Index num_rows() const override
    {
        return matrix_->rows();
    };


    /**
    * @brief Returns the total number of rows in the matrix.
    * @return Number of rows.
    */
    Index num_cols() const override
    {
        return matrix_->cols();
    };


    /**
    * @brief Returns the size (rows * cols) of the matrix.
    * @return Size (rows * cols).
    */
    Index size() const override
    {
        return matrix_->size();
    };


    /**
    * @brief Resizes the matrix to a new number of rows and columns.
    * @param[in] rows - New number of rows.
    * @param[in] cols - New number of columns.
    */
    void resize(Index rows, Index cols) override
    {
        matrix_->resize(rows, cols);
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
    * @brief Sets all matrix entries to zero.
    */
    void set_zero() override
    {
        matrix_->setZero();
        return;
    };


    /**
    * @brief Sets the matrix to identity (ones along the diagonal).
    */
    void set_identity() override
    {
        matrix_->setIdentity();
        return;
    };



    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{X}^T \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ \mathbf{X} \f$ is a given matrix, and \f$ \mathbf{X}^T \f$ is its transpose.
    * @param[in] x - Matrix to transpose.
    */
    void set_transpose(const MatrixBase<T>& x)
    {
        const EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (x);
        (*matrix_) = xd.raw_matrix().transpose();
        return;
    };


    /**
    * @brief Retrieves matrix values on the diagonal, zeroing out all other entries.
    * @param[out] x - Diagonal matrix.
    */
    void get_diagonal(MatrixBase<T>& x) const
    {
        EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<EigenMatrixBase<T, MatrixType>&> (x);
        xd.raw_matrix() = matrix_->diagonal().asDiagonal();
        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{M} + a\mathbf{X} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ is a matrix.
    * @param[in] x - Matrix to scale and add, must have the same dimensions as this matrix.
    * @param[in] a - Scalar which which to scale `x`.
    */
    void add_ax(const MatrixBase<T>& x, const T& a = T(1)) override
    {
        const EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (x);
        raw_matrix() += a * xd.raw_matrix();
        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = a\mathbf{X} + b\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ and \f$ b \f$ are scalars, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to scale and add, must have the same dimensions as `y`.
    * @param[in] y - Second matrix to scale and add, must have the same dimensions as `x`.
    * @param[in] a - Scalar which which to scale `x`.
    * @param[in] b - Scalar which which to scale `y`.
    */
    void set_axpby(const MatrixBase<T>& x, const MatrixBase<T>& y, const T& a = T(1), const T& b = T(1))
    {
        const EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (x);
        const EigenMatrixBase<T, MatrixType>& yd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (y);
        raw_matrix() = a * xd.raw_matrix() + b * yd.raw_matrix();
        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = a\mathbf{X}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$ is this matrix,
    * \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to multiply, must have the same number of columns as `y` has rows.
    * @param[in] y - Second matrix to multiply , must have the same number of rows as `x` has columns.
    * @param[in] a - Scalar which which to scale the product of `x` and `y`.
    */
    void set_mat_mul(const MatrixBase<T>& x, const MatrixBase<T>& y, const T& a = T(1))
    {
        const EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (x);
        const EigenMatrixBase<T, MatrixType>& yd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (y);
        raw_matrix() = xd.raw_matrix() * yd.raw_matrix() * a;
        return;
    };


    /**
    * @brief Computes \f$ \mathbf{M} = \mathbf{M} + a\mathbf{X}\mathbf{Y} \f$ where \f$ \mathbf{M} \f$
    * is this matrix, \f$ a \f$ is a scalar, and \f$ \mathbf{X} \f$ and \f$ \mathbf{Y} \f$ are matrices.
    * @param[in] x - First matrix to multiply, must have the same number of columns as `y` has rows.
    * @param[in] y - Second matrix to multiply , must have the same number of rows as `x` has columns.
    * @param[in] a - Scalar which which to scale the product of `x` and `y`.
    */
    void add_mat_mul(const MatrixBase<T>& x, const MatrixBase<T>& y, const T& a = T(1))
    {
        const EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (x);
        const EigenMatrixBase<T, MatrixType>& yd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (y);
        raw_matrix() += xd.raw_matrix() * yd.raw_matrix() * a;
        return;
    };



    /**
    * @brief Solves \f$ \mathbf{M}\mathbf{X} = \mathbf{B} \f$ for matrix \f$ \mathbf{X} \f$ with an
    * iterative solver, where \f$ \mathbf{M} \f$ is this matrix, and \f$ \mathbf{B} \f$ is a given
    * right-hand side matrix.
    * @param[out] x - Solution.
    * @param[in] b - Right-hand side matrix, must have the same number of rows as this matrix.
    * @param[in] tol - Tolerance for convergence (optional).
    */
    void mat_solve_iterative(MatrixBase<T>& x, const MatrixBase<T>& b, const Float tol = 1e-3)
    {
        mat_solve_gmres(x, b, tol);
        // mat_solve_bicgstab(x, b, tol);
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
        MatrixBase<T>& x, const MatrixBase<T>& b, const Float tol = 1e-3, const Index restart = 100
        )
    {
        Eigen::GMRES<MatrixType, Eigen::IdentityPreconditioner> solver;
        // Eigen::GMRES<MatrixType, Eigen::DiagonalPreconditioner<T>> solver;

        solver.setTolerance(tol);
        solver.setMaxIterations(1000);
        solver.set_restart(restart);

        solver.compute(raw_matrix());
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Matrix solver initialization failed.");

        EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<EigenMatrixBase<T, MatrixType>&> (x);
        const EigenMatrixBase<T, MatrixType>& bd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (b);

        xd.raw_matrix() = solver.solve(bd.raw_matrix());

        std::cout << "GMRES iterations: " << solver.iterations() << " | tolerance: " << solver.tolerance() << " | residual: " << solver.error() << std::endl;

        // xd.resize(num_cols(), bd.num_cols());
        // for (Index ii = 0; ii < bd.num_cols(); ++ii)
        //     xd.raw_matrix().col(ii) = solver.solve(bd.raw_matrix().col(ii));

        if (solver.info() != Eigen::Success)
            std::cout << "Warning: GMRES solve failed." << std::endl;

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
    void mat_solve_bicgstab(MatrixBase<T>& x, const MatrixBase<T>& b, const Float tol = 1e-3)
    {
        Eigen::BiCGSTAB<MatrixType, Eigen::IdentityPreconditioner> solver;
        // Eigen::BiCGSTAB<MatrixType, Eigen::DiagonalPreconditioner<T>> solver;

        solver.setTolerance(tol);
        solver.setMaxIterations(1000);

        solver.compute(raw_matrix());
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Matrix solver initialization failed.");

        EigenMatrixBase<T, MatrixType>& xd = dynamic_cast<EigenMatrixBase<T, MatrixType>&> (x);
        const EigenMatrixBase<T, MatrixType>& bd = dynamic_cast<const EigenMatrixBase<T, MatrixType>&> (b);

        xd.raw_matrix() = solver.solve(bd.raw_matrix());

        std::cout << "BiCGStab iterations: " << solver.iterations() << " | tolerance: " << solver.tolerance() << " | residual: " << solver.error() << std::endl;

        // xd.resize(num_cols(), bd.num_cols());
        // for (Index ii = 0; ii < bd.num_cols(); ++ii)
        //     xd.raw_matrix().col(ii) = solver.solve(bd.raw_matrix().col(ii));

        if (solver.info() != Eigen::Success)
            std::cout << "Warning: BiCGStab solve failed." << std::endl;

        return;
    };


    /**
    * @brief Computes the condition number (ratio of largest to smallest singular value) of the matrix.
    * @return Condition number.
    */
    Float cond() const
    {
        // Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd (
        //     (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>) raw_matrix()
        //     );
        Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd (
            (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>) raw_matrix()
            );
        EigColVec<Float> s = svd.singularValues();
        return s(0) / s(s.size() - 1);
    };


    /**
    * @brief Prints the matrix to the terminal in a formatted manner.
    */
    void print(const std::string name = "matrix") const override
    {
        std::cout << "\n--- " << name << " (" << num_rows() << " x " << num_cols() << ") ---" << std::endl;
        std::cout << raw_matrix() << std::endl;
        std::cout << "---" << std::endl;
        return;
    };


protected:

    std::shared_ptr<MatrixType> matrix_ = nullptr;

};

/**
* @}
*/

}

#endif
