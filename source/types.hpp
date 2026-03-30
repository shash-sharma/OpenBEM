// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Type aliases.
*/

#ifndef TYPES_H
#define TYPES_H

#include <complex>
#include <cstdint>

#include <external/Eigen/Core>


namespace bem
{

/**
* \defgroup types Types
* @brief Classes defining type aliases for convenience and configurability.
* @{
*/

/**
* @name Aliases for built-in C++ types.
* @{
*/

/** @brief Floating point number. */
#if defined(USE_SINGLE_PRECISION)
    using Float = float;
#elif defined(USE_EXTENDED_PRECISION)
    using Float = long double;
#else
    using Float = double;
#endif

/** @brief Complex floating point number. */
using Complex = std::complex<Float>;

/** @brief Unsigned integer type for indices and container sizes. */
using Index = std::size_t;

/** @brief Signed integer type. */
using Int = int;

/**@}*/

/** @name Aliases for Eigen types. */
/**@{*/

/** @brief Dynamic-size matrix containing type `T`. */
template <typename T>
using EigMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/** @brief Fixed-size matrix with `M` rows and `N` columns containing type `T`. */
template <typename T, Index M, Index N>
using EigMatMN = Eigen::Matrix<T, M, N>;

/** @brief Fixed-width matrix with `N` columns containing type `T`. */
template <typename T, Index N>
using EigMatXN = Eigen::Matrix<T, Eigen::Dynamic, N>;

/** @brief Fixed-height matrix with `N` rows containing type `T`. */
template <typename T, Index N>
using EigMatNX = Eigen::Matrix<T, N, Eigen::Dynamic>;

/** @brief Dynamic-size column vector containing type `T`. */
template <typename T>
using EigColVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

/** @brief Fixed-size column vector of size `N` containing type `T`. */
template <typename T, Index N>
using EigColVecN = Eigen::Matrix<T, N, 1>;

/** @brief Dynamic-size row vector containing type `T`. */
template <typename T>
using EigRowVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;

/** @brief Fixed-size row vector of size `N` containing type `T`. */
template <typename T, Index N>
using EigRowVecN = Eigen::Matrix<T, 1, N>;

/** @brief Read-only reference to an Eigen object. */
template <class EigObj>
using ConstEigRef = const Eigen::Ref<const EigObj>;

/** @brief Writable reference to an Eigen object. */
template <class EigObj>
using EigRef = Eigen::Ref<EigObj>;

/**
* @}
*/

/**
* @}
*/

}

#endif




