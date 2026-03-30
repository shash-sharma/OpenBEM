// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Physical and mathematical constants.
*/

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <limits>

#include "types.hpp"

#ifndef M_PI
#define M_PI 3.1415926535897932384626
#endif

namespace bem
{

/**
* \defgroup cons Constants
* @brief Definitions of physical and mathematical constants.
* @{
*/

/** @brief Imaginary unit. */
const Complex J = Complex(0.0, 1.0);

const Float zero = 0.0;
const Float one = 1.0;
const Float two = 2.0;
const Float four = 4.0;
const Float half = 0.5;

const Float pi = M_PI;
const Float two_pi = (Float)(2.0 * M_PI);
const Float four_pi = (Float)(4.0 * M_PI);
const Float half_pi = (Float)(0.5 * M_PI);
const Float quarter_pi = (Float)(M_PI / 4.0);

/** @brief Vacuum permittivity. */
const Float eps0 = 8.854187817e-12;

/** @brief Vacuum permeability. */
const Float mu0 = four_pi * 1.0e-7;

/** @brief Vacuum wave speed. */
const Float c0 = 1.0 / std::sqrt(eps0 * mu0);

/** @brief Vacuum wave impedance. */
const Float eta0 = std::sqrt(mu0 / eps0);

/** @brief Numerical infinity. */
const Float inf = std::numeric_limits<Float>::infinity();

/** @brief Numerical infinitesimal. */
const Float float_eps = std::numeric_limits<Float>::epsilon();

/**
* @}
*/

}

#endif
