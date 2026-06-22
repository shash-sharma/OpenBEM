// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Functionality for setting infinitesimal gap excitation coefficients for RWG-based BEM systems.
*/

#ifndef BEM_RWG_EXC_INF_GAP_H
#define BEM_RWG_EXC_INF_GAP_H

#include <stdexcept>
#include <vector>

#include "types.hpp"
#include "rwg/function_space.hpp"
#include "rwg/excitations/base.hpp"


namespace bem
{
// Forward declarations
template <uint8_t dim> class Triangle;
}


namespace bem::rwg
{

/**
* @brief Default tolerance used to identify mesh edges to associate with the infinitesimal gap
* coordinates.
*/
const Float INF_GAP_DEFAULT_TOL = 1.0e-6;

/**
* \addtogroup rwgexc
* @{
*/

/**
* @brief Class for setting infinitesimal gap excitation coefficients for RWG-based BEM systems.
*/
class InfinitesimalGap: public ExcitationBase
{
public:

    /**
    * @brief Constructs an `InfinitesimalGap` object along a given set of line segments.
    * @param[in] segments - Coordinates of the line segments along which the gap excitations are set.
    * @param[in] amp - Complex amplitudes of each gap excitation.
    * @param[in] tol - Tolerance used to find the mesh edges associated with the line segments (optional).
    * @details Each column of `pos_start` and `pos_stop` (and each entry of `amp`) corresponds to a
    * different excitation.  Multiple columns imply multiple right-hand sides to solve. The
    * excitations are placed along the RWG edges that lie along the line segments, within `tol`,
    * i.e., those RWG edges where both vertices of the RWG edge lie on the line segments. Therefore,
    * `segments` should be defined such that the lines connecting them run along existing mesh
    * edges. IMPORTANT: if the line segments are supposed to define a closed loop, then the first
    * coordinate must explicitly be re-added as the last coordinate, otherwise the loop will not be
    * closed by default. The `InfinitesimalGap` object is agnostic to which field (electric or
    * magnetic) is being considered. In the case of an electric field, it represents a voltage
    * source in the infinitesimal gap between the two triangles associated with all the RWG
    * edge(s). If the testing function is nxRWG rather than RWG, then a magnetic field gap
    * excitation along a closed contour of nxRWG edges could be used to represent a current source.
    */
    InfinitesimalGap(
        const std::vector<EigMatNX<Float, 3>>& segments,
        ConstEigRef<EigRowVec<Complex>> amp,
        const Float tol = INF_GAP_DEFAULT_TOL
        ):
            segments_(segments),
            amp_(amp),
            tol_(tol)
    {
        if (segments_.size() != amp_.cols())
        {
            throw std::invalid_argument(
                "InfinitesimalGap: `segments` and `amp` must have the same number of columns."
                );
        }
        return;
    };


    /**
    * @brief Returns the degrees of freedom for the testing function space.
    * @return Observation degrees of freedom.
    */
    OperatorDof obs_dof() const override { return OperatorDof::EDGE; };


    /**
    * @brief Returns the number of excitations (right-hand sides) to be generated.
    * @return Number of excitations (right-hand sides).
    */
    Index num_excitations() const override { return amp_.cols(); };


    /**
    * @brief Computes the gap excitation coefficients.
    * @param[in] k - Complex wavenumber (ignored).
    * @param[in] obs_tri - Observation triangle in the local coordinate system of `src_tri`.
    * @return Excitation coefficient matrix, where each row corresponds to each edge of `obs_tri`,
    * and each column corresponds to each excitation when there is more than one excitation (i.e.,
    * more than one right-hand side).
    * @details
    * For each edge of `obs_tri`, checks if that edge lies on the line segment(s) defined by
    * `pos_start` and `pos_stop`.  If yes, the excitation amplitude, halved, is assigned. If not,
    * then that edge is skipped. The amplitude is halved because there will be an equal contribution
    * from the other triangle associated with the same edge. If the excitation is placed along a
    * half-edge so that there is only one triangle associated with the half-RWG, then the physical
    * interpretation of such an excitation is left up to the user.
    */
    EigMat<Complex> compute(const Complex k, const Triangle<3>& obs_tri) override;


private:

    const std::vector<EigMatNX<Float, 3>> segments_;
    const EigRowVec<Complex> amp_;
    const Float tol_ = INF_GAP_DEFAULT_TOL;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/excitations/inf_gap.cpp"
#endif

#endif
