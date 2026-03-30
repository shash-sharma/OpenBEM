// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for RWG-based BEM operators.
*/

#ifndef BEM_RWG_OPS_BASE_H
#define BEM_RWG_OPS_BASE_H

#include "types.hpp"
#include "geometry/operations.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

/**
* \ingroup rwgops
* @{
*/

/**
* @brief Base class for RWG-based BEM operators.
* @tparam obs_num_dof - Number of degrees of freedom associated with the observation triangle.
* @tparam src_num_dof - Number of degrees of freedom associated with the source triangle.
*/
template <uint8_t obs_num_dof, uint8_t src_num_dof>
class OperatorBase
{

    static_assert((obs_num_dof > 0), "OperatorBase: `obs_num_dof` must be greater than 0.");
    static_assert((src_num_dof > 0), "OperatorBase: `src_num_dof` must be greater than 0.");

public:

    /**
    * @brief Computes operator values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @return Operator values for each pair of observation and source degrees of freedom.
    * @details
    * Rows of the output matrix correspond to observation degrees of freedom, and columns
    * correspond to source degrees of freedom.
    */
    virtual EigMatMN<Complex, obs_num_dof, src_num_dof> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) = 0;


    /**
    * @brief Transforms the coordinates of the observation and source triangles into a local
    * coordinate system defined by the source triangle.
    * @param[out] obs_tri_local - Observation triangle in the source's local coordinate system.
    * @param[out] src_tri_local - Source triangle in its local coordinate system.
    * @param[in] obs_tri - Original observation triangle.
    * @param[in] src_tri - Original source triangle.
    */
    static void transform_coordinates(
        Triangle<3>& obs_tri_local,
        Triangle<2>& src_tri_local,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        )
    {
        EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
        EigColVecN<Float, 3> local_origin = src_tri.local_origin();

        obs_tri_local.set_v(
            GeometryOps<3>::transform_coordinate_system(
                obs_tri.v(), local_origin, local_uvw
                ),
            obs_tri.edge_polarities()
        );
        src_tri_local.set_v(
            GeometryOps<3>::transform_coordinate_system(
                src_tri.v(), local_origin, local_uvw
                ).topRows(2),
            src_tri.edge_polarities()
        );

        return;
    };

};

/**
* @}
*/

}

#endif
