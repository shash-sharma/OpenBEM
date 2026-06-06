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

#include <memory>

#include "types.hpp"
#include "geometry/operations.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"
#include "rwg/integrators/obs/base.hpp"


namespace bem::rwg
{

/**
* \ingroup rwgops
* @{
*/

/**
* @brief Base class for RWG-based BEM operators.
*/
class OperatorBase
{
public:

    /**
    * @brief Returns the number of degrees of freedom per triangle for the testing function space.
    * @return Number of observation degrees of freedom per triangle.
    */
    virtual uint8_t obs_dof() const = 0;


    /**
    * @brief Returns the number of degrees of freedom per triangle for the expansion function space.
    * @return Number of source degrees of freedom per triangle.
    */
    virtual uint8_t src_dof() const = 0;


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
    virtual EigMat<Complex> compute(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri
        ) = 0;


    /**
    * @brief Assembles the computed integrals into the final operator values.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_tri - Observation triangle.
    * @param[in] src_tri - Source triangle.
    * @param[in] obs_result - Integration result.
    * @return Operator values for each combination of degrees of freedom.
    */
    virtual EigMat<Complex> assemble(
        const Complex k,
        const Triangle<3>& obs_tri,
        const Triangle<3>& src_tri,
        const ObsResult& obs_result
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


    /**
    * @brief Returns a unique pointer to a newly constructed object of the derived type.
    * @return Unique pointer to the new object.
    */
    virtual std::unique_ptr<OperatorBase> clone() const
    { throw std::runtime_error("OperatorBase::clone(): Not implemented."); };


    /**
    * @brief Virtual destructor.
    */
    virtual ~OperatorBase() = default;

};

/**
* @}
*/

}

#endif
