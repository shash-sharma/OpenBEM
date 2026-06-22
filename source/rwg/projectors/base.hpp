// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for RWG-based BEM projectors.
*/

#ifndef BEM_RWG_PROJ_BASE_H
#define BEM_RWG_PROJ_BASE_H

#include "types.hpp"
#include "geometry/operations.hpp"
#include "geometry/primitives/triangle.hpp"
#include "rwg/function_space.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgproj
* @{
*/

/**
* @brief Base class for RWG-based BEM projectors.
*/
class ProjectorBase
{
public:

    /**
    * @brief Returns the degrees of freedom for the expansion function space.
    * @return Source degrees of freedom.
    */
    virtual OperatorDof src_dof() const = 0;


    /**
    * @brief Computes projector values for the given observation and source triangles.
    * @param[in] k - Complex wavenumber.
    * @param[in] obs_points - Observation coordinates on which to project the generated field.
    * @param[in] src_tri - Source triangle.
    * @return Projected field components at each observation point, for each source degree of freedom.
    * @details
    * Rows of the output matrix correspond to field components and observation points, and columns
    * correspond to source degrees of freedom. For vector fields, the rows are ordered first by field
    * component, and then by observation point, e.g., \f$ (F_{x1}, F_{y1}), (F_{x2}, F_{y2}), \ldots \f$,
    * where \f$ (F_{xi}, F_{yi}) \f$ are the components of a two-dimensional vector field \f$ \vec{F} \f$
    * defined at the observation point \f$ (x_i, y_i) \f$.
    */
    virtual EigMat<Complex> compute(
        const Complex k,
        ConstEigRef<EigMatNX<Float, 3>> obs_points,
        const Triangle<3>& src_tri
        ) = 0;


    /**
    * @brief Transforms the coordinates of the observation points and source triangle into a local
    * coordinate system defined by the source triangle.
    * @param[out] obs_points_local - Observation points in the source's local coordinate system.
    * @param[out] src_tri_local - Source triangle in its local coordinate system.
    * @param[in] obs_points - Original observation points.
    * @param[in] src_tri - Original source triangle.
    */
    static void transform_coordinates(
        EigMatNX<Float, 3>& obs_points_local,
        Triangle<2>& src_tri_local,
        ConstEigRef<EigMatNX<Float, 3>> obs_points,
        const Triangle<3>& src_tri
        )
    {
        EigMatMN<Float, 3, 3> local_uvw = src_tri.local_coordinate_basis();
        EigColVecN<Float, 3> local_origin = src_tri.local_origin();

        obs_points_local = GeometryOps<3>::transform_coordinate_system(
            obs_points, local_origin, local_uvw
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
    * @brief Virtual destructor.
    */
    virtual ~ProjectorBase() = default;

};

/**
* @}
*/

}

#endif
