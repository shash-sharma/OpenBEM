// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Convenience wrapper classes to generate any operator based on a given integer flag.
*/

#ifndef BEM_RWG_OPS_GENERIC_I
#define BEM_RWG_OPS_GENERIC_I

#include <stdexcept>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem::rwg
{

template <typename ObsIntegratorType>
EigMatMN<Complex, 3, 3> GenericRwgOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{

    switch (op_name_)
    {

    case OperatorName::VECTOR_SINGLE_LAYER:
    {
        return vector_single_layer_.compute(k, obs_tri, src_tri);
    }

    case OperatorName::ROT_VECTOR_SINGLE_LAYER:
    {
        return rot_vector_single_layer_.compute(k, obs_tri, src_tri);
    }

    case OperatorName::VECTOR_DOUBLE_LAYER_PV:
    {
        return vector_double_layer_pv_.compute(k, obs_tri, src_tri);
    }

    case OperatorName::ROT_VECTOR_DOUBLE_LAYER_PV:
    {
        return rot_vector_double_layer_pv_.compute(k, obs_tri, src_tri);
    }

    case OperatorName::VECTOR_HYPERSINGULAR:
    {
        return vector_hypersingular_.compute(k, obs_tri, src_tri);
    }

    case OperatorName::ROT_VECTOR_HYPERSINGULAR:
    {
        return rot_vector_hypersingular_.compute(k, obs_tri, src_tri);
    }

    case OperatorName::RWG_RWG:
    {
        return rwg_rwg_.compute(k, obs_tri, src_tri);
    }

    case OperatorName::ROT_RWG_RWG:
    {
        return rot_rwg_rwg_.compute(k, obs_tri, src_tri);
    }

    default:
        throw std::logic_error(
            "GenericRwg: `OperatorName` is invalid or not implemented."
            );
    }

};


template <typename ObsIntegratorType>
EigMatMN<Complex, 1, 1> GenericPulseOp<ObsIntegratorType>::compute(
    const Complex k,
    const Triangle<3>& obs_tri,
    const Triangle<3>& src_tri
    )
{

    switch (op_name_)
    {

    case OperatorName::SCALAR_SINGLE_LAYER:
    {
        return scalar_single_layer_.compute(k, obs_tri, src_tri);
    }

    default:
        throw std::invalid_argument(
            "GenericPulse: `OperatorName` is invalid or not implemented."
            );
    }

};

}

#endif
