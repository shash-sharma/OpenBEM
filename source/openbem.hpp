// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Collection of all OpenBEM headers.
*/

#ifndef BEM_HEADERS_H
#define BEM_HEADERS_H

#include "types.hpp"
#include "constants.hpp"
#include "materials.hpp"

#include "geometry/component.hpp"
#include "geometry/structure.hpp"
#include "geometry/operations.hpp"
#include "geometry/point_cloud.hpp"

#include "geometry/primitives/edge.hpp"
#include "geometry/primitives/triangle.hpp"

#include "geometry/mesh/base.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/mesh/mesh_transfer.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_base.hpp"
#include "matrix/eigen_dense.hpp"
#include "matrix/eigen_sparse.hpp"

#include "kernels/base.hpp"
#include "kernels/hgf.hpp"

#include "rwg/function_space.hpp"

#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"
#include "rwg/integrators/src/singularity.hpp"
#include "rwg/integrators/src/line.hpp"
#include "rwg/integrators/src/strategic.hpp"

#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/quadrature.hpp"
#include "rwg/integrators/obs/strategic.hpp"

#include "rwg/integral_equations/base.hpp"
#include "rwg/integral_equations/tefie.hpp"
#include "rwg/integral_equations/nefie.hpp"
#include "rwg/integral_equations/tmfie.hpp"
#include "rwg/integral_equations/nmfie.hpp"
#include "rwg/integral_equations/aefie.hpp"
#include "rwg/integral_equations/continuity.hpp"

#include "rwg/operators/base.hpp"
#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"
#include "rwg/operators/incidence.hpp"
#include "rwg/operators/generic.hpp"
#include "rwg/operators/vector_ops.hpp"

#include "rwg/projectors/base.hpp"
#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "rwg/excitations/base.hpp"
#include "rwg/excitations/plane_wave.hpp"
#include "rwg/excitations/inf_gap.hpp"

#include "rwg/assemblers/base.hpp"
#include "rwg/assemblers/operator_matrix.hpp"
#include "rwg/assemblers/excitation_matrix.hpp"
#include "rwg/assemblers/projector_matrix.hpp"

#include "rwg/lumped_elements/base.hpp"
#include "rwg/lumped_elements/efie.hpp"
#include "rwg/lumped_elements/aefie.hpp"

#endif
