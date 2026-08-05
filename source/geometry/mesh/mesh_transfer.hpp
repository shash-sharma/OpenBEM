// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Mesh input and output.
*/

#ifndef GEOM_MESH_IO_H
#define GEOM_MESH_IO_H

#include <string>

#include "types.hpp"


namespace bem
{

// Forward declarations
template <typename MeshType> class Structure;
template <uint8_t dim> class TriangleMesh;

/**
* \ingroup mesh
* @{
*/

/**
* @brief Class that collects mesh input and output static methods.
*/
class MeshTransfer
{
public:

    /**
     * @brief Reads a GMSH v2 mesh file and populates a `TriangleMesh` in a `Structure`.
     * @param[out] structure - The `Structure` to populate with mesh data.
     * @param[in] msh_filename - Path to the GMSH v2 mesh file.
     * @param[in] decoupled_edges - If true, edges are unique to each face (optional).
     * @details
     * It is assumed that all physical surfaces are numbered consecutively from 1 onwards.
     */
    static void read_gmsh_v2(
        Structure<TriangleMesh<3>>& structure,
        const std::string msh_filename,
        const bool decoupled_edges = false
        );


    /**
     * @brief Writes a GMSH v2 mesh file from a `TriangleMesh` in a `Structure`.
     * @param[in] structure - The `Structure` whose mesh is written to file.
     * @param[in] msh_filename - Path to the output GMSH v2 mesh file.
     * @param[in] extension - Custom file extension for the output mesh file (optional).
     */
    static void write_gmsh_v2(
        const Structure<TriangleMesh<3>>& structure,
        const std::string msh_filename,
        const std::string extension = "msh"
        );


    /**
     * @brief Writes a GMSH v2 mesh file from a `TriangleMesh` in a `Structure` with a superimposed scalar field.
     * @param[in] structure - The `Structure` associated with the scalar field.
     * @param[in] msh_filename - Path to the output GMSH v2 mesh file.
     * @param[in] field - Scalar field to superimpose on the mesh.
     * @param[in] field_name - Name for the field (optional).
     * @param[in] field_plot_type - Either "nodes" (one value per mesh vertex, written as `$NodeData`,
     * which GMSH interpolates across each element for a continuous-looking plot) or "elements" (one
     * value per mesh element, written as `$ElementData`, discontinuous across element boundaries)
     * (optional).
     * @details
     * `field` must have one entry per mesh vertex if `field_plot_type` is "nodes", or one entry per
     * mesh element if "elements".
     */
    static void write_gmsh_v2_scalar_field(
        const Structure<TriangleMesh<3>>& structure,
        const std::string msh_filename,
        ConstEigRef<EigRowVec<Float>> field,
        std::string field_name = "scalar_field",
        const std::string field_plot_type = "nodes"
        );


    /**
     * @brief Writes a GMSH v2 mesh file from a `TriangleMesh` in a `Structure` with a superimposed vector field.
     * @param[in] structure - The `Structure` associated with the vector field.
     * @param[in] msh_filename - Path to the output GMSH v2 mesh file.
     * @param[in] field - Vector field to superimpose on the mesh.
     * @param[in] field_name - Name for the field (optional).
     * @param[in] field_plot_type - Either "nodes" (one value per mesh vertex, written as `$NodeData`,
     * which GMSH interpolates across each element for a continuous-looking plot) or "elements" (one
     * value per mesh element, written as `$ElementData`, discontinuous across element boundaries)
     * (optional).
     * @details
     * `field` must have one column per mesh vertex if `field_plot_type` is "nodes", or one column per
     * mesh element if "elements".
     */
    static void write_gmsh_v2_vector_field(
        const Structure<TriangleMesh<3>>& structure,
        const std::string msh_filename,
        ConstEigRef<EigMatNX<Float, 3>> field,
        std::string field_name = "vector_field",
        const std::string field_plot_type = "nodes"
        );

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "geometry/mesh/mesh_transfer.cpp"
#endif

#endif
