// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for computing coupling matrices for lumped elements.
*/

#ifndef BEM_RWG_LUMPED_ELEM_BASE_H
#define BEM_RWG_LUMPED_ELEM_BASE_H

#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>

#include "types.hpp"
#include "geometry/operations.hpp"
#include "geometry/structure.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/mesh/base.hpp"
#include "geometry/mesh/triangle_mesh.hpp"

#include "matrix/base.hpp"
#include "matrix/eigen_matrix.hpp"


namespace bem::rwg
{

/**
* \addtogroup rwgexc
* @{
*/

/**
* @brief Base class for generating excitation coefficients and coupling matrices for lumped ports.
*/
class LumpedElement
{
public:

    /**
    * @brief Constructs a `LumpedElement` object.
    * @param[in] structure - Structure for which the coupling matrices are to be assembled.
    * @param[in] terminals - Mesh views defining all terminals on `mesh`.
    * @param[in] terminal_components - Lists of components associated with each `terminal`.
    * @param[in] ports - List of index pairs into `terminals` that define each lumped element port.
    * @param[in] impedances - Impedance of each lumped element.
    */
    LumpedElement(
        const Structure<TriangleMesh<3>>& structure,
        const std::vector<MeshView<TriangleMesh<3>>>& terminals,
        const std::vector<std::vector<Index>>& terminal_components,
        const std::vector<std::array<Index, 2>>& ports,
        const std::vector<Complex>& impedances
        ):
            structure_(structure),
            ports_(ports),
            terminals_(terminals),
            terminal_components_(terminal_components),
            impedances_(impedances)
    {
        check_impedances();
        return;
    };


    /**
    * @brief Constructs a `LumpedElement` object.
    * @param[in] structure - Structure for which the coupling matrices are to be assembled.
    * @param[in] terminal_polygons - Vertices defining terminal polygons on the `mesh`.
    * @param[in] ports - List of index pairs into `terminals` that define each port.
    * @param[in] impedances - Impedance of each lumped element.
    * @param[in] single_element - If true, only the mesh element closest to the terminal polygon centroid is kept (optional).
    */
    LumpedElement(
        const Structure<TriangleMesh<3>>& structure,
        const std::vector<EigMatNX<Float, 3>>& terminal_polygons,
        const std::vector<std::array<Index, 2>>& ports,
        const std::vector<Complex>& impedances,
        const bool single_element = false
        ):
            structure_(structure),
            terminal_polygons_(terminal_polygons),
            ports_(ports),
            impedances_(impedances)
    {
        check_impedances();
        set_terminals_from_polygons(terminal_polygons_, single_element);
        return;
    };


    /**
    * @brief Returns the number of excitations (right-hand sides), which equals the number of ports.
    * @return Number of excitations (right-hand sides).
    */
    Index num_ports() const
    { return ports().size(); };


    /**
    * @brief Returns the number of mesh elements associated with ports.
    * @return Number of port mesh elements.
    */
    Index num_port_elems() const;


    /**
    * @brief Returns the number of mesh elements associated with a given port.
    * @return Number of port mesh elements.
    */
    Index num_port_elems(Index idx) const;


    /**
    * @brief Returns a mesh view consisting of all elements associated with the ports.
    * @return Port mesh view.
    */
    MeshView<TriangleMesh<3>> port_mesh_view() const;


    /**
    * @brief Returns a mesh view consisting of all elements associated with a given port.
    * @return Port mesh view.
    */
    MeshView<TriangleMesh<3>> port_mesh_view(Index idx) const;


    /**
    * @brief Computes the right-hand side excitation matrix to set a voltage source at each lumped element.
    * @param[out] mat - Output matrix to populate
    * @param[in] f - Frequency in Hz.
    */
    void get_exc_matrix(MatrixBase<Complex>& mat, const Float f) const;


    /**
    * @brief Computes the matrix that maps the total port currents to the volume current densities on terminal triangles.
    * @param[out] mat - Output matrix to populate
    */
    void get_current_mapping_matrix(MatrixBase<Complex>& mat) const;


    /**
    * @brief Computes the matrix that maps the scalar potential on terminal triangles to the average voltage on the port.
    * @param[out] mat - Output matrix to populate
    */
    void get_voltage_mapping_matrix(MatrixBase<Complex>& mat) const;


    /**
    * @brief Computes the matrix associated with port load impedances.
    * @param[out] mat - Output matrix to populate
    */
    void get_impedance_mapping_matrix(MatrixBase<Complex>& mat) const;


    /**
    * @brief Computes the matrix that maps terminal triangles to their parent mesh triangles.
    * @param[out] mat - Output matrix to populate
    */
    void get_terminal_mapping_matrix(MatrixBase<Complex>& mat) const;


    /**
    * @brief Computes the matrix that maps ports to mesh triangles, signed based on the terminal.
    * @param[out] mat - Output matrix to populate
    */
    void get_port_mapping_matrix(MatrixBase<Complex>& mat) const;


    /**
    * @brief Returns the total area spanned by the mesh triangles associated with a given terminal.
    * @param[in] terminal - Terminal mesh view.
    * @return Terminal area.
    */
    Float terminal_area(const MeshView<TriangleMesh<3>>& terminal) const;


    /**
    * @brief Returns a read-only reference to the terminal mesh views.
    */
    const std::vector<MeshView<TriangleMesh<3>>>& terminals() const { return terminals_; };


    /**
    * @brief Returns a read-only reference to the port map.
    */
    const std::vector<std::array<Index, 2>>& ports() const { return ports_; };


    /**
    * @brief Returns a read-only reference to the lumped impedances.
    */
    const std::vector<Complex>& impedances() const { return impedances_; };


    /**
    * @brief Returns a read-only reference to the components associated with each terminal.
    */
    const std::vector<std::vector<Index>>& terminal_components() const { return terminal_components_; };


    /**
    * @brief Virtual destructor.
    */
    virtual ~LumpedElement() = default;


protected:

    /**
    * @brief Ensures correct number of `impedances` are set.
    */
    void check_impedances();


    /**
    * @brief Populates `terminals_` by finding mesh elements inside each terminal polygon.
    * @param[in] terminal_polygons - Terminal polygons.
    * @param[in] single_element - If true, only the mesh element closest to the terminal polygon centroid is kept (optional).
    * @details
    * A triangle is considered part of a terminal if it is coplanar with the terminal polygon, and
    * its centroid lies on or within the terminal boundary.
    */
    void set_terminals_from_polygons(
        const std::vector<EigMatNX<Float, 3>>& terminal_polygons,
        const bool single_element = false
        );


    const Structure<TriangleMesh<3>>& structure_;
    const std::vector<EigMatNX<Float, 3>> terminal_polygons_;
    const std::vector<std::array<Index, 2>> ports_;
    std::vector<MeshView<TriangleMesh<3>>> terminals_;
    std::vector<std::vector<Index>> terminal_components_;
    std::vector<Complex> impedances_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/excitations/lumped_elements.cpp"
#endif

#endif
