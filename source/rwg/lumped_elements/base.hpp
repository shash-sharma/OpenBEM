// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base class for computing coupling matrices for lumped elements.
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
* @tparam MatrixType - Matrix type, must derive from `MatrixBase<Complex>`.
*/
template <typename MatrixType = EigenMatrix<Complex>>
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
            terminals_(terminals),
            terminal_components_(terminal_components),
            ports_(ports),
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
    Index num_port_elems() const
    {
        Index num_elems = 0;
        for (Index ii = 0; ii < num_ports(); ++ii)
            for (Index term: ports()[ii])
                num_elems += terminals()[term].elem_inds().size();
        return num_elems;
    }


    /**
    * @brief Returns the number of mesh elements associated with a given port.
    * @return Number of port mesh elements.
    */
    Index num_port_elems(Index idx) const
    {
        Index num_elems = 0;
        for (Index term: ports()[idx])
            num_elems += terminals()[term].elem_inds().size();
        return num_elems;
    }


    /**
    * @brief Returns a mesh view consisting of all elements associated with the ports.
    * @return Port mesh view.
    */
    MeshView<TriangleMesh<3>> port_mesh_view() const
    {
        EigRowVec<Index> elem_inds = EigRowVec<Index>::Zero(1, num_port_elems());
        Index col = 0;
        for (Index ii = 0; ii < num_ports(); ++ii)
        {
            for (Index term: ports()[ii])
            {
                elem_inds.middleCols(col, terminals()[term].elem_inds().size()) = terminals()[term].elem_inds();
                col += terminals()[term].elem_inds().size();
            }
        }
        return MeshView(structure_.mesh(), elem_inds, "port_mesh");
    }


    /**
    * @brief Returns a mesh view consisting of all elements associated with a given port.
    * @return Port mesh view.
    */
    MeshView<TriangleMesh<3>> port_mesh_view(Index idx) const
    {
        EigRowVec<Index> elem_inds = EigRowVec<Index>::Zero(1, num_port_elems(idx));
        Index col = 0;
        for (Index term: ports()[idx])
        {
            elem_inds.middleCols(col, terminals()[term].elem_inds().size()) = terminals()[term].elem_inds();
            col += terminals()[term].elem_inds().size();
        }
        return MeshView(structure_.mesh(), elem_inds, "port_mesh");
    }


    /**
    * @brief Returns the right-hand side excitation matrix to set a voltage source at each lumped element.
    * @param[in] f - Frequency in Hz.
    * @return Excitation matrix.
    */
    virtual MatrixType exc_matrix(const Float f) const
    {
        MatrixType mat (num_ports(), num_ports());
        mat.set_identity();
        return mat;
    };


    /**
    * @brief Returns the matrix that maps the total port currents to the volume current densities on terminal triangles.
    * @return Mapping matrix.
    */
    virtual MatrixType current_mapping_matrix() const
    {
        MatrixType mat (num_port_elems(), num_ports());

        Index row = 0;
        for (Index ii = 0; ii < num_ports(); ++ii)
        {
            Float sign = one;
            for (Index term: ports()[ii])
            {
                Float term_area = terminal_area(terminals()[term]);

                for (Index jj = 0; jj < terminals()[term].elem_inds().size(); ++jj)
                {
                    Index elem = terminals()[term].elem_inds()[jj];
                    mat.set_value(row++, ii, sign * structure_.mesh().elem_primitive(elem).area() / term_area);
                }

                sign *= -one;
            }
        }

        mat.assemble();

        return mat;
    };


    /**
    * @brief Returns the matrix that maps the scalar potential on terminal triangles to the average voltage on the port.
    * @return Mapping matrix.
    */
    virtual MatrixType voltage_mapping_matrix() const
    {
        MatrixType mat (num_ports(), num_port_elems());

        Index col = 0;
        for (Index ii = 0; ii < num_ports(); ++ii)
        {
            Float sign = one;
            for (Index term: ports()[ii])
            {
                Float term_area = terminal_area(terminals()[term]);

                for (Index jj = 0; jj < terminals()[term].elem_inds().size(); ++jj)
                {
                    Index elem = terminals()[term].elem_inds()[jj];
                    mat.set_value(ii, col++, sign * structure_.mesh().elem_primitive(elem).area() / term_area);
                }

                sign *= -one;
            }
        }

        mat.assemble();

        return mat;
    };


    /**
    * @brief Returns the matrix associated with port load impedances.
    * @return Load matrix.
    */
    virtual MatrixType impedance_mapping_matrix() const
    {
        MatrixType mat (num_ports(), num_ports());
        for (Index ii = 0; ii < num_ports(); ++ii)
            mat.set_value(ii, ii, impedances()[ii]);
        mat.assemble();
        return mat;
    };


    /**
    * @brief Returns the matrix that maps terminal triangles to their parent mesh triangles.
    * @return Mapping matrix.
    */
    virtual MatrixType terminal_mapping_matrix() const
    {
        MeshView<TriangleMesh<3>> view = port_mesh_view();

        MatrixType mat (structure_.mesh().num_elems(), view.elem_inds().size());

        for (Index ii = 0; ii < view.elem_inds().size(); ++ii)
            mat.set_value(view.elem_inds()[ii], ii, one);
        mat.assemble();

        return mat;
    };


    /**
    * @brief Returns the matrix that maps ports to mesh triangles, signed based on the terminal.
    * @return Mapping matrix.
    */
    virtual MatrixType port_mapping_matrix() const
    {
        MatrixType DtT;
        DtT.set_transpose(terminal_mapping_matrix());

        MatrixType mat;
        mat.set_matmul(voltage_mapping_matrix(), DtT);

        return mat;
    };


    /**
    * @brief Returns the total area spanned by the mesh triangles associated with a given terminal.
    * @param[in] terminal - Terminal mesh view.
    * @return Terminal area.
    */
    Float terminal_area(const MeshView<TriangleMesh<3>>& terminal) const
    {
        Float area = 0;
        for (const Index elem: terminal.elem_inds())
            area += structure_.mesh().elem_primitive(elem).area();
        return area;
    };


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
    void check_impedances()
    {
        if (ports_.size() != impedances_.size() && impedances_.size() != 1)
            throw std::invalid_argument(
                "LumpedElement(): Number of `impedances` must equal number of `ports`, or be a single value."
                );
        if (impedances_.size() == 1 && ports_.size() > 1)
        {
            Complex imp = impedances_[0];
            impedances_.resize(ports_.size(), imp);
        }
        return;
    };


    /**
    * @brief Populates `terminals_` by finding mesh elements inside each terminal polygon.
    * @param[in] terminal_polygons - Terminal polygons.
    * @param[in] single_element - If true, only the mesh element closest to the terminal polygon centroid is kept (optional).
    * @details
    * A triangle is considered part of a terminal if it is coplanar with the terminal polygon, and
    * its centroid lies on or within the terminal boundary.
    */
    void set_terminals_from_polygons(const std::vector<EigMatNX<Float, 3>>& terminal_polygons, const bool single_element = false)
    {
        terminals_.clear();
        terminal_components_.resize(terminal_polygons.size());

        // Find all mesh triangles whose centroid lies within the terminal polygon
        for (Index ii = 0; ii < terminal_polygons.size(); ++ii)
        {
            std::vector<Index> term_elems;

            for (Index jj = 0; jj < structure_.components().size(); ++jj)
            {
                // TODO: for efficiency, first check if the polygon is in the component's bounding box

                for (Index kk = 0; kk < structure_.components()[jj].mesh_view().elem_inds().size(); ++kk)
                {
                    Triangle<3> tri = structure_.mesh().elem_primitive(
                        structure_.components()[jj].mesh_view().elem_inds()[kk]
                        );

                    Triangle<3> poly_tri (terminal_polygons[ii].leftCols(3));

                    if (!GeometryOps<3>::check_coplanar_triangles(tri, poly_tri))
                        continue;

                    if (!GeometryOps<3>::point_in_polygon(tri.centroid(), terminal_polygons[ii]))
                        continue;

                    term_elems.push_back(structure_.components()[jj].mesh_view().elem_inds()[kk]);
                    terminal_components_[ii].push_back(jj);
                }
            }

            if (term_elems.size() == 0)
                throw std::runtime_error(
                    "LumpedElement::set_terminals_from_polygons(): No mesh triangles found for terminal " + std::to_string(ii)
                    );

            // Keep only the mesh triangle closest to the terminal polygon centroid
            if (single_element)
            {
                Float offset = 1e30;
                Index term_elem;
                EigColVecN<Float, 3> term_centroid =
                    terminal_polygons[ii].rowwise().sum() / terminal_polygons[ii].cols();

                for (Index jj = 0; jj < term_elems.size(); ++jj)
                {
                    Float dist = (
                        structure_.mesh().elem_primitive(term_elems[jj]).centroid() - term_centroid
                        ).norm();

                    if (dist < offset)
                    {
                        offset = dist;
                        term_elem = jj;
                    }
                }
                term_elems = { term_elems[term_elem] };
                terminal_components_[ii] = { terminal_components_[ii][term_elem] };
            }

            std::sort(terminal_components_[ii].begin(), terminal_components_[ii].end());
            auto iter = std::unique(terminal_components_[ii].begin(), terminal_components_[ii].end());
            terminal_components_[ii].erase(iter, terminal_components_[ii].end());

            EigRowVec<Index> elem_inds = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (term_elems.data(), term_elems.size());
            MeshView<TriangleMesh<3>> view (structure_.mesh(), elem_inds, "terminal_" + std::to_string(ii));
            terminals_.push_back(view);
        }

        return;
    };


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

#endif
