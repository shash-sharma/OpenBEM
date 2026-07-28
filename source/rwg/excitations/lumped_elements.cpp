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

#include "rwg/excitations/lumped_elements.hpp"

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

Index LumpedElement::num_port_elems() const
{
    Index num_elems = 0;
    for (Index ii = 0; ii < num_ports(); ++ii)
        for (Index term: ports()[ii])
            num_elems += terminals()[term].elem_inds().size();
    return num_elems;
}


Index LumpedElement::num_port_elems(Index idx) const
{
    Index num_elems = 0;
    for (Index term: ports()[idx])
        num_elems += terminals()[term].elem_inds().size();
    return num_elems;
}


MeshView<TriangleMesh<3>> LumpedElement::port_mesh_view() const
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
    return MeshView<TriangleMesh<3>> (structure_.mesh(), elem_inds, "port_mesh");
}


MeshView<TriangleMesh<3>> LumpedElement::port_mesh_view(Index idx) const
{
    EigRowVec<Index> elem_inds = EigRowVec<Index>::Zero(1, num_port_elems(idx));
    Index col = 0;
    for (Index term: ports()[idx])
    {
        elem_inds.middleCols(col, terminals()[term].elem_inds().size()) = terminals()[term].elem_inds();
        col += terminals()[term].elem_inds().size();
    }
    return MeshView<TriangleMesh<3>> (structure_.mesh(), elem_inds, "port_mesh");
}


void LumpedElement::get_exc_matrix(MatrixBase<Complex>& mat, const Float f) const
{
    mat.resize(num_ports(), num_ports());
    mat.set_identity();
    return;
};


void LumpedElement::get_current_mapping_matrix(MatrixBase<Complex>& mat) const
{
    mat.resize(num_port_elems(), num_ports());

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

    return;
};


void LumpedElement::get_voltage_mapping_matrix(MatrixBase<Complex>& mat) const
{
    mat.resize(num_ports(), num_port_elems());

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

    return;
};


void LumpedElement::get_impedance_mapping_matrix(MatrixBase<Complex>& mat) const
{
    mat.resize(num_ports(), num_ports());
    for (Index ii = 0; ii < num_ports(); ++ii)
        mat.set_value(ii, ii, impedances()[ii]);
    mat.assemble();
    return;
};


void LumpedElement::get_terminal_mapping_matrix(MatrixBase<Complex>& mat) const
{
    MeshView<TriangleMesh<3>> view = port_mesh_view();

    mat.resize(structure_.mesh().num_elems(), view.elem_inds().size());

    for (Index ii = 0; ii < view.elem_inds().size(); ++ii)
        mat.set_value(view.elem_inds()[ii], ii, one);
    mat.assemble();

    return;
};


void LumpedElement::get_port_mapping_matrix(MatrixBase<Complex>& mat) const
{
    std::unique_ptr<MatrixBase<Complex>> Dt = mat.clone();
    get_terminal_mapping_matrix(*Dt);

    std::unique_ptr<MatrixBase<Complex>> DtT = mat.clone();
    DtT->set_transpose(*Dt);

    std::unique_ptr<MatrixBase<Complex>> Dp = mat.clone();
    get_voltage_mapping_matrix(*Dp);

    Dp->matmul(mat, *DtT);

    return;
};


Float LumpedElement::terminal_area(const MeshView<TriangleMesh<3>>& terminal) const
{
    Float area = 0;
    for (const Index elem: terminal.elem_inds())
        area += structure_.mesh().elem_primitive(elem).area();
    return area;
};


void LumpedElement::check_impedances()
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


void LumpedElement::set_terminals_from_polygons(
    const std::vector<EigMatNX<Float, 3>>& terminal_polygons,
    const bool single_element
    )
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

}
