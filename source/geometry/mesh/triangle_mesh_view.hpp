// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Lightweight view into a triangular mesh.
*/

#ifndef GEOM_TRIANGLE_MESH_VIEW_H
#define GEOM_TRIANGLE_MESH_VIEW_H

#include <string>
#include <set>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"


namespace bem
{

/**
* \ingroup mesh
* @{
*/

/**
* @brief Class that provides a lightweight view into a `TriangleMesh` object.
* @tparam dim - The dimension of the mesh (2 or 3).
*/
template <uint8_t dim>
class TriangleMeshView
{
public:

    /**
    * @brief Constructs a `TriangleMeshView` from a mesh and specified face indices.
    * @param[in] mesh - The parent mesh into which this is a view.
    * @param[in] face_inds - Indices of the faces of `mesh` to include in the view.
    * @param[in] name - Name of the view (optional).
    */
    TriangleMeshView(
        const TriangleMesh<dim>& mesh,
        ConstEigRef<EigRowVec<Index>> face_inds,
        const std::string name = "view"
        ): mesh_(mesh), name_(name), face_inds_(face_inds)
    {
        set_edge_inds();
        return;
    };


    /**
    * @brief Constructs a `TriangleMeshView` from a mesh, containing all its faces.
    * @param[in] mesh - The parent mesh into which this is a view.
    * @param[in] name - Name of the view (optional).
    */
    TriangleMeshView(
        const TriangleMesh<dim>& mesh,
        const std::string name = "view"
        ):
        TriangleMeshView(
            mesh,
            EigRowVec<Index>::LinSpaced(mesh.num_faces(), 0, mesh.num_faces() - 1),
            name
            ) {};


    /**
    * @brief Returns the parent mesh into which this is a view.
    * @return Parent mesh.
    */
    const TriangleMesh<dim>& parent_mesh() const
    { return mesh_; };


    /**
    * @brief Creates and returns a new, self-contained mesh containing the faces of the parent
    * mesh associated with this view.
    * @return Mesh object containing the faces of the parent mesh associated with this view.
    */
    TriangleMesh<dim> mesh() const
    {
        return mesh_.partition_by_faces(face_inds_);
    };


    /**
    * @brief Returns the parent mesh's face indices associated with this view.
    * @return Parent mesh's face indices associated with this view.
    */
    const EigRowVec<Index>& face_inds() const
    { return face_inds_; };


    /**
    * @brief Returns the parent mesh's edge indices associated with this view.
    * @return Parent mesh's edge indices associated with this view.
    */
    const EigRowVec<Index>& edge_inds() const
    { return edge_inds_; };


    /**
    * @brief Returns the number of faces in the view.
    * @return Number of faces.
    */
    Index num_faces() const
    { return face_inds_.size(); };


    /**
    * @brief Returns the number of edges in the view.
    * @return Number of edges.
    */
    Index num_edges() const
    { return edge_inds_.size(); };


    /**
    * @brief Returns the name of the view.
    * @return Name of the view.
    */
    const std::string& name() const
    { return name_; };


protected:

    /**
    * @brief Collects the parent mesh's edge indices associated with this view's faces.
    */
    void set_edge_inds()
    {
        std::set<Index> unique_edges;
        for (Index ii = 0; ii < face_inds_.size(); ++ii)
            for (uint8_t jj = 0; jj < 3; ++jj)
                unique_edges.insert(mesh_.face_edges()(jj, face_inds_[ii]));

        edge_inds_.resize(1, unique_edges.size());

        Index kk = 0;
        for (const Index edge: unique_edges)
            edge_inds_[kk++] = edge;

        return;
    };


    const TriangleMesh<dim>& mesh_;
    const std::string name_ = "view";
    const EigRowVec<Index> face_inds_;
    EigRowVec<Index> edge_inds_;

};

/**
* @}
*/

}

#endif