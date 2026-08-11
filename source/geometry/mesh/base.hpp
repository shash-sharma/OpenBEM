// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Mesh base and lightweight view classes.
*/

#ifndef GEOM_MESH_BASE_H
#define GEOM_MESH_BASE_H

#include <string>

#include "types.hpp"


namespace bem
{

/**
* \ingroup mesh
* @{
*/

/**
* @brief Mesh base class.
* @tparam dim - The dimension of the mesh (1, 2, or 3).
* @tparam verts_per_elem - Number of vertices per element (e.g., 2 for edges, 3 for triangles, 4 for tetrahedra).
*/
template <uint8_t dim, uint8_t verts_per_elem>
class MeshBase
{

    static_assert((dim == 1 || dim == 2 || dim == 3), "MeshBase: `dim` must be 1, 2, or 3.");

public:

    /**
    * @brief Returns the coordinates of the mesh vertices.
    * @return Vertex coordinates.
    */
    const EigMatNX<Float, dim>& verts() const
    { return verts_; };


    /**
    * @brief Returns the coordinates of a specific vertex.
    * @param[in] vert - Vertex index.
    * @return Coordinates of the vertex.
    */
    EigColVecN<Float, dim> verts(Index vert) const
    { return verts_.col(vert); };


    /**
    * @brief Returns the vertex indices of each element.
    * @return Element-wise triplets of vertex indices.
    */
    const EigMatNX<Index, verts_per_elem>& elems() const
    { return elems_; };


    /**
    * @brief Returns the vertex indices of a specific element.
    * @param[in] elem - Element index.
    * @return Vertex indices of the element.
    */
    EigColVecN<Index, 3> elems(Index elem) const
    { return elems_.col(elem); };


    /**
    * @brief Returns the index of a specific vertex of a specific element.
    * @param[in] vert - Vertex index.
    * @param[in] elem - Element index.
    * @return Index of the specified vertex in the specified element.
    */
    Index elems(uint8_t vert, Index elem) const
    { return elems_(vert, elem); };


    /**
    * @brief Returns the element tags.
    * @return Element-wise tags.
    * @details
    * Elements that have the same tag are considered to be part of the same region, for the purpose
    * of defining edges. I.e., an edge between two elements is formed only if they have the same
    * tag.
    */
    const EigRowVec<Index>& elem_tags() const
    { return elem_tags_; };


    /**
    * @brief Returns the tag associated with a specified element.
    * @return Element tag.
    * @details
    * Elements that have the same tag are considered to be part of the same region, for the purpose
    * of defining edges. I.e., an edge between two elements is formed only if they have the same
    * tag.
    */
    Index elem_tags(Index elem) const
    { return elem_tags_[elem]; };


    /**
    * @brief Returns the number of vertices in the mesh.
    * @return Number of vertices.
    */
    Index num_verts() const
    { return verts_.cols(); };


    /**
    * @brief Returns the number of elements in the mesh.
    * @return Number of elements.
    */
    Index num_elems() const
    { return elems_.cols(); };


    /**
    * @brief Returns a sub-mesh that contains only specified elements of this mesh.
    * @param[out] partition - Partitioned mesh object containing the specified elements.
    * @param[in] elem_inds - Indices of elements to keep in the sub-mesh.
    */
    virtual void partition_by_elems(
        MeshBase<dim, verts_per_elem>& partition,
        ConstEigRef<EigRowVec<Index>> elem_inds
        ) const = 0;


    /**
    * @brief Computes and returns the centroid of each element.
    * @return Element-wise centroids.
    */
    EigMatNX<Float, dim> elem_centroids() const
    {
        EigMatNX<Float, dim> centroids = EigMatNX<Float, dim>::Zero(dim, elems_.cols());
        for (Index ii = 0; ii < elems_.cols(); ++ii)
        {
            EigMatMN<Float, dim, verts_per_elem> coords = verts_(
                Eigen::placeholders::all, elems_.col(ii)
                );
            centroids(Eigen::placeholders::all, ii) = coords.rowwise().mean();
        }
        return centroids;
    };


    /**
    * @brief Reverses the orientation of each element.
    */
    virtual void reverse_orientation()
    {
        elems_ = elems_.colwise().reverse().eval();
        return;
    };


    /**
    * @brief Virtual destructor.
    */
    virtual ~MeshBase() = default;


protected:

    EigMatNX<Float, dim> verts_;
    EigMatNX<Index, verts_per_elem> elems_;
    EigRowVec<Index> elem_tags_;

};


/**
* @brief Class that provides a lightweight view into a `MeshBase` object.
* @tparam MeshType - Type of the mesh.
*/
template <typename MeshType>
class MeshView
{
public:

    /**
    * @brief Constructs a `MeshView` from a mesh and specified element indices.
    * @param[in] mesh - The parent mesh into which this is a view.
    * @param[in] elem_inds - Indices of the elements of `mesh` to include in the view.
    * @param[in] name - Name of the view (optional).
    */
    MeshView(
        const MeshType& mesh,
        ConstEigRef<EigRowVec<Index>> elem_inds,
        const std::string name = "view"
        ): mesh_(mesh), elem_inds_(elem_inds), name_(name) {};


    /**
    * @brief Constructs a `MeshView` from a mesh, containing all its elements.
    * @param[in] mesh - The parent mesh into which this is a view.
    * @param[in] name - Name of the view (optional).
    */
    MeshView(
        const MeshType& mesh,
        const std::string name = "view"
        ):
        MeshView(
            mesh,
            EigRowVec<Index>::LinSpaced(mesh.num_elems(), 0, mesh.num_elems() - 1),
            name
            ) {};


    /**
    * @brief Creates and returns a new mesh containing the elements of the parent mesh associated with this view.
    * @return Mesh object containing the elements of the parent mesh associated with this view.
    */
    MeshType mesh() const
    {
        MeshType submesh;
        mesh_.partition_by_elems(submesh, elem_inds_);
        return submesh;
    };


    /**
    * @brief Returns the parent mesh's element indices associated with this view.
    * @return Parent mesh's element indices associated with this view.
    */
    const EigRowVec<Index>& elem_inds() const { return elem_inds_; };


    /**
    * @brief Returns the name of the view.
    * @return Name of the view.
    */
    const std::string& name() const { return name_; };


protected:

    const MeshType& mesh_;
    const EigRowVec<Index> elem_inds_;
    const std::string name_ = "view";
    
};

/**
* @}
*/

}

#endif
