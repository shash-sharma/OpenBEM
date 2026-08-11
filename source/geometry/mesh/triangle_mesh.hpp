// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Triangular mesh functionality.
*/

#ifndef GEOM_TRIANGLE_MESH_H
#define GEOM_TRIANGLE_MESH_H

#include "types.hpp"
#include "geometry/mesh/base.hpp"
#include "geometry/primitives/triangle.hpp"


namespace bem
{

/**
* \ingroup mesh
* @{
*/

/**
* @brief Class defining a mesh with triangle elements.
* @tparam dim - The dimension of the mesh (2 or 3).
*/
template <uint8_t dim>
class TriangleMesh: public MeshBase<dim, 3>
{

    using base = MeshBase<dim, 3>;

    static_assert((dim == 2 || dim == 3), "TriangleMesh: `dim` must be 2 or 3.");

public:

    /**
    * @brief Constructs an empty `TriangleMesh`.
    */
    TriangleMesh() {};


    /**
    * @brief Constructs a `TriangleMesh` with given vertex and element data.
    * @param[in] verts - Coordinates of the mesh vertices.
    * @param[in] elems - Element-wise triplets of vertex indices.
    * @param[in] decoupled_edges - If true, edges are unique to each element (optional).
    */
    TriangleMesh(
        ConstEigRef<EigMatNX<Float, dim>> verts,
        ConstEigRef<EigMatNX<Index, 3>> elems,
        const bool decoupled_edges = false
        )
    {
        EigRowVec<Index> elem_tags = EigRowVec<Index>::Zero(1, elems.cols());
        set_data(verts, elems, elem_tags, decoupled_edges);
        return;
    };


    /**
    * @brief Constructs a `TriangleMesh` with given vertex and element data, and element tags.
    * @param[in] verts - Coordinates of the mesh vertices.
    * @param[in] elems - Element-wise triplets of vertex indices.
    * @param[in] elem_tags - Element-wise tags.
    * @param[in] decoupled_edges - If true, edges are unique to each element (optional).
    * @details
    * Elements that have the same tag are considered to be part of the same region, for the purpose
    * of defining edges. I.e., an edge between two elements is formed only if they have the same
    * tag.
    */
    TriangleMesh(
        ConstEigRef<EigMatNX<Float, dim>> verts,
        ConstEigRef<EigMatNX<Index, 3>> elems,
        ConstEigRef<EigRowVec<Index>> elem_tags,
        const bool decoupled_edges = false
        )
    {
        set_data(verts, elems, elem_tags, decoupled_edges);
        return;
    };


    /**
    * @brief Sets the mesh data.
    * @param[in] verts - Coordinates of the mesh vertices.
    * @param[in] elems - Element-wise triplets of vertex indices.
    * @param[in] elem_tags - Element-wise tags.
    * @param[in] decoupled_edges - If true, edges are unique to each element (optional).
    * @details
    * Elements that have the same tag are considered to be part of the same region, for the purpose
    * of defining edges. I.e., an edge between two elements is formed only if they have the same
    * tag.
    */
    void set_data(
        ConstEigRef<EigMatNX<Float, dim>> verts,
        ConstEigRef<EigMatNX<Index, 3>> elems,
        ConstEigRef<EigRowVec<Index>> elem_tags,
        const bool decoupled_edges = false
        );


    /**
    * @brief Returns index pairs of vertices of each edge in the mesh.
    * @return Edge-wise pairs of vertex indices.
    */
    const EigMatNX<Index, 2>& edges() const
    { return edges_; };


    /**
    * @brief Returns the edge indices of each element in the mesh.
    * @return Element-wise triplets of edge indices.
    */
    const EigMatNX<Index, 3>& elem_edges() const
    { return elem_edges_; };


    /**
    * @brief Returns the element indices of each edge in the mesh.
    * @return Edge-wise pairs of element indices ordered as (plus, minus) polarity.
    */
    const EigMatNX<Index, 2>& edge_elems() const
    { return edge_elems_; };


    /**
    * @brief Returns the polarities of the edges of each element.
    * @return Element-wise triplets of edge polarities.
    */
    const EigMatNX<Float, 3>& elem_edge_polarities() const
    { return elem_edge_polarities_; };


    /**
    * @brief Returns the indices of boundary elements in the mesh.
    * @return Indices of boundary elements.
    */
    const EigRowVec<Index>& boundary_elems() const
    { return boundary_elems_; };


    /**
    * @brief Returns the indices of junction elements in the mesh.
    * @return Indices of junction elements.
    */
    const EigRowVec<Index>& junction_elems() const
    { return junction_elems_; };


    /**
    * @brief Returns the indices of internal elements in the mesh.
    * @return Indices of internal elements.
    */
    const EigRowVec<Index>& internal_elems() const
    { return internal_elems_; };


    /**
    * @brief Returns the indices of boundary edges in the mesh.
    * @return Indices of boundary edges.
    */
    const EigRowVec<Index>& boundary_edges() const
    { return boundary_edges_; };


    /**
    * @brief Returns the indices of junction edges in the mesh.
    * @return Indices of junction edges.
    */
    const EigRowVec<Index>& junction_edges() const
    { return junction_edges_; };


    /**
    * @brief Returns the indices of internal edges in the mesh.
    * @return Indices of internal edges.
    */
    const EigRowVec<Index>& internal_edges() const
    { return internal_edges_; };


    /**
    * @brief Returns the number of edges in the mesh.
    * @return Number of edges.
    */
    Index num_edges() const
    { return edges_.cols(); };


    /**
    * @brief Returns a sub-mesh that contains only specified elements of this mesh.
    * @param[out] partition - Partitioned mesh object containing the specified elements.
    * @param[in] elem_inds - Indices of elements to keep in the sub-mesh.
    */
    void partition_by_elems(
        MeshBase<dim, 3>& partition,
        ConstEigRef<EigRowVec<Index>> elem_inds
        ) const;


    /**
    * @brief Returns a sub-mesh that contains only elements of this mesh that lie within a given bounding box.
    * @param[out] partition - Partitioned mesh object containing elements within the given bounding box.
    * @param[in] bbox - Bounding box defined by two corners (min and max).
    * @param[in] strict - If true, only elements fully contained within the bounding box are included (optional).
    */
    void partition_by_bbox(
        MeshBase<dim, 3>& partition,
        ConstEigRef<EigMatMN<Float, dim, 2>> bbox,
        const bool strict = true
        ) const;


    /**
    * @brief Returns a `Triangle` primitive object representing a specific element of the mesh.
    * @param[in] elem - Index of the element.
    * @return `Triangle` object representing the specified element.
    */
    Triangle<dim> elem_primitive(Index elem) const
    {
        return Triangle<dim> (
            base::verts()(Eigen::placeholders::all, base::elems(elem)),
            elem_edge_polarities().col(elem),
            base::elem_tags(elem),
            elem
            );
    };


    /**
    * @brief Reverses the orientation of each element.
    */
    virtual void reverse_orientation()
    {
        base::elems_ = base::elems_.colwise().reverse().eval();
        generate_edges();
        return;
    };


protected:

    /**
    * @brief Generates edges for the mesh based on element vertices.
    * @details
    * Edges are generated tag-wise. Tags represent distinct mesh regions, and edges cannot be
    * associated with two elements that belong to different mesh regions. For example, if two mesh
    * regions are in contact with one another, each region will have its own edges, rather than
    * creating artificial junctions at the point of contact.
    */
    void generate_edges();


    bool decoupled_edges_ = false;

    EigMatNX<Index, 2> edges_;
    EigMatNX<Index, 3> elem_edges_;
    EigMatNX<Float, 3> elem_edge_polarities_;

    EigMatNX<Index, 2> edge_elems_;

    EigRowVec<Index> boundary_elems_;
    EigRowVec<Index> junction_elems_;
    EigRowVec<Index> internal_elems_;

    EigRowVec<Index> boundary_edges_;
    EigRowVec<Index> junction_edges_;
    EigRowVec<Index> internal_edges_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "geometry/mesh/triangle_mesh.cpp"
#endif

#endif
