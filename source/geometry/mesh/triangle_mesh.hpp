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
#include "geometry/primitives/triangle.hpp"


namespace bem
{

/**
* \ingroup mesh
* @{
*/

/**
* @brief Class defining a mesh with triangle faces.
* @tparam dim - The dimension of the mesh (2 or 3).
*/
template <uint8_t dim>
class TriangleMesh
{

    static_assert((dim == 2 || dim == 3), "TriangleMesh: `dim` must be 2 or 3.");

public:

    /**
    * @brief Constructs an empty `TriangleMesh`.
    */
    TriangleMesh() {};


    /**
    * @brief Constructs a `TriangleMesh` with given vertex and face data.
    * @param[in] vertices - Coordinates of the mesh vertices.
    * @param[in] faces - Face-wise triplets of vertex indices.
    * @param[in] decoupled_edges - If true, edges are unique to each face (optional).
    */
    TriangleMesh(
        ConstEigRef<EigMatNX<Float, dim>> vertices,
        ConstEigRef<EigMatNX<Index, 3>> faces,
        const bool decoupled_edges = false
        )
    {
        EigRowVec<Index> face_tags = EigRowVec<Index>::Zero(1, faces.cols());
        set_data(vertices, faces, face_tags, decoupled_edges);
        return;
    };


    /**
    * @brief Constructs a `TriangleMesh` with given vertex and face data, and face tags.
    * @param[in] vertices - Coordinates of the mesh vertices.
    * @param[in] faces - Face-wise triplets of vertex indices.
    * @param[in] face_tags - Face-wise tags.
    * @param[in] decoupled_edges - If true, edges are unique to each face (optional).
    * @details
    * Faces that have the same tag are considered to be part of the same region, for the purpose
    * of defining edges. I.e., an edge between two faces is formed only if they have the same
    * tag.
    */
    TriangleMesh(
        ConstEigRef<EigMatNX<Float, dim>> vertices,
        ConstEigRef<EigMatNX<Index, 3>> faces,
        ConstEigRef<EigRowVec<Index>> face_tags,
        const bool decoupled_edges = false
        )
    {
        set_data(vertices, faces, face_tags, decoupled_edges);
        return;
    };


    /**
    * @brief Sets the mesh data.
    * @param[in] vertices - Coordinates of the mesh vertices.
    * @param[in] faces - Face-wise triplets of vertex indices.
    * @param[in] face_tags - Face-wise tags.
    * @param[in] decoupled_edges - If true, edges are unique to each face (optional).
    * @details
    * Faces that have the same tag are considered to be part of the same region, for the purpose
    * of defining edges. I.e., an edge between two faces is formed only if they have the same
    * tag.
    */
    void set_data(
        ConstEigRef<EigMatNX<Float, dim>> vertices,
        ConstEigRef<EigMatNX<Index, 3>> faces,
        ConstEigRef<EigRowVec<Index>> face_tags,
        const bool decoupled_edges = false
        );


    /**
    * @brief Returns the coordinates of the mesh vertices.
    * @return Vertex coordinates.
    */
    const EigMatNX<Float, dim>& vertices() const
    { return vertices_; };


    /**
    * @brief Returns the vertex indices of each face.
    * @return Face-wise triplets of vertex indices.
    */
    const EigMatNX<Index, 3>& faces() const
    { return faces_; };


    /**
    * @brief Returns the face tags.
    * @return Face-wise tags.
    * @details
    * Faces that have the same tag are considered to be part of the same region, for the purpose
    * of defining edges. I.e., an edge between two faces is formed only if they have the same
    * tag.
    */
    const EigRowVec<Index>& face_tags() const
    { return face_tags_; };


    /**
    * @brief Returns index pairs of vertices of each edge in the mesh.
    * @return Edge-wise pairs of vertex indices.
    */
    const EigMatNX<Index, 2>& edges() const
    { return edges_; };


    /**
    * @brief Returns the edge indices of each face in the mesh.
    * @return Face-wise triplets of edge indices.
    */
    const EigMatNX<Index, 3>& face_edges() const
    { return face_edges_; };


    /**
    * @brief Returns the face indices of each edge in the mesh.
    * @return Edge-wise pairs of face indices ordered as (plus, minus) polarity.
    */
    const EigMatNX<Index, 2>& edge_faces() const
    { return edge_faces_; };


    /**
    * @brief Returns the polarities of the edges of each face.
    * @return Face-wise triplets of edge polarities.
    */
    const EigMatNX<Float, 3>& face_edge_polarities() const
    { return face_edge_polarities_; };


    /**
    * @brief Returns the indices of boundary faces in the mesh.
    * @return Indices of boundary faces.
    */
    const EigRowVec<Index>& boundary_faces() const
    { return boundary_faces_; };


    /**
    * @brief Returns the indices of junction faces in the mesh.
    * @return Indices of junction faces.
    */
    const EigRowVec<Index>& junction_faces() const
    { return junction_faces_; };


    /**
    * @brief Returns the indices of internal faces in the mesh.
    * @return Indices of internal faces.
    */
    const EigRowVec<Index>& internal_faces() const
    { return internal_faces_; };


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
    * @brief Returns the number of vertices in the mesh.
    * @return Number of vertices.
    */
    Index num_vertices() const
    { return vertices_.cols(); };


    /**
    * @brief Returns the number of faces in the mesh.
    * @return Number of faces.
    */
    Index num_faces() const
    { return faces_.cols(); };


    /**
    * @brief Returns the number of edges in the mesh.
    * @return Number of edges.
    */
    Index num_edges() const
    { return edges_.cols(); };


    /**
    * @brief Returns a sub-mesh that contains only specified faces of this mesh.
    * @param[in] face_inds - Indices of faces to keep in the sub-mesh.
    * @return Partitioned mesh containing the specified faces.
    */
    TriangleMesh<dim> partition_by_faces(
        ConstEigRef<EigRowVec<Index>> face_inds
        ) const;


    /**
    * @brief Returns a sub-mesh that contains only faces of this mesh that lie within a given bounding box.
    * @param[in] bbox - Bounding box defined by two corners (min and max).
    * @param[in] strict - If true, only faces fully contained within the bounding box are included (optional).
    * @return Partitioned mesh containing faces within the given bounding box.
    */
    TriangleMesh<dim> partition_by_bbox(
        ConstEigRef<EigMatMN<Float, dim, 2>> bbox,
        const bool strict = true
        ) const;


    /**
    * @brief Computes and returns the centroid of each face.
    * @return Face-wise centroids.
    */
    EigMatNX<Float, dim> face_centroids() const
    {
        EigMatNX<Float, dim> centroids = EigMatNX<Float, dim>::Zero(dim, faces_.cols());
        for (Index ii = 0; ii < faces_.cols(); ++ii)
        {
            EigMatMN<Float, dim, 3> coords = vertices_(
                Eigen::placeholders::all, faces_.col(ii)
                );
            centroids(Eigen::placeholders::all, ii) = coords.rowwise().mean();
        }
        return centroids;
    };


    /**
    * @brief Returns a `Triangle` primitive object representing a specific face of the mesh.
    * @param[in] face - Index of the face.
    * @return `Triangle` object representing the specified face.
    */
    Triangle<dim> face_primitive(Index face) const
    {
        return Triangle<dim> (
            vertices_(Eigen::placeholders::all, faces_.col(face)),
            face_edge_polarities_.col(face),
            face_tags_[face],
            face
            );
    };


    /**
    * @brief Reverses the orientation of each face.
    */
    void reverse_orientation()
    {
        faces_ = faces_.colwise().reverse().eval();
        generate_edges();
        return;
    };


protected:

    /**
    * @brief Generates edges for the mesh based on face vertices.
    * @details
    * Edges are generated tag-wise. Tags represent distinct mesh regions, and edges cannot be
    * associated with two faces that belong to different mesh regions. For example, if two mesh
    * regions are in contact with one another, each region will have its own edges, rather than
    * creating artificial junctions at the point of contact.
    */
    void generate_edges();


    EigMatNX<Float, dim> vertices_;
    EigMatNX<Index, 3> faces_;
    EigRowVec<Index> face_tags_;

    bool decoupled_edges_ = false;

    EigMatNX<Index, 2> edges_;
    EigMatNX<Index, 3> face_edges_;
    EigMatNX<Float, 3> face_edge_polarities_;

    EigMatNX<Index, 2> edge_faces_;

    EigRowVec<Index> boundary_faces_;
    EigRowVec<Index> junction_faces_;
    EigRowVec<Index> internal_faces_;

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
