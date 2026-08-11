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

#include "geometry/mesh/triangle_mesh.hpp"

#include <map>
#include <utility>
#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <set>
#include <stdexcept>

#include "types.hpp"


namespace bem
{

template <uint8_t dim>
void TriangleMesh<dim>::set_data(
    ConstEigRef<EigMatNX<Float, dim>> vertices,
    ConstEigRef<EigMatNX<Index, 3>> faces,
    ConstEigRef<EigRowVec<Index>> face_tags,
    const bool decoupled_edges
    )
{
    root_faces_.resize(1, 0);
    root_edges_.resize(1, 0);
    root_vertices_.resize(1, 0);

    std::vector<Index> order (face_tags.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
        [&] (Index a, Index b) { return face_tags[a] < face_tags[b]; });

    vertices_ = vertices;
    faces_.resize(3, faces.cols());
    face_tags_.resize(1, face_tags.size());
    for (Index new_pos = 0; new_pos < order.size(); ++new_pos)
    {
        faces_.col(new_pos) = faces.col(order[new_pos]);
        face_tags_[new_pos] = face_tags[order[new_pos]];
    }

    decoupled_edges_ = decoupled_edges;
    generate_edges();
    return;
};


template <uint8_t dim>
TriangleMesh<dim> TriangleMesh<dim>::partition_by_faces(
    ConstEigRef<EigRowVec<Index>> face_inds
    ) const
{

    Index num_new_vertices = 0;
    std::map<Index, Index> new_vert_map;

    for (Index jj = 0; jj < face_inds.size(); ++jj)
    {
        for (uint8_t kk = 0; kk < 3; ++kk)
        {
            bool inserted = new_vert_map.insert(
                std::make_pair(faces_(kk, face_inds[jj]), num_new_vertices)
                ).second;
            if (inserted)
                num_new_vertices++;
        }
    }

    EigMatNX<Float, dim> new_vertices = EigMatNX<Float, dim>::Zero(dim, num_new_vertices);
    for (auto const& x: new_vert_map)
        new_vertices.col(x.second) = vertices_.col(x.first);

    EigMatNX<Index, 3> new_faces = EigMatNX<Index, 3>::Zero(3, face_inds.size());

    EigRowVec<Index> new_face_tags = EigRowVec<Index>::Zero(1, face_inds.size());

    for (Index jj = 0; jj < face_inds.size(); ++jj)
    {
        for (uint8_t kk = 0; kk < 3; ++kk)
        {
            new_faces(kk, jj) = new_vert_map[
                faces_(kk, face_inds[jj])
                ];
        }
        new_face_tags[jj] = face_tags_[face_inds[jj]];
    }

    EigRowVec<Index> kept_edges = compute_face_edges(face_inds);

    std::map<Index, Index> edge_reverse_map;
    for (Index ee = 0; ee < kept_edges.size(); ++ee)
        edge_reverse_map[kept_edges[ee]] = ee;

    EigMatNX<Index, 2> new_edges (2, kept_edges.size());
    for (Index ee = 0; ee < kept_edges.size(); ++ee)
    {
        new_edges(0, ee) = new_vert_map[edges_(0, kept_edges[ee])];
        new_edges(1, ee) = new_vert_map[edges_(1, kept_edges[ee])];
    }

    EigMatNX<Index, 3> new_face_edges(3, face_inds.size());
    EigMatNX<Float, 3> new_face_edge_polarities(3, face_inds.size());

    for (Index jj = 0; jj < face_inds.size(); ++jj)
    {
        for (uint8_t kk = 0; kk < 3; ++kk)
        {
            new_face_edges(kk, jj) = edge_reverse_map[face_edges_(kk, face_inds[jj])];
            new_face_edge_polarities(kk, jj) = face_edge_polarities_(kk, face_inds[jj]);
        }
    }

    TriangleMesh<dim> partition;
    partition.vertices_ = new_vertices;
    partition.faces_ = new_faces;
    partition.face_tags_ = new_face_tags;
    partition.decoupled_edges_ = decoupled_edges_;
    partition.edges_ = new_edges;
    partition.face_edges_ = new_face_edges;
    partition.face_edge_polarities_ = new_face_edge_polarities;
    partition.edge_faces_ = partition.compute_edge_faces();
    partition.classify_edges_and_faces();

    partition.root_faces_.resize(1, face_inds.size());
    for (Index jj = 0; jj < face_inds.size(); ++jj)
        partition.root_faces_[jj] = root_face(face_inds[jj]);

    partition.root_edges_.resize(1, kept_edges.size());
    for (Index ee = 0; ee < kept_edges.size(); ++ee)
        partition.root_edges_[ee] = root_edge(kept_edges[ee]);

    partition.root_vertices_.resize(1, num_new_vertices);
    for (auto const& x: new_vert_map)
        partition.root_vertices_[x.second] = root_vertex(x.first);

    return partition;

};


template <uint8_t dim>
TriangleMesh<dim> TriangleMesh<dim>::partition_by_bbox(
    ConstEigRef<EigMatMN<Float, dim, 2>> bbox,
    const bool strict
    ) const
{

    std::vector<Index> faces_in_bbox;

    for (Index ii = 0; ii < faces_.cols(); ++ii)
    {
        EigMatMN<Float, dim, 3> coords = vertices_(
            Eigen::placeholders::all, faces_.col(ii)
            );
        EigColVecN<Float, dim> centroid = coords.rowwise().mean();
        bool centroid_in_bbox =
            (centroid.array() <= (bbox.rowwise().maxCoeff()).array()).all() &&
            (centroid.array() >= (bbox.rowwise().minCoeff()).array()).all();

        if (centroid_in_bbox)
        {
            if (!strict)
                faces_in_bbox.push_back(ii);
            else
            {
                bool fully_in_bbox =
                    ((coords.rowwise().maxCoeff()).array() <= (bbox.rowwise().maxCoeff()).array()
                    ).all() &&
                    ((coords.rowwise().minCoeff()).array() >= (bbox.rowwise().minCoeff()).array()
                    ).all();

                if (fully_in_bbox)
                    faces_in_bbox.push_back(ii);
            }
        }
    }

    return partition_by_faces(
        Eigen::Map<EigRowVec<Index>> (faces_in_bbox.data(), faces_in_bbox.size())
        );

};


template <uint8_t dim>
void TriangleMesh<dim>::generate_edges()
{

    Index num_edges_found = 0;
    std::vector<std::pair<std::pair<Index, Index>, Index>> edges;

    face_edges_.resize(3, faces_.cols());
    face_edge_polarities_.resize(3, faces_.cols());

    std::set<Index> unique_tags { face_tags_.begin(), face_tags_.end() };

    for (Index tag: unique_tags)
    {
        std::map<std::pair<Index, Index>, Index> unique_edges;

        for (Index face = 0; face < faces_.cols(); ++face)
        {
            if (face_tags_[face] != tag)
                continue;

            for (uint8_t edge = 0; edge < 3; ++edge)
            {
                std::pair<Index, Index> vertices = std::make_pair(
                    faces_(edge, face),
                    faces_((edge + 1) % 3, face)
                    );

                if (!decoupled_edges_ && vertices.first > vertices.second)
                    std::swap(vertices.first, vertices.second);

                bool inserted = unique_edges.insert(std::make_pair(vertices, num_edges_found)).second;

                if (inserted)
                {
                    face_edges_(edge, face) = num_edges_found;
                    face_edge_polarities_(edge, face) = 1;
                    edges.push_back(std::make_pair(vertices, num_edges_found));
                    num_edges_found++;
                }
                else
                {
                    face_edges_(edge, face) = unique_edges[vertices];
                    face_edge_polarities_(edge, face) = -1;
                }
            }
        }
    }

    edges_.resize(2, num_edges_found);
    for (const auto& edge: edges)
        edges_.col(edge.second) = EigColVecN<Index, 2> ({ edge.first.first, edge.first.second });

    edge_faces_ = compute_edge_faces();
    classify_edges_and_faces();

    return;

};


template <uint8_t dim>
EigRowVec<Index> TriangleMesh<dim>::compute_face_edges(
    ConstEigRef<EigRowVec<Index>> face_inds
    ) const
{

    std::set<Index> unique_edges;
    for (Index ii = 0; ii < face_inds.size(); ++ii)
        for (uint8_t jj = 0; jj < 3; ++jj)
            unique_edges.insert(face_edges_(jj, face_inds[ii]));

    EigRowVec<Index> result (1, unique_edges.size());
    Index kk = 0;
    for (const Index edge: unique_edges)
        result[kk++] = edge;

    return result;

};


template <uint8_t dim>
EigMatNX<Index, 2> TriangleMesh<dim>::compute_edge_faces() const
{

    EigMatNX<Index, 2> result (2, num_edges());
    for (Index face = 0; face < faces_.cols(); ++face)
    {
        for (uint8_t edge = 0; edge < 3; ++edge)
        {
            const Index e = face_edges_(edge, face);
            if (face_edge_polarities_(edge, face) > 0)
                result(0, e) = face;
            else
                result(1, e) = face;
        }
    }

    return result;

};


template <uint8_t dim>
void TriangleMesh<dim>::classify_edges_and_faces()
{

    EigRowVec<Index> edge_counts = EigRowVec<Index>::Zero(1, num_edges());
    for (Index face = 0; face < faces_.cols(); ++face)
        for (uint8_t edge = 0; edge < 3; ++edge)
            edge_counts[face_edges_(edge, face)]++;

    std::vector<Index> boundary_edges, junction_edges, internal_edges;
    for (Index edge = 0; edge < edge_counts.size(); ++edge)
    {
        if (edge_counts[edge] == 1 && !decoupled_edges_)
            boundary_edges.push_back(edge);
        else if (edge_counts[edge] > 2 && !decoupled_edges_)
            junction_edges.push_back(edge);
        if (edge_counts[edge] > 1 && !decoupled_edges_)
            internal_edges.push_back(edge);
    }

    boundary_edges_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        boundary_edges.data(), boundary_edges.size()
        );
    junction_edges_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        junction_edges.data(), junction_edges.size()
        );
    internal_edges_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        internal_edges.data(), internal_edges.size()
        );

    std::vector<Index> boundary_faces, junction_faces, internal_faces;
    for (Index face = 0; face < faces_.cols(); ++face)
    {
        for (uint8_t edge = 0; edge < 3; ++edge)
        {
            const Index e = face_edges_(edge, face);
            if (edge_counts[e] == 1 && !decoupled_edges_)
                boundary_faces.push_back(face);
            else if (edge_counts[e] > 2 && !decoupled_edges_)
                junction_faces.push_back(face);
            if (edge_counts[e] > 1 && !decoupled_edges_)
                internal_faces.push_back(face);
        }
    }

    auto remove_duplicates = [] (std::vector<Index>& v)
    {
        std::sort(v.begin(), v.end());
        auto last = std::unique(v.begin(), v.end());
        v.erase(last, v.end());
        return;
    };

    remove_duplicates(boundary_faces);
    remove_duplicates(junction_faces);
    remove_duplicates(internal_faces);

    boundary_faces_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        boundary_faces.data(), boundary_faces.size()
        );
    junction_faces_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        junction_faces.data(), junction_faces.size()
        );
    internal_faces_ = Eigen::Map<EigRowVec<Index>, Eigen::Unaligned> (
        internal_faces.data(), internal_faces.size()
        );

    return;

};


template class TriangleMesh<2>;
template class TriangleMesh<3>;

}
