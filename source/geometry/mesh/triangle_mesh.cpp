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
    vertices_ = vertices;
    faces_ = faces;
    face_tags_ = face_tags;
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

    TriangleMesh<dim> partition;
    partition.set_data(new_vertices, new_faces, new_face_tags, decoupled_edges_);

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

    Index num_edges = 0;
    std::vector<std::pair<Index, uint8_t>> edge_counts;
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

                bool inserted = unique_edges.insert(std::make_pair(vertices, num_edges)).second;

                if (inserted)
                {
                    face_edges_(edge, face) = num_edges;
                    face_edge_polarities_(edge, face) = 1;
                    edge_counts.push_back(std::make_pair(num_edges, 1));
                    edges.push_back(std::make_pair(vertices, num_edges));
                    num_edges++;
                }
                else
                {
                    face_edges_(edge, face) = unique_edges[vertices];
                    face_edge_polarities_(edge, face) = -1;
                    edge_counts[unique_edges[vertices]].second++;
                }
            }
        }
    }

    edges_.resize(2, num_edges);
    for (const auto& edge: edges)
        edges_.col(edge.second) = EigColVecN<Index, 2> ({ edge.first.first, edge.first.second });

    edge_faces_.resize(2, num_edges);
    for (Index face = 0; face < face_edges_.cols(); ++face)
    {
        for (uint8_t edge = 0; edge < 3; ++edge)
        {
            if (face_edge_polarities_(edge, face) > 0)
                edge_faces_(0, face_edges_(edge, face)) = face;
            else
                edge_faces_(1, face_edges_(edge, face)) = face;
        }
    }

    std::vector<Index> boundary_edges, junction_edges, internal_edges;
    for (Index edge = 0; edge < num_edges; ++edge)
    {
        if (edge_counts[edge].second == 1 && !decoupled_edges_)
            boundary_edges.push_back(edge);
        else if (edge_counts[edge].second > 2 && !decoupled_edges_)
            junction_edges.push_back(edge);
        if (edge_counts[edge].second > 1 && !decoupled_edges_)
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
            if (edge_counts[face_edges_(edge, face)].second == 1 && !decoupled_edges_)
                boundary_faces.push_back(face);
            else if (edge_counts[face_edges_(edge, face)].second > 2 && !decoupled_edges_)
                junction_faces.push_back(face);
            if (edge_counts[face_edges_(edge, face)].second > 1 && !decoupled_edges_)
                internal_faces.push_back(face);
        }
    }

    auto remove_duplicates = [] (std::vector<Index> &v)
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
