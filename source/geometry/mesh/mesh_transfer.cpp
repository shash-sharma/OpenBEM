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

#include "geometry/mesh/mesh_transfer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <string>
#include <map>

#include "types.hpp"
#include "geometry/structure.hpp"
#include "geometry/mesh/triangle_mesh.hpp"


namespace bem
{

void MeshTransfer::read_gmsh_v2(
    Structure<TriangleMesh<3>>& structure,
    const std::string msh_filename,
    const bool decoupled_edges
    )
{

    std::ifstream file(msh_filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open mesh file: " + msh_filename);

    // Check version and format
    std::string line;
    std::getline(file, line);
    if (line != "$MeshFormat")
        throw std::runtime_error("Invalid Gmsh file format: missing $MeshFormat header");

    std::getline(file, line); // version file-type data-size
    std::istringstream iss (line);
    float version;
    int file_type, data_size;
    iss >> version >> file_type >> data_size;

    if (version < 2.0 || version >= 3.0)
        throw std::runtime_error(".msh file version must be 2.x");

    std::getline(file, line); // $EndMeshFormat

    // First pass: count vertices, elements, surfaces, and physical names
    std::size_t num_vertices = 0;
    std::size_t num_faces = 0;

    bool in_elements = false;
    bool in_physical_names = false;

    std::map<Index, Index> surface_to_num_elems;
    std::map<Index, Index> physical_to_num_elems;
    std::vector<std::string> physical_names;

    uint8_t elm_map = 2;

    while (std::getline(file, line))
    {
        if (line == "$Nodes")
        {
            std::getline(file, line); // Read number of vertices
            num_vertices = std::stoull(line);
            continue;
        }
        if (line == "$EndNodes")
        {
            continue;
        }
        if (line == "$Elements")
        {
            in_elements = true;
            std::getline(file, line); // Skip number of elements line
            continue;
        }
        if (line == "$EndElements")
        {
            in_elements = false;
            continue;
        }
        if (line == "$PhysicalNames")
        {
            in_physical_names = true;
            std::getline(file, line); // Read number of physical names
            continue;
        }
        if (line == "$EndPhysicalNames")
        {
            in_physical_names = false;
            continue;
        }

        if (in_physical_names)
        {
            std::istringstream iss(line);
            Int n1, n2;
            std::string name;
            iss >> n1 >> n2 >> name;
            name.erase(std::remove(name.begin(), name.end(), '\"' ), name.end());
            physical_names.push_back(name);
            continue;
        }

        if (in_elements)
        {
            std::istringstream iss(line);
            std::size_t elm_id, elm_type, num_tags, phys_tag, surf_tag;
            iss >> elm_id >> elm_type >> num_tags >> phys_tag >> surf_tag;
            phys_tag -= 1;
            surf_tag -= 1;

            // Only count faces of the given type
            if (elm_type == elm_map)
            {
                num_faces++;

                bool inserted_surface = surface_to_num_elems.insert(std::make_pair(surf_tag, 1)).second;
                if (!inserted_surface)
                    surface_to_num_elems[surf_tag]++;

                bool inserted_physical = physical_to_num_elems.insert(std::make_pair(phys_tag, 1)).second;
                if (!inserted_physical)
                    physical_to_num_elems[phys_tag]++;
            }
        }
    }

    EigMatNX<Float, 3> verts = EigMatNX<Float, 3>::Zero(3, num_vertices);
    EigMatNX<Index, 3> elems = EigMatNX<Index, 3>::Zero(3, num_faces);
    EigRowVec<Index> elem_tags = EigRowVec<Index>::Zero(1, num_faces);

    std::map<Index, EigRowVec<Index>> surface_elems;
    std::map<Index, Index> surface_elem_counters;
    for (auto& [key, value]: surface_to_num_elems)
    {
        surface_elems[key] = EigRowVec<Index>::Zero(1, value);
        surface_elem_counters[key] = 0;
    }

    std::map<Index, EigRowVec<Index>> physical_elems;
    std::map<Index, Index> physical_elem_counters;
    for (auto& [key, value]: physical_to_num_elems)
    {
        physical_elems[key] = EigRowVec<Index>::Zero(1, value);
        physical_elem_counters[key] = 0;
    }

    // Second pass: read actual data
    file.clear();
    file.seekg(0);

    // Skip to $Nodes section
    while (std::getline(file, line) && line != "$Nodes") {}
    std::getline(file, line); // Skip number of vertices line

    // Read vertex coordinates
    std::size_t vertex_idx = 0;
    while (std::getline(file, line) && line != "$EndNodes")
    {
        std::istringstream iss(line);
        std::size_t vertex_id;
        Float x, y, z;
        iss >> vertex_id >> x >> y >> z;
        verts.col(vertex_idx++) << x, y, z;
    }

    // Skip to $Elements section
    while (std::getline(file, line) && line != "$Elements") {}
    std::getline(file, line); // Skip number of elements line

    // Read faces
    std::size_t face_idx = 0;
    while (std::getline(file, line) && line != "$EndElements")
    {
        std::istringstream iss(line);
        std::size_t elm_id, elm_type, num_tags;
        iss >> elm_id >> elm_type >> num_tags;

        if (num_tags != 2)
            throw std::runtime_error(
                "Unexpected number of tags in file " + msh_filename + " for element " + std::to_string(elm_id)
            );

        std::vector<std::size_t> tags (num_tags);
        for (std::size_t ii = 0; ii < num_tags; ++ii)
            iss >> tags[ii];

        std::size_t ptag = tags[0] - 1;
        std::size_t stag = tags[1] - 1;

        // Only process faces of the given type
        if (elm_type == elm_map)
        {
            physical_elems[ptag][physical_elem_counters[ptag]++] = face_idx;
            surface_elems[stag][surface_elem_counters[stag]++] = face_idx;

            for (std::size_t ii = 0; ii < 3; ++ii)
            {
                std::size_t v;
                iss >> v;
                // Convert from 1-based to 0-based indexing
                elems(ii, face_idx) = v - 1;
            }
            // Convert from 1-based to 0-based indexing
            elem_tags[face_idx] = ptag;
            face_idx++;
        }
    }

    structure.mesh().set_data(verts, elems, elem_tags, decoupled_edges);

    for (const auto& [key, value]: surface_elems)
    {
        MeshView view (structure.mesh(), value, "surface_id_" + std::to_string(key));
        Component metacomp (view, PerfectDielectricMaterial(1, 1), "surface_id_" + std::to_string(key), true);
        structure.add_metacomponent(metacomp);
    }
    for (const auto& [key, value]: physical_elems)
    {
        std::string name = physical_names[key] + "_physical_id_" + std::to_string(key);
        MeshView view (structure.mesh(), value, name);
        Component comp (view, PerfectDielectricMaterial(1, 1), name, true);
        structure.add_component(comp);
    }

    return;

};


void MeshTransfer::write_gmsh_v2(
    const Structure<TriangleMesh<3>>& structure,
    const std::string msh_filename,
    const std::string extension
    )
{

    std::ofstream file(msh_filename + "." + extension);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + msh_filename + "." + extension);

    // Write header
    file << "$MeshFormat\n";
    file << "2.2 0 8\n"; // Gmsh version 2.2, ASCII format, data size 8 bytes
    file << "$EndMeshFormat\n";

    // Write physical names
    file << "$PhysicalNames\n";
    file << structure.components().size() << "\n";
    for (Index ii = 0; ii < structure.components().size(); ++ii)
        file << "2 " << ii + 1 << " \"" << structure.components()[ii].name() << "\"\n";
    file << "$EndPhysicalNames\n";

    // Nodes
    file << "$Nodes\n";
    file << structure.mesh().num_verts() << "\n";
    for (Index ii = 0; ii < structure.mesh().num_verts(); ++ii)
    {
        const EigColVecN<Float, 3>& v = structure.mesh().verts(ii);
        file << (ii + 1) << " " << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
    file << "$EndNodes\n";

    // Elements
    Index num_elems = 0;
    for (Index jj = 0; jj < structure.components().size(); ++jj)
        num_elems += structure.components()[jj].mesh_view().elem_inds().size();

    file << "$Elements\n";
    file << num_elems << "\n";
    Index elem_idx = 1;
    for (Index jj = 0; jj < structure.components().size(); ++jj)
    {
        for (Index ii = 0; ii < structure.components()[jj].mesh_view().elem_inds().size(); ++ii)
        {
            const EigColVecN<Index, 3>& fv = structure.mesh().elems(
                structure.components()[jj].mesh_view().elem_inds()[ii]
                );
            file << elem_idx++ << " 2 2 ";
            file << jj + 1 << " " << jj + 1 << " ";
            file << (fv[0] + 1) << " " << (fv[1] + 1) << " " << (fv[2] + 1) << "\n";
        }
    }
    file << "$EndElements\n";

    return;

};


void MeshTransfer::write_gmsh_v2_scalar_field(
    const Structure<TriangleMesh<3>>& structure,
    const std::string msh_filename,
    ConstEigRef<EigRowVec<Float>> field,
    std::string field_name,
    const std::string field_plot_type
    )
{

    MeshTransfer::write_gmsh_v2(structure, msh_filename, "pos");

    const bool at_nodes = (field_plot_type == "nodes");
    const Index num_entities = at_nodes ? structure.mesh().num_verts() : structure.mesh().num_elems();

    if (field.size() != num_entities)
        throw std::invalid_argument(
            "Field size must match the number of " +
            std::string(at_nodes ? "vertices" : "faces") + " in the mesh."
            );

    std::ofstream file(msh_filename + ".pos", std::ios_base::app);
    if (!file.is_open())
        throw std::runtime_error("Could not open mesh file: " + msh_filename + ".pos");

    file << (at_nodes ? "$NodeData\n" : "$ElementData\n");
    file << "1\n" << field_name << "\n";
    file << "1\n" << 0 << "\n";
    file << "4\n" << 0 << "\n" << 1 << "\n" << num_entities << "\n" << "0\n";
    for (Index ii = 0; ii < num_entities; ++ii)
        file << (ii + 1) << " " << field(ii) << "\n";
    file << (at_nodes ? "$EndNodeData\n" : "$EndElementData\n");

    return;

};


void MeshTransfer::write_gmsh_v2_vector_field(
    const Structure<TriangleMesh<3>>& structure,
    const std::string msh_filename,
    ConstEigRef<EigMatNX<Float, 3>> field,
    std::string field_name,
    const std::string field_plot_type
    )
{

    MeshTransfer::write_gmsh_v2_scalar_field(
        structure, msh_filename, field.colwise().norm(), "scalar_field", field_plot_type
        );

    const bool at_nodes = (field_plot_type == "nodes");
    const Index num_entities = at_nodes ? structure.mesh().num_verts() : structure.mesh().num_elems();

    if (field.cols() != num_entities)
        throw std::invalid_argument(
            "Field size must match the number of " +
            std::string(at_nodes ? "vertices" : "faces") + " in the mesh."
            );

    std::ofstream file(msh_filename + ".pos", std::ios_base::app);
    if (!file.is_open())
        throw std::runtime_error("Could not open mesh file: " + msh_filename + ".pos");

    file << (at_nodes ? "$NodeData\n" : "$ElementData\n");
    file << "1\n" << field_name << "\n";
    file << "1\n" << 0 << "\n";
    file << "4\n" << 0 << "\n" << 3 << "\n" << num_entities << "\n" << "0\n";
    for (Index ii = 0; ii < num_entities; ++ii)
        file << (ii + 1) << " " << field(0, ii) << " " << field(1, ii) << " " << field(2, ii) << "\n";
    file << (at_nodes ? "$EndNodeData\n" : "$EndElementData\n");

    return;

};

}
