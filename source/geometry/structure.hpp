// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Structure management class.
*/

#ifndef BEM_STRUCTURE_H
#define BEM_STRUCTURE_H

#include <string>
#include <vector>

#include "types.hpp"
#include "materials.hpp"
#include "geometry/component.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/mesh/triangle_mesh_view.hpp"


namespace bem
{

/**
* \ingroup geom
* @{
*/

/**
* @brief Class that defines a structure.
* @tparam dim - The dimension of the mesh (2 or 3).
*/
template <uint8_t dim>
class Structure
{
public:

    /**
    * @brief Constructs an empty `Structure`.
    * @param[in] background_material - Background material in which the structure is embedded (optional).
    */
    Structure(
        const Material background_material = PerfectDielectricMaterial(1, 1)
        ): background_material_(background_material) {};


    /**
    * @brief Constructs a `Structure` with a given mesh.
    * @param[in] mesh - The mesh associated with the structure.
    * @param[in] background_material - Background material in which the structure is embedded (optional).
    */
    Structure(
        const TriangleMesh<dim>& mesh,
        const Material background_material = PerfectDielectricMaterial(1, 1)
        ): mesh_(mesh), background_material_(background_material) {};


    /**
    * @brief Constructs a `Structure` from a single component of another structure.
    * @param[in] parent - Structure that holds the component.
    * @param[in] idx - Index of the component in `parent`.
    */
    Structure(
        const Structure<dim>& parent,
        const Index idx
        ):
            mesh_(parent.components_.at(idx).mesh_view().parent_mesh().partition_by_faces(
                parent.components_.at(idx).mesh_view().face_inds(), true
                )),
            background_material_(parent.background_material_)
        {
            components_.push_back(
                Component<dim> (TriangleMeshView<dim> (mesh_), parent.components_[idx])
                );
            return;
        };


    /**
    * @brief Adds a component to the structure.
    * @param[in] component - Component to add.
    */
    void add_component(const Component<dim>& component)
    {
        components_.push_back(component);
        return;
    };


    /**
    * @brief Adds a meta`Component` to the structure which corresponds to a given subset of the mesh.
    * @param[in] metacomponent - Metacomponent to add.
    */
    void add_metacomponent(const Component<dim>& metacomponent)
    {
        metacomponents_.push_back(metacomponent);
        return;
    };


    /**
    * @brief Returns the mesh associated with the structure in editable form.
    * @return Mesh associated with the structure.
    */
    TriangleMesh<dim>& mesh()
    { return mesh_; };


    /**
    * @brief Returns the mesh associated with the structure in read-only form.
    * @return Mesh associated with the structure.
    */
    const TriangleMesh<dim>& mesh() const
    { return mesh_; };


    /**
    * @brief Returns the background material of the structure.
    * @return Background material of the structure.
    */
    const Material& background_material() const
    { return background_material_; };


    /**
    * @brief Returns an editable list of the components in the structure.
    * @return Components.
    */
    std::vector<Component<dim>>& components()
    { return components_; };


    /**
    * @brief Returns a read-only list of the components in the structure.
    * @return Components.
    */
    const std::vector<Component<dim>>& components() const
    { return components_; };


    /**
    * @brief Returns an editable list of the metacomponents in the structure.
    * @return Metacomponents.
    */
    std::vector<Component<dim>>& metacomponents()
    { return metacomponents_; };


    /**
    * @brief Returns a read-only list of the metacomponents in the structure.
    * @return Metacomponents.
    */
    const std::vector<Component<dim>>& metacomponents() const
    { return metacomponents_; };


private:

    TriangleMesh<dim> mesh_;
    Material background_material_ = PerfectDielectricMaterial(1, 1);

    std::vector<Component<dim>> components_;
    std::vector<Component<dim>> metacomponents_;

};

/**
* @}
*/

}

#endif
