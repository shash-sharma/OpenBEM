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

#include "materials.hpp"


namespace bem
{

// Forward declarations
template <typename MeshType> class Component;
template <typename MeshType> class MeshView;

/**
* \ingroup geom
* @{
*/

/**
* @brief Class that defines a structure.
* @tparam MeshType - Type of the mesh for representing the structure.
*/
template <typename MeshType>
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
        const MeshType& mesh,
        const Material background_material = PerfectDielectricMaterial(1, 1)
        ): mesh_(mesh), background_material_(background_material) {};


    /**
    * @brief Adds a component to the structure.
    * @param[in] component - Component to add.
    */
    void add_component(const Component<MeshType>& component)
    {
        components_.push_back(component);
        return;
    };


    /**
    * @brief Adds a meta`Component` to the structure which corresponds to a given subset of the mesh.
    * @param[in] metacomponent - Metacomponent to add.
    */
    void add_metacomponent(const Component<MeshType>& metacomponent)
    {
        metacomponents_.push_back(metacomponent);
        return;
    };


    /**
    * @brief Returns the mesh associated with the structure in editable form.
    * @return Mesh associated with the structure.
    */
    MeshType& mesh()
    { return mesh_; };


    /**
    * @brief Returns the mesh associated with the structure in read-only form.
    * @return Mesh associated with the structure.
    */
    const MeshType& mesh() const
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
    std::vector<Component<MeshType>>& components()
    { return components_; };


    /**
    * @brief Returns a read-only list of the components in the structure.
    * @return Components.
    */
    const std::vector<Component<MeshType>>& components() const
    { return components_; };


    /**
    * @brief Returns an editable list of the metacomponents in the structure.
    * @return Metacomponents.
    */
    std::vector<Component<MeshType>>& metacomponents()
    { return metacomponents_; };


    /**
    * @brief Returns a read-only list of the metacomponents in the structure.
    * @return Metacomponents.
    */
    const std::vector<Component<MeshType>>& metacomponents() const
    { return metacomponents_; };


    /**
    * @brief Returns a list of the components whose name contains a given string.
    * @param[in] name - Search string.
    * @param[in] search_metacomponents - Whether to search the metacomponents instead of components (optional).
    * @return Named components.
    */
    std::vector<Component<MeshType>> components_by_name(
        const std::string name,
        const bool search_metacomponents = false
        );


    /**
    * @brief Returns a list of mesh views associated with components whose name contains a given string.
    * @param[in] name - Search string.
    * @param[in] search_metacomponents - Whether to search the metacomponents instead of components (optional).
    * @param[in] case_sensitive - Whether the search should be case sensitive (optional).
    * @return Named mesh views.
    */
    std::vector<MeshView<MeshType>> mesh_views_by_name(
        const std::string name,
        const bool search_metacomponents = false,
        const bool case_sensitive = false
        );


private:

    MeshType mesh_;
    Material background_material_ = PerfectDielectricMaterial(1, 1);

    std::vector<Component<MeshType>> components_;
    std::vector<Component<MeshType>> metacomponents_;

};

/**
* @}
*/

}

#include "geometry/structure.tpp"

#endif
