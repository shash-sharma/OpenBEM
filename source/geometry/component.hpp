// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Component management class.
*/

#ifndef BEM_COMPONENT_H
#define BEM_COMPONENT_H

#include <string>
#include <memory>

#include "materials.hpp"
#include "geometry/mesh/base.hpp"


namespace bem
{

/**
* \ingroup geom
* @{
*/

/**
* @brief Class that defines a component in a structure.
* @tparam MeshType - Type of the mesh for representing the component.
*/
template <typename MeshType>
class Component
{
public:

    /**
    * @brief Constructs a `Component` with a mesh view and associated material.
    * @param[in] mesh_view - The mesh view associated with the component.
    * @param[in] material - Material associated with the component.
    * @param[in] name - Name of the component (optional).
    * @param[in] cache_mesh - Whether to cache the mesh data for faster access (optional).
    */
    template <typename MaterialType>
    Component(
        const MeshView<MeshType>& mesh_view,
        const MaterialType& material,
        const std::string name = "component",
        const bool cache_mesh = false
        ):
            mesh_view_(mesh_view),
            material_(std::make_shared<MaterialType> (material)),
            name_(name),
            cache_mesh_(cache_mesh)
        {
            if (cache_mesh_)
                mesh_ = mesh_view_.mesh();
            return;
        };


    /**
    * @brief Constructs a `Component` with a mesh view.
    * @param[in] mesh_view - The mesh view associated with the component.
    * @param[in] name - Name of the component (optional).
    * @param[in] cache_mesh - Whether to cache the mesh data for faster access (optional).
    */
    Component(
        const MeshView<MeshType>& mesh_view,
        const std::string name = "component",
        const bool cache_mesh = false
        ): Component(mesh_view, PerfectDielectricMaterial(1, 1), name, cache_mesh) {};


    /**
    * @brief Returns the mesh view associated with the component.
    * @return Mesh view associated with the component.
    */
    const MeshView<MeshType>& mesh_view() const
    { return mesh_view_; };


    /**
    * @brief Returns the material associated with the component.
    * @return Material associated with the component.
    */
    const Material& material() const
    { return *material_; };


    /**
    * @brief Returns the mesh associated with the component.
    * @return Mesh associated with the component.
    */
    MeshType mesh() const
    {
        if (cache_mesh_)
            return mesh_;
        else
            return mesh_view_.mesh();
    };


    /**
    * @brief Returns a reference to a cached mesh for the component; generates and caches
    * the mesh if not already cached.
    * @return Reference to the cached mesh.
    */
    const MeshType& cached_mesh() const
    {
        if (cache_mesh_)
            return mesh_;
        else
            mesh_ = mesh_view_.mesh();
        cache_mesh_ = true;
        return mesh_;
    };


    /**
    * @brief Returns the name of the component.
    * @return Name of the component.
    */
    const std::string& name() const { return name_; };


    /**
    * @brief Sets the material of the component.
    * @param[in] material - Material to set for the component.
    */
    template <typename MaterialType>
    void set_material(const MaterialType& material)
    { material_ = std::make_shared<MaterialType> (material); return; };


    /**
    * @brief Sets the name of the component.
    * @param[in] name - Name to set for the component.
    */
    void set_name(const std::string& name)
    { name_ = name; return; };


private:

    const MeshView<MeshType> mesh_view_;
    std::shared_ptr<Material> material_ = std::make_shared<PerfectDielectricMaterial> (
        PerfectDielectricMaterial(1, 1)
        );
    std::string name_ = "component";
    mutable MeshType mesh_;
    mutable bool cache_mesh_ = false;
};

/**
* @}
*/

}

#endif
