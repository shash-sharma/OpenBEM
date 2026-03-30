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

#ifndef BEM_STRUCTURE_I
#define BEM_STRUCTURE_I

#include <string>
#include <vector>

#include "geometry/component.hpp"
#include "geometry/mesh/base.hpp"


namespace bem
{

template <typename MeshType>
std::vector<Component<MeshType>> Structure<MeshType>::components_by_name(
    const std::string name,
    const bool search_metacomponents,
    const bool case_sensitive
    )
{
    std::vector<Component<MeshType>> comps;

    auto run = [&] (const Component<MeshType>& comp)
    {
        std::string comp_name;
        if (case_sensitive)
            comp_name = comp.name();
        if (!case_sensitive)
            comp_name = comp.name().tolower();
        if (comp_name.find(name) != std::string::npos)
            comps.push_back(comp);
    };

    if (!search_metacomponents)
    {
        for (const auto& comp: components())
            run(comp);
    }
    else
    {
        for (const auto& comp: metacomponents())
            run(comp);
    }

    return comps;
};


template <typename MeshType>
std::vector<MeshView<MeshType>> Structure<MeshType>::mesh_views_by_name(
    const std::string name,
    const bool search_metacomponents,
    const bool case_sensitive
    )
{
    std::vector<Component<MeshType>> comps = components_by_name(
        name, search_metacomponents, case_sensitive
        );
    std::vector<MeshView<MeshType>> mvs;
    for (const auto& comp: comps)
        mvs.push_back(comp.mesh_view());
    return mvs;
};

}

#endif
