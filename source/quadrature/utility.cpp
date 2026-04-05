// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Utility functions for quadrature.
*/

#include "quadrature/utility.hpp"

#include <vector>
#include <cmath>
#include <fstream>
#include <string>

#include "external/json.hpp"

#include "types.hpp"
#include "constants.hpp"

using json = nlohmann::json;


namespace bem
{

bool compare_with_tol(
    const Complex val,
    const Complex val_ref,
    const Float tol,
    const uint8_t mode
    )
{

    if (val == val_ref)
        return true;

    if (mode == 1)
    {
        Float test_val_real = std::abs(val.real() - val_ref.real());
        Float ref_val_real = std::abs(val.real()) + float_eps;

        Float test_val_imag = std::abs(val.imag() - val_ref.imag());
        Float ref_val_imag = std::abs(val.imag()) + float_eps;

        if (test_val_real <= tol * ref_val_real &&
            test_val_imag <= tol * ref_val_imag)
            return true;
    }
    else if (mode == 2)
    {
        Float test_val_abs = std::abs(std::abs(val.imag() - val_ref.imag()));
        Float ref_val_abs = std::abs(val) + float_eps;

        Float test_val_arg = std::abs(std::arg(val) - std::arg(val_ref));
        Float ref_val_arg = std::abs(std::arg(val)) + float_eps;

        if (test_val_abs <= tol * ref_val_abs &&
            test_val_arg <= tol * ref_val_arg)
            return true;
    }
    else if (mode == 3)
    {
        if (std::abs(val - val_ref) <= tol * (std::abs(val_ref) + float_eps))
            return true;
    }

    return false;

};


template <uint8_t dim>
std::vector<QuadratureRule<dim>> load_rules(
    const std::string file,
    ConstEigRef<EigColVec<Index>> orders
    )
{

    std::size_t pos = std::string(__FILE__).find_last_of("/");
    std::string file_with_path = std::string(__FILE__).substr(0, pos) + "/" + file;

    std::ifstream in_stream (file_with_path);
    json data = json::parse(in_stream);

    std::vector<QuadratureRule<dim>> rules;
    rules.reserve(orders.size());

    for (Index order: orders)
    {
        json order_data = data[std::to_string(order)];

        QuadratureRule<dim> rule;
        rule.num_nodes = order_data["num_nodes"].template get<Index> ();

        std::vector<std::vector<Float>> nodes = order_data["nodes"];

        rule.nodes.resize(dim, rule.num_nodes);
        for (Index ii = 0; ii < nodes.size(); ++ii)
            for (uint8_t jj = 0; jj < dim; ++jj)
                rule.nodes(jj, ii) = nodes[ii][jj];

        rule.weights = Eigen::Map<EigMat<Float>> (
            ((std::vector<Float>) order_data["weights"]).data(), 1, rule.num_nodes
            ).eval();

        rules.push_back(rule);
    }

    return rules;

};


template std::vector<QuadratureRule<1>> load_rules(
    const std::string file,
    ConstEigRef<EigColVec<Index>> orders
    );

template std::vector<QuadratureRule<2>> load_rules(
    const std::string file,
    ConstEigRef<EigColVec<Index>> orders
    );

template std::vector<QuadratureRule<3>> load_rules(
    const std::string file,
    ConstEigRef<EigColVec<Index>> orders
    );

}

