// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Base classes for quadrature.
*/

#ifndef BEM_QUAD_BASE_H
#define BEM_QUAD_BASE_H

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

/**
* \ingroup quad
* @{
*/

/**
* @brief Data structure for storing quadrature points and weights.
* @tparam dim - Dimension of the integration domain (1, 2, or 3).
*/
template <uint8_t dim>
struct QuadratureData
{
    static_assert((dim == 1 || dim == 2 || dim == 3), "`dim` must be 1, 2, or 3.");

    EigMatNX<Float, dim> points;
    EigRowVec<Float> weights;

    bool converged = false;
    Index converged_iter = 0;
};


/**
* @brief Base class for quadrature.
* @tparam dim - Dimension of the integration domain (1, 2, or 3).
*/
template <uint8_t dim>
class QuadratureBase
{

    static_assert((dim == 1 || dim == 2 || dim == 3), "`dim` must be 1, 2, or 3.");

public:

    /**
    * @brief Sets the quadrature order.
    * @param[in] order - Quadrature order.
    */
    virtual void set_order(const uint8_t order) { order_ = order; return; };


    /**
    * @brief Returns the quadrature order.
    * @return Quadrature order.
    */
    uint8_t order() const { return order_; };


    /**
    * @brief Loads quadrature rules from the specified json file assumed to be located in the
    * same directory as this file.
    * @tparam dim - Dimension in which the quadrature rules are defined.
    * @param[in] file - json file name.
    * @param[in] orders - List of orders to load.
    * @return Vector of quadrature rule objects.
    */
    static std::vector<QuadratureData<dim>> load_rules(
        const std::string file,
        ConstEigRef<EigColVec<Index>> orders
        );


    /**
    * @brief Virtual destructor.
    */
    virtual ~QuadratureBase() = default;


protected:

    uint8_t order_;

};


template <uint8_t dim>
std::vector<QuadratureData<dim>> QuadratureBase<dim>::load_rules(
    const std::string file,
    ConstEigRef<EigColVec<Index>> orders
    )
{

    std::size_t pos = std::string(__FILE__).find_last_of("/");
    std::string file_with_path = std::string(__FILE__).substr(0, pos) + "/" + file;

    std::ifstream in_stream (file_with_path);
    json data = json::parse(in_stream);

    std::vector<QuadratureData<dim>> rules;
    rules.reserve(orders.size());

    for (Index order: orders)
    {
        json order_data = data[std::to_string(order)];

        QuadratureData<dim> rule;
        Index num_nodes = order_data["num_nodes"].template get<Index> ();

        std::vector<std::vector<Float>> nodes = order_data["nodes"];

        rule.points.resize(dim, num_nodes);
        for (Index ii = 0; ii < nodes.size(); ++ii)
            for (uint8_t jj = 0; jj < dim; ++jj)
                rule.points(jj, ii) = nodes[ii][jj];

        rule.weights = Eigen::Map<EigMat<Float>> (
            ((std::vector<Float>) order_data["weights"]).data(), 1, num_nodes
            ).eval();

        rules.push_back(rule);
    }

    return rules;

};


/**
* @brief Compares two complex numbers within a given tolerance based on a given rule.
* @param[in] val - Value to check.
* @param[in] val_ref - Reference value.
* @param[in] tol - Tolerance (optional).
* @param[in] mode - Comparison mode:
*   - 1: Compare real and imaginary parts separately (default).
*   - 2: Compare magnitude and phase separately.
*   - 3: Compare the absolute value.
* @return `true` if the two values are within the specified tolerance, `false` otherwise.
*/
inline bool compare_with_tol(
    const Complex val,
    const Complex val_ref,
    const Float tol = 1e-3,
    const uint8_t mode = 1
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


/**
* @}
*/

}

#endif
